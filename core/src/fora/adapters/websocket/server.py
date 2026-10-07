from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import secrets
import socket
import sys
import threading
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import quote, unquote

from aiohttp import WSMsgType, web

from fora.adapters.jsonl.protocol import Dispatcher, RequestScope, encode, parse
from fora.adapters.websocket.access_error import ACCESS_ERROR_PAGE
from fora.core.attachment import Upload
from fora.core.errors import DomainError
from fora.services.uploads import Uploads

SocketHandler = Callable[[web.WebSocketResponse], Awaitable[None]]


BROADCAST_FRAMES = 256
BROADCAST_BYTES = 16 * 1024 * 1024
CLOSE_TIMEOUT = 5


def payload_size(value: Any, available: int) -> int:
    size = sys.getsizeof(value)
    if size > available:
        return size
    if isinstance(value, dict):
        for key, item in value.items():
            size += payload_size(key, available - size)
            if size > available:
                return size
            size += payload_size(item, available - size)
            if size > available:
                return size
    elif isinstance(value, (list, tuple)):
        for item in value:
            size += payload_size(item, available - size)
            if size > available:
                return size
    return size


@dataclass(eq=False)
class ControlRequest:
    identifier: int | None
    scope: RequestScope
    business: asyncio.Event
    task: asyncio.Task[None] | None = None
    thread: Future[None] | None = None


@dataclass
class OutgoingFrame:
    text: str
    cost: int
    completion: asyncio.Future[None] | None
    on_sent: Callable[[], None] | None = None


class ControlOutbox:
    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.lock = threading.Condition()
        self.encoder = threading.Lock()
        self.items: deque[OutgoingFrame] = deque()
        self.responses: set[asyncio.Future[None]] = set()
        self.available = asyncio.Event()
        self.closing = asyncio.Event()
        self.closed = False
        self.wake_pending = False
        self.frames = 0
        self.used = 0
        self.producers = 0

    def _wake_locked(self) -> None:
        if not self.wake_pending:
            self.wake_pending = True
            self.loop.call_soon_threadsafe(self._wake)

    def _wake(self) -> None:
        with self.lock:
            self.wake_pending = False
            closed = self.closed
        self.available.set()
        if closed:
            self.closing.set()

    def fail(self) -> None:
        with self.lock:
            self.closed = True
            self._wake_locked()

    def register_response(self) -> asyncio.Future[None]:
        result = self.loop.create_future()
        self.responses.add(result)
        return result

    def complete(self, result: asyncio.Future[None]) -> None:
        if not result.done():
            result.set_result(None)
        self.responses.discard(result)

    def broadcast(self, payload: dict[str, Any]) -> None:
        self.put(payload)

    def put(
        self,
        payload: dict[str, Any],
        completion: asyncio.Future[None] | None = None,
        on_sent: Callable[[], None] | None = None,
    ) -> None:
        reserved = 0
        counted = False
        text: str | None = None
        data: bytes | None = None
        frame: OutgoingFrame | None = None
        with self.lock:
            if self.closed:
                return
            self.producers += 1
        try:
            if completion is None:
                with self.lock:
                    reserved = payload_size(payload, BROADCAST_BYTES - self.used)
                    if (
                        self.frames >= BROADCAST_FRAMES
                        or self.used + reserved > BROADCAST_BYTES
                    ):
                        reserved = 0
                        self.closed = True
                        self._wake_locked()
                        return
                    self.used += reserved
                    self.frames += 1
                    counted = True
            with self.encoder:
                with self.lock:
                    if self.closed:
                        return
                text = encode(payload)
                data = text.encode("utf-8")
                cost = sys.getsizeof(text) + 2 * sys.getsizeof(data)
                data = None
                payload = {}
                with self.lock:
                    if self.closed:
                        return
                    if (
                        completion is None
                        and self.used - reserved + cost > BROADCAST_BYTES
                    ):
                        self.closed = True
                        self._wake_locked()
                        return
                    frame = OutgoingFrame(text, cost, completion, on_sent)
                    text = None
                    if completion is None:
                        self.used += cost - reserved
                    self.items.append(frame)
                    frame = None
                    counted = False
                    reserved = 0
                    self._wake_locked()
        except Exception as error:
            self.fail()
            log.exception(
                "Control connection encoding failed: %s",
                type(error).__name__,
                exc_info=False,
            )
        finally:
            payload = {}
            text = None
            data = None
            frame = None
            with self.lock:
                if counted:
                    self.used -= reserved
                    self.frames -= 1
                self.producers -= 1
                self.lock.notify_all()

    async def take(self) -> OutgoingFrame | None:
        while True:
            with self.lock:
                if self.closed:
                    return None
                if self.items:
                    return self.items.popleft()
                self.available.clear()
            await self.available.wait()

    def release(self, frame: OutgoingFrame) -> None:
        if frame.completion is None:
            with self.lock:
                self.frames -= 1
                self.used -= frame.cost

    async def send(self, connection: web.WebSocketResponse) -> None:
        while (frame := await self.take()) is not None:
            try:
                if frame.on_sent is None:
                    await connection.send_str(frame.text)
                else:
                    sending = asyncio.Task(
                        connection.send_str(frame.text),
                        loop=self.loop,
                        eager_start=True,
                    )
                    try:
                        if sending.done():
                            sending.result()
                        frame.on_sent()
                        await sending
                    finally:
                        if not sending.done():
                            sending.cancel()
                            await asyncio.gather(sending, return_exceptions=True)
                if frame.completion is not None:
                    self.complete(frame.completion)
            finally:
                frame.text = ""
                self.release(frame)
                del frame

    def close(self) -> None:
        with self.lock:
            self.closed = True
            while self.items:
                frame = self.items.popleft()
                frame.text = ""
                self.release(frame)
                del frame
        for result in self.responses:
            if not result.done():
                result.set_exception(ConnectionError("Control connection closed"))
        self.responses.clear()
        self.available.set()
        self.closing.set()

    def wait_producers(self) -> None:
        with self.lock:
            self.lock.wait_for(lambda: self.producers == 0)


async def worker[T](operation: Callable[..., T], *args: Any) -> T:
    task = asyncio.create_task(asyncio.to_thread(operation, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


@asynccontextmanager
async def resource[T](
    operation: Callable[..., T],
    release: Callable[[T], None],
    *args: Any,
) -> AsyncIterator[T]:
    task = asyncio.create_task(asyncio.to_thread(operation, *args))
    try:
        result = await asyncio.shield(task)
        yield result
    finally:
        result = await task
        await worker(release, result)


HOST = "127.0.0.1"
CONTENT_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".ico": "image/x-icon",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".map": "application/json; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".ttf": "font/ttf",
    ".wasm": "application/wasm",
    ".webp": "image/webp",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
}

log = logging.getLogger(__name__)


def webui_directory(override: str | None = None) -> Path | None:
    if override:
        return Path(override).expanduser().resolve()
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        return Path(bundle) / "webui"
    return None


def static_response(directory: Path, path: str) -> web.Response:
    if not directory.is_dir():
        return web.Response(status=503, text="Frontend not built")
    relative = unquote(path).lstrip("/")
    target = directory / "index.html"
    if "." in relative:
        target = (directory / relative).resolve()
        if not target.is_relative_to(directory.resolve()) or not target.is_file():
            return web.Response(status=404, text="Not found")
    if not target.is_file():
        return web.Response(status=503, text="Frontend not built")
    content_type = CONTENT_TYPES.get(target.suffix.lower(), "application/octet-stream")
    return web.Response(
        headers={"Content-Type": content_type}, body=target.read_bytes()
    )


class WebServer:
    def __init__(
        self,
        dispatcher: Dispatcher,
        token: str,
        directory: Path | None,
        port: int = 0,
        *,
        uploads: Uploads | None = None,
    ) -> None:
        self._dispatcher = dispatcher
        self._token = token
        self._directory = directory
        self._uploads = uploads
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.bind((HOST, port))
        self._socket.listen(128)
        self._socket.setblocking(False)
        self._port = int(self._socket.getsockname()[1])
        self._ready: Future[None] = Future()
        self._finished: Future[None] = Future()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._connections: set[web.WebSocketResponse] = set()
        self._control_transports: dict[web.WebSocketResponse, asyncio.Transport] = {}
        self._handlers: set[asyncio.Task[Any]] = set()
        self._http: set[asyncio.Task[Any]] = set()
        self._requests: set[ControlRequest] = set()
        self._executor = ThreadPoolExecutor(
            max_workers=16, thread_name_prefix="fora-control"
        )
        self._app = web.Application(
            middlewares=[self._boundary], client_max_size=20 * 1024 * 1024
        )
        self._thread = threading.Thread(target=self._run, name="fora-web")
        self._origins = {
            f"http://127.0.0.1:{self._port}",
            f"http://localhost:{self._port}",
            "http://localhost:1420",
            "tauri://localhost",
            "http://tauri.localhost",
        }
        self.add_websocket_route("/ws", self._control, max_msg_size=2**20)
        self._app.router.add_put("/uploads/{upload_id}/content", self._upload)
        self._app.router.add_get(
            "/discussions/{discussion_id}/messages/{message_id}/attachments/{attachment_id}/content",
            self._download,
        )

    @property
    def port(self) -> int:
        return self._port

    def add_websocket_route(
        self, path: str, handler: SocketHandler, *, max_msg_size: int
    ) -> None:
        if self._thread.is_alive():
            raise RuntimeError("Register routes before starting the server")

        async def connected(request: web.Request) -> web.StreamResponse:
            self._authenticate(request, websocket=True)
            connection = web.WebSocketResponse(
                max_msg_size=max_msg_size,
                compress=False,
                heartbeat=20 if path == "/ws" else None,
            )
            if (
                path == "/ws"
                and request.headers.get("Upgrade", "").lower() != "websocket"
            ):
                return web.Response(status=204)
            await connection.prepare(request)
            task = asyncio.current_task()
            assert task is not None
            self._handlers.add(task)
            self._connections.add(connection)
            if path == "/ws":
                assert request.transport is not None
                self._control_transports[connection] = request.transport
            try:
                await handler(connection)
            finally:
                try:
                    await self._close_socket(connection)
                finally:
                    self._control_transports.pop(connection, None)
                    self._connections.discard(connection)
                    self._handlers.discard(task)
            return connection

        self._app.router.add_get(path, connected)

    def start(self) -> None:
        self._app.router.add_route("*", "/{path:.*}", self._static)
        self._thread.start()
        self._ready.result()

    def stop(self) -> None:
        assert self._loop is not None and self._stop is not None
        self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join()
        self._finished.result()

    def _run(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as error:
            log.exception("Web server failed")
            if not self._ready.done():
                self._ready.set_exception(error)
            self._finished.set_exception(error)
        else:
            self._finished.set_result(None)
        finally:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._socket.close()

    async def _serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        runner = web.AppRunner(self._app, access_log=None, shutdown_timeout=130)
        await runner.setup()
        site = web.SockSite(runner, self._socket)
        try:
            await site.start()
            async with asyncio.TaskGroup() as tasks:
                cleanup = tasks.create_task(self._cleanup())
                self._ready.set_result(None)
                try:
                    await self._stop.wait()
                finally:
                    await site.stop()
                    cleanup.cancel()
                    await asyncio.gather(
                        *(
                            self._close_socket(connection, code=1001)
                            for connection in tuple(self._connections)
                        )
                    )
                    handlers = tuple(self._handlers)
                    requests = tuple(self._http)
                    for task in requests:
                        task.cancel()
                    results = await asyncio.gather(
                        *handlers,
                        *requests,
                        return_exceptions=True,
                    )
                    errors = [
                        result
                        for result in results
                        if isinstance(result, BaseException)
                        and not isinstance(result, asyncio.CancelledError)
                    ]
                    if errors:
                        raise BaseExceptionGroup("Request shutdown failed", errors)
        finally:
            await runner.cleanup()

    async def _cleanup(self) -> None:
        while True:
            await asyncio.sleep(15)
            if self._uploads is not None:
                await worker(self._uploads.cleanup)

    def _authenticate(self, request: web.Request, *, websocket: bool = False) -> None:
        supplied = (
            request.query.get("token", "")
            if websocket
            else request.headers.get("Authorization", "").removeprefix("Bearer ")
        )
        if not supplied or not secrets.compare_digest(
            supplied.encode(), self._token.encode()
        ):
            if request.path == "/ws":
                raise web.HTTPUnauthorized(
                    text=ACCESS_ERROR_PAGE.decode(), content_type="text/html"
                )
            raise web.HTTPUnauthorized(text="Unauthorized")

    @web.middleware
    async def _boundary(
        self,
        request: web.Request,
        handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        origin = request.headers.get("Origin")
        if origin is not None and origin not in self._origins:
            if (
                request.path != "/ws"
                or request.headers.get("Upgrade", "").lower() == "websocket"
            ):
                raise web.HTTPForbidden(text="Origin not allowed")
            origin = None
        try:
            if request.method == "OPTIONS":
                response: web.StreamResponse = web.Response(status=204)
            else:
                response = await handler(request)
        except web.HTTPException as error:
            response = web.Response(
                status=error.status,
                reason=error.reason,
                headers=error.headers,
                body=error.body,
            )
        except DomainError as error:
            response = web.json_response(
                {"error": {"code": error.code, "message": str(error)}}, status=400
            )
        except TimeoutError:
            response = web.json_response(
                {
                    "error": {
                        "code": "upload_timeout",
                        "message": "Upload timed out; your draft is preserved",
                    }
                },
                status=408,
            )
        if request.path == "/ws" and not isinstance(response, web.WebSocketResponse):
            response.headers.update(
                {
                    "Cache-Control": "no-store",
                    "Referrer-Policy": "no-referrer",
                    "Vary": "Origin",
                }
            )
        if origin is not None:
            response.headers.update(
                {
                    "Access-Control-Allow-Origin": origin,
                    "Vary": "Origin",
                    "Access-Control-Allow-Methods": "GET, PUT, OPTIONS",
                    "Access-Control-Allow-Headers": "Authorization, Content-Type",
                }
            )
        return response

    async def _static(self, request: web.Request) -> web.Response:
        if request.method != "GET":
            raise web.HTTPMethodNotAllowed(request.method, ["GET"])
        if self._directory is None:
            return web.Response(status=404, text="Not found")
        return await worker(static_response, self._directory, request.path)

    async def _close_socket(
        self, connection: web.WebSocketResponse, *, code: int = 1000
    ) -> None:
        transport = self._control_transports.get(connection)
        if transport is None:
            await connection.close(code=code)
            return
        try:
            async with asyncio.timeout(CLOSE_TIMEOUT):
                await connection.close(code=code)
        except TimeoutError:
            pass
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise
        finally:
            transport.abort()

    async def _control(self, connection: web.WebSocketResponse) -> None:
        outbox = ControlOutbox()
        records: set[ControlRequest] = set()
        identifiers: dict[int, ControlRequest] = {}
        positions = asyncio.BoundedSemaphore(16)

        async def execute(
            record: ControlRequest,
            text: str,
            dependencies: tuple[asyncio.Event, ...],
            failure: tuple[str, str] | None,
        ) -> None:
            for dependency in dependencies:
                await dependency.wait()
            dependencies = ()
            async with positions:
                if outbox.closed:
                    raise asyncio.CancelledError
                completion = outbox.register_response()
                responded = False

                def delivered() -> None:
                    if (
                        record.identifier is not None
                        and identifiers.get(record.identifier) is record
                    ):
                        identifiers.pop(record.identifier, None)

                def response(payload: dict[str, Any]) -> None:
                    nonlocal responded
                    if responded:
                        raise RuntimeError("Request already responded")
                    responded = True
                    outbox.put(payload, completion, delivered)

                def operation() -> None:
                    if failure is None:
                        self._dispatcher.receive(text, response)
                    else:
                        request = parse(text)
                        assert request is not None
                        self._dispatcher._fail(request, response, *failure)

                try:
                    record.thread = self._executor.submit(
                        contextvars.copy_context().run, operation
                    )
                    thread = asyncio.wrap_future(record.thread)
                    cancelled = False
                    try:
                        while True:
                            try:
                                await asyncio.shield(thread)
                                break
                            except asyncio.CancelledError:
                                cancelled = True
                                if record.thread.cancel():
                                    break
                                if thread.done():
                                    thread.result()
                                    break
                    finally:
                        record.business.set()
                    if cancelled:
                        raise asyncio.CancelledError
                    if not responded:
                        outbox.complete(completion)
                    await asyncio.shield(completion)
                finally:
                    if not completion.done():
                        outbox.complete(completion)
                    if not completion.cancelled():
                        completion.exception()
                    record.thread = None

        def completed(record: ControlRequest, task: asyncio.Task[None]) -> None:
            record.business.set()
            records.discard(record)
            self._requests.discard(record)
            if (
                record.identifier is not None
                and identifiers.get(record.identifier) is record
            ):
                identifiers.pop(record.identifier, None)
            record.task = None
            if not task.cancelled() and (error := task.exception()) is not None:
                if not isinstance(error, ConnectionError):
                    log.error("Control request failed: %s", type(error).__name__)
                outbox.fail()

        async def receive() -> None:
            async for message in connection:
                if message.type not in (WSMsgType.TEXT, WSMsgType.BINARY):
                    if message.type == WSMsgType.ERROR:
                        return
                    continue
                text = (
                    message.data
                    if message.type == WSMsgType.TEXT
                    else message.data.decode("utf-8")
                )
                request = parse(text)
                identifier = request.id if request is not None else None
                if identifier is not None and identifier in identifiers:
                    await self._close_socket(connection, code=1002)
                    return
                scope = RequestScope()
                failure = None
                if request is not None:
                    try:
                        scope = self._dispatcher.scope(request)
                    except DomainError as error:
                        failure = (error.code, str(error))
                    except (TypeError, ValueError, KeyError) as error:
                        failure = ("invalid_params", f"{type(error).__name__}: {error}")
                dependencies = tuple(
                    previous.business
                    for previous in records
                    if not previous.business.is_set() and scope.follows(previous.scope)
                )
                record = ControlRequest(identifier, scope, asyncio.Event())
                records.add(record)
                self._requests.add(record)
                if identifier is not None:
                    identifiers[identifier] = record
                task = asyncio.create_task(execute(record, text, dependencies, failure))
                record.task = task
                task.add_done_callback(partial(completed, record))

        self._dispatcher.attach(outbox.broadcast)
        sender = asyncio.create_task(outbox.send(connection))
        receiver = asyncio.create_task(receive())
        closing = asyncio.create_task(outbox.closing.wait())
        try:
            finished, _ = await asyncio.wait(
                (sender, receiver, closing), return_when=asyncio.FIRST_COMPLETED
            )
            for task in finished:
                task.result()
        except ConnectionError:
            pass
        finally:
            self._dispatcher.detach(outbox.broadcast)
            outbox.close()
            sender.cancel()
            receiver.cancel()
            closing.cancel()
            requests = tuple(
                record.task for record in records if record.task is not None
            )
            for task in requests:
                task.cancel()

            async def finish() -> None:
                try:
                    await asyncio.gather(sender, closing, return_exceptions=True)
                    await self._close_socket(connection, code=1013)
                finally:
                    await asyncio.gather(receiver, *requests, return_exceptions=True)
                    await worker(outbox.wait_producers)

            cleanup = asyncio.create_task(finish())
            cancelled = False
            while True:
                try:
                    await asyncio.shield(cleanup)
                    break
                except asyncio.CancelledError:
                    cancelled = True
                    if cleanup.done():
                        cleanup.result()
                        break
            if cancelled:
                raise asyncio.CancelledError

    def _uploads_service(self) -> Uploads:
        if self._uploads is None:
            raise web.HTTPNotFound()
        return self._uploads

    async def _upload(self, request: web.Request) -> web.Response:
        self._authenticate(request)
        uploads = self._uploads_service()
        task = asyncio.current_task()
        assert task is not None
        self._http.add(task)

        def release(result: tuple[Upload, BinaryIO]) -> None:
            record, target = result
            try:
                target.close()
            finally:
                uploads.end(record)

        try:
            async with resource(
                uploads.begin,
                release,
                request.match_info["upload_id"],
                1,
            ) as (record, target):
                async with asyncio.timeout(120):
                    if (
                        request.content_length is not None
                        and request.content_length != record.size
                    ):
                        raise DomainError(
                            "upload_size_mismatch",
                            "The request size does not match the file",
                        )
                    received = 0
                    while True:
                        async with asyncio.timeout(30):
                            chunk = await request.content.read(64 * 1024)
                        if not chunk:
                            break
                        received += len(chunk)
                        if received > record.size:
                            raise DomainError(
                                "file_too_large",
                                "The request exceeds the declared file size",
                            )
                        await worker(target.write, chunk)
                    await worker(target.flush)
                    await worker(os.fsync, target.fileno())
                    await worker(target.close)
                    result = await worker(uploads.complete, record)
                    return web.json_response(asdict(result))
        finally:
            self._http.discard(task)

    async def _download(self, request: web.Request) -> web.StreamResponse:
        self._authenticate(request)
        uploads = self._uploads_service()
        task = asyncio.current_task()
        assert task is not None
        self._http.add(task)
        try:
            async with resource(
                uploads.read,
                lambda result: result[1].close(),
                int(request.match_info["discussion_id"]),
                int(request.match_info["message_id"]),
                request.match_info["attachment_id"],
                1,
            ) as (metadata, source):
                response = web.StreamResponse(
                    headers={
                        "Content-Type": metadata.media_type,
                        "Content-Length": str(metadata.size),
                        "Content-Disposition": f"attachment; filename*=UTF-8''{quote(metadata.name, safe='')}",
                        "Cache-Control": "no-store",
                        "X-Content-Type-Options": "nosniff",
                    }
                )
                origin = request.headers.get("Origin")
                if origin is not None:
                    response.headers["Access-Control-Allow-Origin"] = origin
                    response.headers["Vary"] = "Origin"
                await response.prepare(request)
                while chunk := await worker(source.read, 64 * 1024):
                    await response.write(chunk)
                await response.write_eof()
                return response
        finally:
            self._http.discard(task)
