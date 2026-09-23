import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { backend } from "@/lib/backend";
import { type VoiceEvent, VoiceRecording } from "@/lib/voice";

class TestSocket {
  static OPEN = 1;
  static current: TestSocket;
  readyState = 1;
  bufferedAmount = 0;
  sent: unknown[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  close = vi.fn();
  constructor() {
    TestSocket.current = this;
  }
  send(data: unknown) {
    this.sent.push(typeof data === "string" ? JSON.parse(data) : data);
  }
  receive(message: unknown) {
    this.onmessage?.({ data: JSON.stringify(message) });
  }
}

class TestContext {
  static current: TestContext;
  sampleRate = 48_000;
  state = "running";
  destination = {};
  audioWorklet = { addModule: vi.fn().mockResolvedValue(undefined) };
  source = { connect: vi.fn(), disconnect: vi.fn() };
  createMediaStreamSource = vi.fn(() => this.source);
  resume = vi.fn().mockResolvedValue(undefined);
  close = vi.fn().mockResolvedValue(undefined);
  constructor() {
    TestContext.current = this;
  }
}

class TestProcessor {
  static current: TestProcessor;
  port = {
    onmessage: null as ((event: { data: unknown }) => void) | null,
    postMessage: vi.fn(),
    close: vi.fn(),
  };
  connect = vi.fn();
  disconnect = vi.fn();
  constructor() {
    TestProcessor.current = this;
  }
  receive(data: unknown) {
    this.port.onmessage?.({ data });
  }
}

const track = { stop: vi.fn(), onended: null as (() => void) | null };
const stream = { getTracks: () => [track], getAudioTracks: () => [track] };
let getUserMedia = vi.fn();
let events: VoiceEvent[];
let recording: VoiceRecording;

beforeEach(() => {
  vi.useFakeTimers();
  vi.spyOn(backend, "voiceUrl").mockResolvedValue(
    "ws://localhost/voice?token=test",
  );
  getUserMedia = vi.fn().mockResolvedValue(stream);
  vi.stubGlobal("navigator", { mediaDevices: { getUserMedia } });
  vi.stubGlobal("WebSocket", TestSocket);
  vi.stubGlobal("AudioContext", TestContext);
  vi.stubGlobal("AudioWorkletNode", TestProcessor);
  events = [];
  recording = new VoiceRecording((event) => events.push(event));
  track.stop.mockClear();
});

afterEach(() => {
  recording.cancel();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

async function ready() {
  await recording.start();
  TestSocket.current.onopen?.();
  TestSocket.current.receive({ type: "ready", mode: "local" });
  await vi.advanceTimersByTimeAsync(0);
  expect(recording.state).toBe("recording");
}

describe("VoiceRecording", () => {
  it("declares the actual sample rate and sends tail PCM before stop", async () => {
    await ready();
    const socket = TestSocket.current;
    const processor = TestProcessor.current;
    expect(socket.sent).toEqual([
      { type: "start", sample_rate: 48_000, channels: 1, format: "f32le" },
    ]);
    recording.stop();
    expect(processor.port.postMessage).toHaveBeenCalledWith("stop");
    expect(socket.sent).toHaveLength(1);
    const samples = new Float32Array([0.25, -0.25]);
    processor.receive({ type: "audio", samples, level: 0.25 });
    processor.receive({ type: "stopped" });
    expect(socket.sent).toEqual([
      { type: "start", sample_rate: 48_000, channels: 1, format: "f32le" },
      samples,
      { type: "stop" },
    ]);
    expect(track.stop).toHaveBeenCalledTimes(1);
    expect(TestContext.current.close).toHaveBeenCalledTimes(1);
    socket.receive({ type: "finished", text: "final text" });
    expect(events).toContainEqual({ type: "transcript", text: "final text" });
    expect(recording.state).toBe("closed");
    expect(vi.getTimerCount()).toBe(0);
  });

  it("releases a microphone granted after cancellation", async () => {
    let grant!: (value: typeof stream) => void;
    getUserMedia.mockReturnValue(
      new Promise((resolve) => {
        grant = resolve;
      }),
    );
    const started = recording.start();
    recording.cancel();
    grant(stream);
    await started;
    expect(track.stop).toHaveBeenCalledTimes(1);
    expect(backend.voiceUrl).not.toHaveBeenCalled();
    expect(recording.state).toBe("closed");
  });

  it("discards late transcripts after cancellation", async () => {
    await ready();
    recording.cancel();
    TestSocket.current.receive({ type: "transcript", text: "late" });
    expect(events.filter((event) => event.type === "transcript")).toEqual([]);
    expect(TestSocket.current.close).toHaveBeenCalledTimes(1);
    expect(track.stop).toHaveBeenCalledTimes(1);
  });

  it("closes audio resources when cancelled while loading the worklet", async () => {
    let resolveModule!: () => void;
    const moduleLoaded = new Promise<void>((resolve) => {
      resolveModule = resolve;
    });
    class PendingContext extends TestContext {
      constructor() {
        super();
        this.audioWorklet.addModule = vi.fn().mockReturnValue(moduleLoaded);
      }
    }
    vi.stubGlobal("AudioContext", PendingContext);
    const started = recording.start();
    await Promise.resolve();
    await Promise.resolve();
    expect(TestContext.current).toBeDefined();
    recording.cancel();
    resolveModule();
    await started;
    expect(TestContext.current.close).toHaveBeenCalledTimes(1);
    expect(track.stop).toHaveBeenCalledTimes(1);
  });

  it("keeps emitted text and releases resources on server capacity errors", async () => {
    await ready();
    TestSocket.current.receive({ type: "transcript", text: "retained" });
    TestSocket.current.receive({
      type: "error",
      code: "voice_capacity",
      message: "Audio capacity reached",
    });
    expect(events.filter((event) => event.type === "transcript")).toEqual([
      { type: "transcript", text: "retained" },
    ]);
    expect(events).toContainEqual({
      type: "error",
      message: "Audio capacity reached",
    });
    expect(track.stop).toHaveBeenCalledTimes(1);
    expect(recording.state).toBe("closed");
  });

  it("ends capture when outgoing audio exceeds thirty seconds", async () => {
    await ready();
    TestSocket.current.bufferedAmount = 48_000 * 4 * 30;
    TestProcessor.current.receive({
      type: "audio",
      samples: new Float32Array([1]),
      level: 1,
    });
    expect(recording.state).toBe("closed");
    expect(TestSocket.current.sent).toHaveLength(1);
    expect(track.stop).toHaveBeenCalledTimes(1);
  });

  it("reports permission failures and clears the start deadline", async () => {
    getUserMedia.mockRejectedValue(new Error("Microphone permission denied"));
    await recording.start();
    expect(events).toContainEqual({
      type: "error",
      message: "Microphone permission denied",
    });
    expect(recording.state).toBe("closed");
    expect(vi.getTimerCount()).toBe(0);
  });

  it("ends a silent voice handshake and releases the microphone", async () => {
    await recording.start();
    await vi.advanceTimersByTimeAsync(60_000);
    expect(recording.state).toBe("closed");
    expect(track.stop).toHaveBeenCalledTimes(1);
    expect(TestSocket.current.close).toHaveBeenCalledTimes(1);
  });
});
