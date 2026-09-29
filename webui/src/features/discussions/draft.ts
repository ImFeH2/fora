import { type DBSchema, openDB } from "idb";
import { useEffect, useState, useSyncExternalStore } from "react";
import { BackendError, backend, type UploadRecord } from "@/lib/backend";

export type DraftFile = {
  id: string;
  file: File;
  clientId: string;
  upload?: UploadRecord;
};
export type Submission = {
  id: string;
  body: string;
  bodyRevision: number;
  files: DraftFile[];
  phase: "uploading" | "sending";
  cancelRequested?: boolean;
};
export type Draft = {
  key: string;
  updatedAt: number;
  body: string;
  bodyRevision: number;
  files: DraftFile[];
  pending: Submission | null;
  voiceSubmission?: { id: string; submittedBody: string } | null;
};
interface DraftDatabase extends DBSchema {
  drafts: { key: string; value: Draft };
}
const database = () =>
  openDB<DraftDatabase>("huddol-drafts", 1, {
    upgrade(db) {
      db.createObjectStore("drafts", { keyPath: "key" });
    },
  });

let tab: Promise<string> | undefined;
function tabId() {
  if (tab) return tab;
  tab = new Promise<string>((resolve, reject) => {
    const claim = (id: string) => {
      void navigator.locks
        .request(
          `huddol-draft-tab:${id}`,
          { ifAvailable: true },
          async (lock) => {
            if (!lock) {
              claim(crypto.randomUUID());
              return;
            }
            sessionStorage.setItem("huddol.draft-tab", id);
            resolve(id);
            await new Promise<void>(() => {});
          },
        )
        .catch(reject);
    };
    claim(sessionStorage.getItem("huddol.draft-tab") ?? crypto.randomUUID());
  });
  return tab;
}

export function fileProblem(file: File, existing: DraftFile[]): string | null {
  const images = ["image/png", "image/jpeg", "image/webp"];
  if (existing.length >= 10) return "A message can contain at most 10 files.";
  if (file.size > 20 * 1024 * 1024)
    return `${file.name}: files must be at most 20 MiB.`;
  if (images.includes(file.type) && file.size > 5 * 1024 * 1024)
    return `${file.name}: images must be at most 5 MiB.`;
  if (
    existing.reduce((sum, item) => sum + item.file.size, file.size) >
    50 * 1024 * 1024
  )
    return "Message files must total at most 50 MiB.";
  if (
    !file.name.trim() ||
    file.name.length > 255 ||
    Array.from(file.name).some(
      (character) =>
        character.charCodeAt(0) < 32 ||
        character.charCodeAt(0) === 127 ||
        character === "/" ||
        character === "\\",
    )
  )
    return "Choose a filename of 1–255 characters without control characters.";
  return null;
}

export type DraftView = {
  draft: Draft;
  busy: boolean;
  submissionResult: { id: string; state: "sent" | "cancelled" } | null;
  saving: boolean;
  storageError: string | null;
  error: string | null;
};
export type SendDraft = (
  body: string,
  attachments: string[],
  id: string,
) => Promise<boolean | "cancelled">;

export class DraftController {
  #view: DraftView;
  #listeners = new Set<() => void>();
  #writes: Promise<unknown> = Promise.resolve();
  #abort: AbortController | null = null;
  #writeNumber = 0;
  #retryCancellationAfterBusy = false;
  #cancellationRecovery: Promise<void> | null = null;

  constructor(
    readonly discussionId: number,
    draft: Draft,
  ) {
    this.#view = {
      draft,
      busy: false,
      submissionResult: null,
      saving: false,
      storageError: null,
      error: null,
    };
  }

  snapshot = () => this.#view;
  subscribe = (listener: () => void) => {
    this.#listeners.add(listener);
    return () => {
      this.#listeners.delete(listener);
    };
  };
  #notify(update: Partial<DraftView>) {
    this.#view = { ...this.#view, ...update };
    for (const listener of this.#listeners) listener();
  }

  #persist(draft: Draft): Promise<boolean> {
    const number = ++this.#writeNumber;
    draft = { ...draft, updatedAt: Date.now() };
    this.#notify({ draft, saving: true });
    const operation = this.#writes.then(async () => {
      try {
        const db = await database();
        try {
          await db.put("drafts", draft);
        } finally {
          db.close();
        }
        if (number === this.#writeNumber)
          this.#notify({ saving: false, storageError: null });
        return true;
      } catch (error) {
        this.#notify({
          saving: false,
          storageError: `Draft not saved: ${error instanceof Error ? error.message : String(error)}`,
        });
        return false;
      }
    });
    this.#writes = operation;
    return operation;
  }

  setBody = (body: string) => {
    const draft = this.#view.draft;
    void this.#persist({
      ...draft,
      body,
      bodyRevision: draft.bodyRevision + 1,
      voiceSubmission: null,
    });
  };

  setVoiceBody = (body: string, voiceSubmission: Draft["voiceSubmission"]) => {
    const draft = this.#view.draft;
    if (voiceSubmission && !body.startsWith(voiceSubmission.submittedBody))
      throw new Error("Voice draft must retain its pending submission prefix");
    void this.#persist({
      ...draft,
      body,
      bodyRevision: draft.bodyRevision + 1,
      voiceSubmission: voiceSubmission ?? null,
    });
  };

  addFiles = (incoming: File[]) => {
    const files = [...this.#view.draft.files];
    const errors = [];
    for (const file of incoming) {
      const problem = fileProblem(file, files);
      if (problem) errors.push(problem);
      else
        files.push({
          id: crypto.randomUUID(),
          clientId: crypto.randomUUID(),
          file,
        });
    }
    this.#notify({ error: errors.length ? errors.join(" ") : null });
    void this.#persist({ ...this.#view.draft, files });
  };

  removeFile = (id: string) => {
    void this.#persist({
      ...this.#view.draft,
      files: this.#view.draft.files.filter((item) => item.id !== id),
    });
  };

  saveAgain = () => {
    void this.#persist(this.#view.draft);
  };
  cancel = () => {
    const pending = this.#view.draft.pending;
    if (pending?.phase === "uploading" && !pending.cancelRequested) {
      const requested = { ...pending, cancelRequested: true };
      const draft = { ...this.#view.draft, pending: requested };
      this.#notify({ draft });
      void this.#persist(draft);
    }
    this.#abort?.abort();
  };

  async #pending(pending: Submission): Promise<void> {
    if (!(await this.#persist({ ...this.#view.draft, pending })))
      throw new Error("Save the draft before sending.");
  }

  async #finish(submission: Submission) {
    this.#notify({
      submissionResult: { id: submission.id, state: "sent" },
    });
    const current = this.#view.draft;
    const sent = new Set(submission.files.map((file) => file.id));
    const voiceSubmission =
      current.voiceSubmission?.id === submission.id
        ? current.voiceSubmission
        : null;
    const body = voiceSubmission
      ? current.body.startsWith(voiceSubmission.submittedBody)
        ? current.body.slice(voiceSubmission.submittedBody.length)
        : current.body
      : current.bodyRevision === submission.bodyRevision
        ? ""
        : current.body;
    const saved = await this.#persist({
      ...current,
      body,
      files: current.files.filter((file) => !sent.has(file.id)),
      pending: null,
      voiceSubmission: null,
    });
    if (!saved) {
      this.#notify({
        draft: { ...this.#view.draft, pending: submission },
        storageError: null,
        error: "Could not update draft",
      });
      throw new Error("Could not update draft");
    }
    this.#notify({
      error: null,
      submissionResult: { id: submission.id, state: "sent" },
    });
  }

  #requestedCancellation(submission: Submission): Submission {
    const current = this.#view.draft.pending;
    return {
      ...(current?.id === submission.id ? current : submission),
      cancelRequested: true,
    };
  }

  async #resolveCancellationUploads(
    submission: Submission,
  ): Promise<Submission> {
    let resolved = submission;
    for (let index = 0; index < resolved.files.length; index++) {
      const item = resolved.files[index];
      if (item.upload) continue;
      const upload = await backend.createUpload(
        this.discussionId,
        item.clientId,
        item.file,
      );
      resolved = {
        ...resolved,
        files: resolved.files.map((entry, at) =>
          at === index ? { ...entry, upload } : entry,
        ),
      };
      const draft = { ...this.#view.draft, pending: resolved };
      this.#notify({ draft });
      if (!(await this.#persist(draft)))
        throw new Error("Could not save cancellation state");
    }
    return resolved;
  }

  #cancellationFailure(error: unknown): string | null {
    if (error instanceof BackendError && error.transport) return null;
    return `Could not finish cancelling send: ${error instanceof Error ? error.message : String(error)}`;
  }

  #scheduleCancellationRecovery() {
    if (
      !this.#retryCancellationAfterBusy ||
      !this.#view.draft.pending?.cancelRequested
    )
      return;
    this.#retryCancellationAfterBusy = false;
    void this.#completeCancellation();
  }

  async #completeCancellation(): Promise<void> {
    const pending = this.#view.draft.pending;
    if (!pending?.cancelRequested) return;
    if (this.#view.busy) {
      this.#retryCancellationAfterBusy = true;
      return;
    }
    if (this.#cancellationRecovery) {
      await this.#cancellationRecovery;
      return;
    }
    const operation = (async () => {
      this.#notify({ busy: true });
      try {
        await this.#cancelled(pending);
      } catch (error) {
        if (!(error instanceof BackendError && error.transport))
          this.#notify({ error: this.#cancellationFailure(error) });
      } finally {
        this.#notify({ busy: false });
      }
    })();
    this.#cancellationRecovery = operation;
    try {
      await operation;
    } finally {
      if (this.#cancellationRecovery === operation)
        this.#cancellationRecovery = null;
      this.#scheduleCancellationRecovery();
    }
  }

  connectionRestored = () => this.#completeCancellation();

  async #cancelled(submission: Submission) {
    let requested = this.#requestedCancellation(submission);
    const marked = { ...this.#view.draft, pending: requested };
    this.#notify({
      draft: marked,
      submissionResult: { id: requested.id, state: "cancelled" },
    });
    await this.#persist(marked);
    try {
      requested = await this.#resolveCancellationUploads(requested);
      const ids = requested.files.flatMap((item) =>
        item.upload ? [item.upload.id] : [],
      );
      if (ids.length) await backend.cancelUploads(ids);
    } catch (error) {
      const current = this.#view.draft.pending;
      this.#notify({
        draft: {
          ...this.#view.draft,
          pending: current?.id === requested.id ? current : requested,
        },
        error: this.#cancellationFailure(error),
      });
      throw error;
    }
    const current = this.#view.draft;
    const cancelled = new Set(requested.files.map((item) => item.id));
    const saved = await this.#persist({
      ...current,
      files: current.files.map((item) =>
        cancelled.has(item.id)
          ? { ...item, clientId: crypto.randomUUID(), upload: undefined }
          : item,
      ),
      pending: null,
      voiceSubmission:
        current.voiceSubmission?.id === requested.id
          ? null
          : current.voiceSubmission,
    });
    if (!saved) {
      const reason = this.#view.storageError ?? "Draft could not be saved";
      this.#notify({
        draft: { ...this.#view.draft, pending: requested },
        storageError: null,
        error: `Could not finish cancelling send: ${reason}`,
      });
      throw new Error(`Could not finish cancelling send: ${reason}`);
    }
    this.#notify({
      error: null,
      submissionResult: { id: requested.id, state: "cancelled" },
    });
  }

  async checkResult() {
    const pending = this.#view.draft.pending;
    if (pending?.cancelRequested) {
      await this.#completeCancellation();
      return;
    }
    if (pending?.phase !== "sending" || this.#view.busy) return;
    this.#notify({ busy: true });
    try {
      const known =
        this.#view.submissionResult?.id === pending.id
          ? this.#view.submissionResult.state
          : null;
      if (known === "sent") {
        await this.#finish(pending);
        return;
      }
      if (known === "cancelled") {
        await this.#cancelled(pending);
        return;
      }
      const result = await backend.sendStatus(this.discussionId, pending.id);
      if (result.state === "sent") await this.#finish(pending);
      else if (result.state === "cancelled") await this.#cancelled(pending);
      else
        this.#notify({
          error:
            "Send result is unconfirmed. Retry or cancel the saved send attempt.",
        });
    } catch (error) {
      if (
        !(
          error instanceof BackendError &&
          error.transport &&
          this.#view.draft.pending?.cancelRequested
        )
      )
        this.#notify({
          error: `Send result is unconfirmed. ${error instanceof Error ? error.message : String(error)}`,
        });
    } finally {
      this.#notify({ busy: false });
    }
  }

  async discardAttempt() {
    const pending = this.#view.draft.pending;
    if (pending?.cancelRequested) {
      await this.#completeCancellation();
      return;
    }
    if (!pending || this.#view.busy) return;
    this.#notify({ busy: true });
    try {
      if (pending.phase === "sending") {
        const result = await backend.cancelSend(this.discussionId, pending.id);
        if (result.state === "sent") {
          await this.#finish(pending);
          return;
        }
      }
      await this.#cancelled(pending);
    } catch (error) {
      if (!(error instanceof BackendError && error.transport))
        this.#notify({
          error: error instanceof Error ? error.message : String(error),
        });
    } finally {
      this.#notify({ busy: false });
      this.#scheduleCancellationRecovery();
    }
  }

  async send(onSend: SendDraft) {
    if (this.#view.busy) return;
    const draft = this.#view.draft;
    if (draft.pending?.cancelRequested) {
      await this.#completeCancellation();
      return;
    }
    if (!draft.pending && !draft.body.trim() && !draft.files.length) return;
    this.#notify({
      busy: true,
      error: null,
      submissionResult: null,
    });
    const abort = new AbortController();
    this.#abort = abort;
    let submission: Submission = draft.pending ?? {
      id: crypto.randomUUID(),
      body: draft.body.trim(),
      bodyRevision: draft.bodyRevision,
      files: draft.files.map((item) => ({ ...item })),
      phase: "uploading",
    };
    try {
      if (submission.phase === "sending") {
        const result = await backend.sendStatus(
          this.discussionId,
          submission.id,
        );
        if (result.state === "sent") {
          await this.#finish(submission);
          return;
        }
        if (result.state === "cancelled") {
          await this.#cancelled(submission);
          return;
        }
      }
      await this.#pending(submission);
      for (
        let index = 0;
        submission.phase === "uploading" && index < submission.files.length;
        index++
      ) {
        abort.signal.throwIfAborted();
        let item = submission.files[index];
        let upload = item.upload
          ? (await backend.uploadStatus([item.upload.id]))[0]
          : await backend.createUpload(
              this.discussionId,
              item.clientId,
              item.file,
            );
        if (
          upload.state !== "ready" ||
          upload.expires_at * 1000 <= Date.now()
        ) {
          if (
            upload.state !== "reserved" ||
            upload.expires_at * 1000 <= Date.now()
          ) {
            await backend.cancelUploads([upload.id]);
            abort.signal.throwIfAborted();
            item = { ...item, clientId: crypto.randomUUID() };
            upload = await backend.createUpload(
              this.discussionId,
              item.clientId,
              item.file,
            );
          }
          item = { ...item, upload };
          submission = {
            ...submission,
            files: submission.files.map((entry, at) =>
              at === index ? item : entry,
            ),
          };
          await this.#pending(submission);
          abort.signal.throwIfAborted();
          await backend.uploadFile(
            upload.id,
            item.file,
            abort.signal,
            () => {},
          );
          upload = { ...upload, state: "ready" };
        }
        item = { ...item, upload };
        submission = {
          ...submission,
          files: submission.files.map((entry, at) =>
            at === index ? item : entry,
          ),
        };
        await this.#pending(submission);
      }
      abort.signal.throwIfAborted();
      submission = { ...submission, phase: "sending" };
      await this.#pending(submission);
      const ids = submission.files.map((item) => {
        if (!item.upload) throw new Error("An uploaded file is missing its ID");
        return item.upload.id;
      });
      const result = await onSend(submission.body, ids, submission.id);
      if (result === "cancelled") {
        await this.#cancelled(submission);
        return;
      }
      if (!result)
        throw new Error(
          "Send result is unconfirmed. Check the saved send attempt before retrying.",
        );
      await this.#finish(submission);
    } catch (error) {
      if (abort.signal.aborted) {
        try {
          await this.#cancelled(submission);
        } catch (cancellationError) {
          if (
            !(
              cancellationError instanceof BackendError &&
              cancellationError.transport
            )
          )
            this.#notify({
              error:
                cancellationError instanceof Error
                  ? cancellationError.message
                  : String(cancellationError),
            });
        }
      } else {
        this.#notify({
          error: error instanceof Error ? error.message : String(error),
        });
      }
    } finally {
      this.#abort = null;
      this.#notify({ busy: false });
      this.#scheduleCancellationRecovery();
    }
  }
}

const controllers = new Map<number, Promise<DraftController>>();
function controllerFor(discussion: number) {
  const existing = controllers.get(discussion);
  if (existing) return existing;
  const pending = (async () => {
    const organization = await backend.organization();
    const tab = await tabId();
    const prefix = `${organization.uuid}:${organization.human_id}:${discussion}:`;
    const key = `${prefix}${tab}`;
    const db = await database();
    let stored: Draft | undefined;
    try {
      stored = await db.get("drafts", key);
      if (!stored) {
        const candidates = (await db.getAll("drafts"))
          .filter(
            (entry) =>
              entry.key.startsWith(prefix) &&
              (entry.body || entry.files.length || entry.pending),
          )
          .sort((a, b) => b.updatedAt - a.updatedAt);
        for (const candidate of candidates) {
          stored = await navigator.locks.request(
            `huddol-draft-tab:${candidate.key.slice(prefix.length)}`,
            { ifAvailable: true },
            async (lock) => {
              if (!lock) return undefined;
              const transaction = db.transaction("drafts", "readwrite");
              const current = await transaction.store.get(candidate.key);
              if (!current) {
                await transaction.done;
                return undefined;
              }
              const recovered = { ...current, key };
              await transaction.store.put(recovered);
              await transaction.store.delete(candidate.key);
              await transaction.done;
              return recovered;
            },
          );
          if (stored) break;
        }
      }
    } finally {
      db.close();
    }
    const controller = new DraftController(
      discussion,
      stored ?? {
        key,
        updatedAt: Date.now(),
        body: "",
        bodyRevision: 0,
        files: [],
        pending: null,
      },
    );
    void controller.checkResult();
    return controller;
  })();
  controllers.set(discussion, pending);
  void pending.catch(() => {
    if (controllers.get(discussion) === pending) controllers.delete(discussion);
  });
  return pending;
}

export function useDraft(discussion: number) {
  const [controller, setController] = useState<DraftController | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retryNumber, setRetryNumber] = useState(0);
  useEffect(() => {
    let live = true;
    let loading: Promise<void> | null = null;
    let retryAfterLoading = false;
    let current: DraftController | null = null;
    const load = () => {
      if (!live || current) return;
      if (loading) {
        retryAfterLoading = true;
        return;
      }
      setError(null);
      const attempt = controllerFor(discussion).then(
        (value) => {
          if (!live) return;
          current = value;
          setController(value);
          setError(null);
        },
        (failure: unknown) => {
          if (!live) return;
          setError(
            failure instanceof BackendError && failure.transport
              ? null
              : failure instanceof Error
                ? failure.message
                : String(failure),
          );
        },
      );
      loading = attempt;
      void attempt.finally(() => {
        if (loading !== attempt) return;
        loading = null;
        if (retryAfterLoading) {
          retryAfterLoading = false;
          load();
        }
      });
    };
    const off = backend.onEvent((event) => {
      if (event.type !== "connection.restored") return;
      if (current) {
        void current.connectionRestored();
        return;
      }
      setError(null);
      load();
    });
    load();
    return () => {
      live = false;
      off();
    };
  }, [discussion, retryNumber]);
  const view = useSyncExternalStore(
    controller?.subscribe ?? (() => () => {}),
    controller?.snapshot ?? (() => null),
    () => null,
  );
  return {
    controller,
    view,
    error,
    retry: () => setRetryNumber((value) => value + 1),
  };
}
