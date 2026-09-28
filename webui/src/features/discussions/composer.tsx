import { clsx } from "clsx";
import {
  type CSSProperties,
  Fragment,
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Avatar, autoGrowHeight, Textarea, toast } from "@/components/ui/index";
import { DraftAttachments } from "@/features/discussions/attachments";
import { type SendDraft, useDraft } from "@/features/discussions/draft";
import {
  candidatesFor,
  completeMention,
  mentionQuery,
} from "@/features/mentions";
import { backend, type Member } from "@/lib/backend";
import { VoiceRecording, type VoiceState } from "@/lib/voice";

const MENU_LIMIT = 8;
const composerTheme = {
  "--composer-background": "oklch(14.5% 0 0)",
  "--composer-foreground": "oklch(98.5% 0 0)",
  "--composer-card": "oklch(20.5% 0 0)",
  "--composer-primary": "oklch(92.2% 0 0)",
  "--composer-primary-foreground": "oklch(20.5% 0 0)",
  "--composer-accent": "oklch(26.9% 0 0)",
  "--composer-muted-foreground": "oklch(70.8% 0 0)",
  "--composer-border": "rgb(255 255 255 / 0.1)",
  "--composer-ring": "oklch(55.6% 0 0)",
} as CSSProperties;

function ArrowUpIcon() {
  return (
    <svg
      width="12"
      height="12"
      viewBox="0 0 14 14"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M7 12V2M7 2L2.5 6.5M7 2L11.5 6.5"
        stroke="currentColor"
        strokeWidth="1.75"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function MicIcon() {
  return (
    <svg
      width="13"
      height="13"
      viewBox="0 0 14 14"
      fill="none"
      aria-hidden="true"
    >
      <rect
        x="5"
        y="1"
        width="4"
        height="7"
        rx="2"
        stroke="currentColor"
        strokeWidth="1.5"
      />
      <path
        d="M2.75 6.5V7a4.25 4.25 0 0 0 8.5 0v-.5M7 11.25V13"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  );
}

function PlusIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 14 14"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M7 2.5V11.5M2.5 7H11.5"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function composerFades(top: number, height: number, viewport: number) {
  return {
    top: Math.min(top / 20, 1),
    bottom: Math.min(Math.max(height - viewport - top - 16, 0) / 10, 1),
  };
}

export function composerMultiline(
  value: string,
  height: number,
  line: number,
  padding: number,
) {
  return (
    value.includes("\n") ||
    (value.length > 0 && height > Math.ceil(line + padding) + 1)
  );
}

export function composerHeight(expanded: boolean, inputHeight: number) {
  return expanded ? Math.max(116, inputHeight + 48) : 48;
}

export type ComposerKey = {
  key: string;
  shiftKey: boolean;
  ctrlKey: boolean;
  metaKey: boolean;
  isComposing?: boolean;
};

export type ComposerAction =
  | "send"
  | "newline"
  | "accept"
  | "dismiss"
  | "up"
  | "down";

export function composerKey(
  event: ComposerKey,
  suggesting: boolean,
): ComposerAction | null {
  if (event.isComposing) return null;
  if (suggesting) {
    if (event.key === "ArrowDown") return "down";
    if (event.key === "ArrowUp") return "up";
    if (event.key === "Escape") return "dismiss";
    if (event.key === "Tab") return "accept";
  }
  if (event.key !== "Enter") return null;
  if (event.ctrlKey || event.metaKey) return "send";
  if (event.shiftKey) return "newline";
  return suggesting ? "accept" : "send";
}

export function Composer({
  discussionId,
  members,
  memberIds,
  busy,
  placeholder,
  onSend,
  onHeightChange,
  onOpenVoiceSettings,
}: {
  discussionId: number;
  members: Member[];
  memberIds: ReadonlySet<number>;
  busy: boolean;
  placeholder: string;
  onSend: SendDraft;
  onHeightChange: (height: number) => void;
  onOpenVoiceSettings: () => void;
}) {
  const { controller, view, error: draftError } = useDraft(discussionId);
  const body = view?.draft.body ?? "";
  const files = view?.draft.files ?? [];
  const setBody = (value: string) => controller?.setBody(value);
  const sending = busy || !!view?.busy;
  const canSend =
    !!controller && !!(body.trim() || files.length || view?.draft.pending);
  const fileInput = useRef<HTMLInputElement>(null);
  const extra = useRef<HTMLDivElement>(null);
  const [extraHeight, setExtraHeight] = useState(0);
  useLayoutEffect(() => {
    const element = extra.current;
    if (!element) return;
    const update = () => setExtraHeight(element.getBoundingClientRect().height);
    const observer = new ResizeObserver(update);
    update();
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  const [voiceState, setVoiceState] = useState<VoiceState>("closed");
  const [voiceError, setVoiceError] = useState("");
  const [levels, setLevels] = useState([0, 0, 0, 0, 0]);
  const recording = useRef<VoiceRecording | null>(null);
  const voiceConfigRead = useRef<number | null>(null);
  const voiceConfigVersion = useRef(0);
  const mounted = useRef(true);
  const nextVoiceConfigVersion = useRef(0);
  const submissionUnsubscribe = useRef<(() => void) | null>(null);
  const recordingActive = voiceState !== "closed";
  const releaseSubmission = () => {
    submissionUnsubscribe.current?.();
    submissionUnsubscribe.current = null;
  };
  const cancelRecording = () => {
    const active = recording.current;
    recording.current = null;
    releaseSubmission();
    active?.cancel();
    setVoiceState("closed");
  };
  const currentDraft = useRef({ controller, discussionId });
  currentDraft.current = { controller, discussionId };
  useEffect(
    () => () => {
      const active = recording.current;
      recording.current = null;
      releaseSubmission();
      active?.cancel();
      voiceConfigRead.current = null;
      if (
        currentDraft.current.controller !== controller ||
        currentDraft.current.discussionId !== discussionId
      )
        setVoiceState("closed");
    },
    [controller, discussionId],
  );
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      voiceConfigRead.current = null;
    };
  }, []);
  const showVoiceSettingsToast = () => {
    toast({
      id: "voice-settings",
      tone: "info",
      title: "Set up voice transcription",
      description: "Add an API key in Voice settings to start voice input.",
      duration: null,
      action: {
        label: "Open Voice settings",
        onClick: onOpenVoiceSettings,
      },
    });
  };
  const startRecording = async () => {
    if (!controller || recording.current || voiceConfigRead.current !== null)
      return;
    const target = controller;
    const version = ++nextVoiceConfigVersion.current;
    voiceConfigVersion.current = version;
    voiceConfigRead.current = version;
    setVoiceError("");
    const isCurrentVoiceConfigRead = () =>
      mounted.current &&
      voiceConfigVersion.current === version &&
      voiceConfigRead.current === version &&
      currentDraft.current.controller === target &&
      currentDraft.current.discussionId === discussionId;
    const releaseVoiceConfigRead = () => {
      if (voiceConfigRead.current === version) voiceConfigRead.current = null;
    };
    let values: Record<string, unknown>;
    try {
      values = await backend.settings("voice");
    } catch (failure) {
      const current = isCurrentVoiceConfigRead();
      releaseVoiceConfigRead();
      if (current)
        setVoiceError(
          failure instanceof Error ? failure.message : String(failure),
        );
      return;
    }
    if (!isCurrentVoiceConfigRead()) {
      releaseVoiceConfigRead();
      return;
    }
    releaseVoiceConfigRead();
    if (typeof values.api_key_set !== "boolean") {
      setVoiceError("Voice settings did not include API key status.");
      return;
    }
    if (!values.api_key_set) {
      showVoiceSettingsToast();
      return;
    }
    releaseSubmission();
    const initial = target.snapshot();
    const revision = initial.draft.bodyRevision;
    const pending = initial.draft.pending;
    const voiceOrigin = initial.draft.voiceSubmission;
    const startsFromSubmission =
      pending?.bodyRevision === revision &&
      initial.draft.body.trim() === pending.body;
    const continuesSubmission = pending && voiceOrigin?.id === pending.id;
    const submission =
      startsFromSubmission || continuesSubmission ? pending : null;
    const submittedBody = submission
      ? startsFromSubmission
        ? initial.draft.body
        : (voiceOrigin?.submittedBody ?? "")
      : "";
    const voiceBase =
      submission && initial.draft.body.startsWith(submittedBody)
        ? initial.draft.body.slice(submittedBody.length)
        : initial.draft.body;
    const session: {
      submittedBody: string;
      voiceBase: string;
      bodyRevision: number;
      submissionId: string | null;
      submissionState: "pending" | "sent" | "cancelled" | null;
    } = {
      submittedBody,
      voiceBase,
      bodyRevision: revision,
      submissionId: submission?.id ?? null,
      submissionState: submission ? "pending" : null,
    };
    if (submission) {
      let unsubscribe!: () => void;
      const updateResult = () => {
        const result = target.snapshot().submissionResult;
        if (result?.id !== submission.id) return;
        session.submissionState = result.state;
        unsubscribe();
        if (submissionUnsubscribe.current === unsubscribe)
          submissionUnsubscribe.current = null;
      };
      unsubscribe = target.subscribe(updateResult);
      submissionUnsubscribe.current = unsubscribe;
      updateResult();
    }
    setVoiceError("");
    setLevels([0, 0, 0, 0, 0]);
    const active = new VoiceRecording((event) => {
      if (
        recording.current !== active ||
        currentDraft.current.controller !== target ||
        currentDraft.current.discussionId !== discussionId
      )
        return;
      if (event.type === "state") {
        setVoiceState(event.state);
        if (event.state === "closed") recording.current = null;
      } else if (event.type === "transcript") {
        if (target.snapshot().draft.bodyRevision !== session.bodyRevision) {
          cancelRecording();
          return;
        }
        const snapshot = target.snapshot();
        const retainsSubmission =
          session.submissionId !== null &&
          session.submissionState !== "sent" &&
          snapshot.draft.body.startsWith(session.submittedBody) &&
          (snapshot.draft.pending?.id === session.submissionId ||
            session.submissionState === "cancelled");
        const submittedBody = retainsSubmission ? session.submittedBody : "";
        const text = submittedBody + session.voiceBase + event.text;
        const voiceSubmission =
          session.submissionId !== null &&
          snapshot.draft.pending?.id === session.submissionId &&
          retainsSubmission
            ? { id: session.submissionId, submittedBody }
            : null;
        target.setVoiceBody(text, voiceSubmission);
        session.bodyRevision = target.snapshot().draft.bodyRevision;
        setCaret(text.length);
      } else if (event.type === "level") {
        setLevels((previous) => [...previous.slice(1), event.level]);
      } else if (event.code === "voice_config") {
        showVoiceSettingsToast();
      } else {
        setVoiceError(event.message);
      }
    });
    recording.current = active;
    void active.start();
  };
  const [caret, setCaret] = useState(0);
  const [highlighted, setHighlighted] = useState(0);
  const [dismissed, setDismissed] = useState(false);
  const input = useRef<HTMLTextAreaElement>(null);
  const menuId = useId();
  const card = useRef<HTMLFieldSetElement>(null);
  const probe = useRef<HTMLTextAreaElement>(null);
  const topFade = useRef<HTMLDivElement>(null);
  const bottomFade = useRef<HTMLDivElement>(null);
  const updateFades = useCallback(() => {
    const element = input.current;
    if (!element || !topFade.current || !bottomFade.current) return;
    const fades = composerFades(
      element.scrollTop,
      element.scrollHeight,
      element.clientHeight,
    );
    topFade.current.style.opacity = String(fades.top);
    bottomFade.current.style.opacity = String(fades.bottom);
    topFade.current.style.top = `${element.offsetTop}px`;
    bottomFade.current.style.top = `${element.offsetTop + element.offsetHeight - 32}px`;
  }, []);
  const [layout, setLayout] = useState({
    expanded: false,
    height: 48,
    smooth: false,
  });

  useLayoutEffect(() => {
    const element = input.current;
    const measureInput = probe.current;
    if (!element || !measureInput) return;
    const measure = (typing = false) => {
      const compactStyle = getComputedStyle(measureInput);
      const compactLine = Number.parseFloat(compactStyle.lineHeight);
      const compactPadding =
        Number.parseFloat(compactStyle.paddingTop) +
        Number.parseFloat(compactStyle.paddingBottom);
      measureInput.style.height = "auto";
      const expanded =
        files.length > 0 ||
        !!view?.draft.pending ||
        recordingActive ||
        composerMultiline(
          element.value,
          measureInput.scrollHeight,
          compactLine,
          compactPadding,
        );
      element.style.paddingBlock = expanded ? "14px" : "12px";
      const style = getComputedStyle(element);
      const line = Number.parseFloat(style.lineHeight);
      const padding =
        Number.parseFloat(style.paddingTop) +
        Number.parseFloat(style.paddingBottom);
      const scrollTop = element.scrollTop;
      element.style.height = "auto";
      const height = expanded
        ? Math.max(
            68,
            autoGrowHeight(element.scrollHeight, line, padding, 0, 8),
          )
        : line + padding;
      element.style.height = `${height}px`;
      element.scrollTop = scrollTop;
      updateFades();
      setLayout((previous) => {
        const nextHeight = composerHeight(expanded, height);
        const smooth =
          previous.expanded !== expanded
            ? false
            : typing
              ? true
              : previous.smooth;
        return previous.expanded === expanded &&
          previous.height === nextHeight &&
          previous.smooth === smooth
          ? previous
          : { expanded, height: nextHeight, smooth };
      });
    };
    measure(true);
    let width = element.getBoundingClientRect().width;
    let compactWidth = measureInput.getBoundingClientRect().width;
    const observer = new ResizeObserver(() => {
      const nextWidth = element.getBoundingClientRect().width;
      const nextCompactWidth = measureInput.getBoundingClientRect().width;
      if (nextWidth === width && nextCompactWidth === compactWidth) return;
      width = nextWidth;
      compactWidth = nextCompactWidth;
      measure();
    });
    observer.observe(element);
    observer.observe(measureInput);
    let mounted = true;
    void document.fonts.ready.then(() => {
      if (mounted) measure();
    });
    return () => {
      mounted = false;
      observer.disconnect();
    };
  }, [
    body,
    files.length,
    view?.draft.pending,
    extraHeight,
    recordingActive,
    updateFades,
  ]);

  useLayoutEffect(() => {
    const element = card.current;
    if (!element) return;
    let height = -1;
    const update = () => {
      const next = element.getBoundingClientRect().height;
      if (height === next) return;
      height = next;
      onHeightChange(next);
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(element);
    return () => observer.disconnect();
  }, [onHeightChange]);

  const mention = mentionQuery(body, caret);
  const grouped = useMemo(
    () =>
      mention
        ? candidatesFor(members, memberIds, mention.query)
        : { inDiscussion: [], elsewhere: [] },
    [mention, members, memberIds],
  );

  const candidates = [...grouped.inDiscussion, ...grouped.elsewhere].slice(
    0,
    MENU_LIMIT,
  );
  const suggesting = mention !== null && candidates.length > 0 && !dismissed;
  const active = candidates.length > 0 ? highlighted % candidates.length : 0;
  const elsewhereFrom = Math.min(
    grouped.inDiscussion.length,
    candidates.length,
  );

  const accept = (member: Member) => {
    if (!mention) return;
    const next = completeMention(body, mention, caret, member.name);
    cancelRecording();
    setBody(next.text);
    setCaret(next.caret);
    setHighlighted(0);
    requestAnimationFrame(() => {
      input.current?.focus();
      input.current?.setSelectionRange(next.caret, next.caret);
    });
  };

  const submit = async () => {
    if (!canSend || sending || !controller || recording.current) return;
    await controller.send(onSend);
    if (!controller.snapshot().draft.body) setCaret(0);
  };
  const draftStatus =
    view?.progress ??
    (view?.saving
      ? "Saving draft…"
      : view?.storageError
        ? "Draft not saved"
        : null);

  return (
    <div
      style={composerTheme}
      className="pointer-events-none absolute inset-x-4 bottom-4 flex justify-center"
    >
      <div
        className="invisible pointer-events-none absolute top-0 w-full @[601px]:w-3/4 border border-transparent"
        aria-hidden="true"
      >
        <Textarea
          ref={probe}
          variant="composer"
          style={{ paddingLeft: 44 }}
          value={body}
          readOnly
          rows={1}
          tabIndex={-1}
        />
      </div>
      <fieldset
        ref={card}
        aria-label="Message composer"
        onDragOver={(event) => {
          if (event.dataTransfer.types.includes("Files"))
            event.preventDefault();
        }}
        onDrop={(event) => {
          if (!event.dataTransfer.files.length) return;
          event.preventDefault();
          controller?.addFiles(Array.from(event.dataTransfer.files));
        }}
        data-expanded={layout.expanded}
        className={clsx(
          "pointer-events-auto relative w-full rounded-[24px] border border-(--composer-border) bg-(--composer-card) text-(--composer-foreground) shadow-[0_1px_3px_0_rgb(0_0_0/0.1),0_1px_2px_-1px_rgb(0_0_0/0.1)] focus-within:border-(--composer-ring)/40 focus-within:ring-1 focus-within:ring-(--composer-ring)/20 hover:border-(--composer-border)/80 transition-[width,height] motion-reduce:transition-none",
          layout.expanded ? "@[601px]:w-[90%]" : "@[601px]:w-3/4",
        )}
        style={{
          height: layout.height + extraHeight,
          transitionDuration: layout.smooth ? "0.4s, 0.15s" : "0.4s, 0.4s",
          transitionTimingFunction: layout.smooth
            ? "cubic-bezier(0.175,0.885,0.32,1.275), ease-out"
            : "cubic-bezier(0.175,0.885,0.32,1.275)",
        }}
      >
        <input
          ref={fileInput}
          type="file"
          multiple
          hidden
          aria-label="Choose attachments"
          onChange={(event) => {
            controller?.addFiles(Array.from(event.currentTarget.files ?? []));
            event.currentTarget.value = "";
          }}
        />
        <div ref={extra}>
          {files.length ? (
            <DraftAttachments
              files={files}
              onRemove={(id) => controller?.removeFile(id)}
            />
          ) : null}
          {draftError || view?.storageError || view?.error ? (
            <div role="alert" className="px-4 pt-3 text-sm text-red-300">
              {draftError ?? view?.storageError ?? view?.error}
              {view?.storageError ? (
                <button
                  type="button"
                  className="ml-2 underline"
                  onClick={() => controller?.saveAgain()}
                >
                  Retry saving draft
                </button>
              ) : null}
            </div>
          ) : null}
          {view?.draft.pending ? (
            <div className="px-4 pt-2 text-xs">
              <div className="truncate">
                Saved send attempt: {view.draft.pending.body || "Files"} ·{" "}
                {view.draft.pending.files.length} files
              </div>
              {sending ? (
                view.draft.pending.phase === "uploading" ? (
                  <button
                    type="button"
                    className="mt-1 underline"
                    onClick={() => controller?.cancel()}
                  >
                    Cancel upload
                  </button>
                ) : null
              ) : (
                <div className="mt-1 flex gap-3">
                  <button
                    type="button"
                    className="underline"
                    onClick={() => void submit()}
                  >
                    Retry saved send
                  </button>
                  {view.draft.pending.phase === "sending" ? (
                    <button
                      type="button"
                      className="underline"
                      onClick={() => void controller?.checkResult()}
                    >
                      Check send result
                    </button>
                  ) : null}
                  <button
                    type="button"
                    className="underline"
                    onClick={() => void controller?.discardAttempt()}
                  >
                    Discard send attempt
                  </button>
                </div>
              )}
            </div>
          ) : null}
          {draftStatus ? (
            <div role="status" className="px-4 pt-2 text-xs opacity-70">
              {draftStatus}
            </div>
          ) : null}
        </div>
        {voiceError ? (
          <div
            role="alert"
            className="absolute bottom-full left-0 right-0 mb-2 rounded-lg bg-surface-raised p-3 text-sm text-fg"
          >
            {voiceError}
          </div>
        ) : null}
        {suggesting ? (
          <div
            className="absolute bottom-full mb-2 left-0 right-0 z-(--layer-popover) max-h-66 overflow-y-auto rounded-sm bg-surface-raised p-1 shadow-popover origin-bottom animate-pop-in"
            id={menuId}
            role="listbox"
            aria-label="Members"
          >
            {candidates.map((member, index) => {
              const inside = memberIds.has(member.id);
              return (
                <Fragment key={member.id}>
                  {index === elsewhereFrom && index > 0 ? (
                    <div
                      className="mt-1 border-t border-line px-2 pt-3 pb-1 text-xs text-fg-muted"
                      role="presentation"
                    >
                      Not in this Discussion
                    </div>
                  ) : null}
                  <button
                    type="button"
                    className={clsx(
                      "flex w-full items-center gap-2 rounded-xs border-0 px-2 py-1 text-left text-inherit cursor-pointer transition-[background-color] duration-(--duration-fast) ease-linear hover:bg-surface-hover",
                      index === active
                        ? "bg-surface-hover shadow-[inset_2px_0_0_var(--color-blue-300)]"
                        : "bg-transparent",
                    )}
                    role="option"
                    id={`${menuId}-${member.id}`}
                    aria-selected={index === active}
                    onMouseDown={(event) => {
                      event.preventDefault();
                      accept(member);
                    }}
                  >
                    <Avatar memberId={member.id} size="sm" />
                    <span
                      className={clsx(
                        "min-w-0 flex-1 truncate",
                        inside ? "font-medium" : "font-normal text-fg-muted",
                      )}
                    >
                      {member.name}
                    </span>
                    {inside ? (
                      <span className="flex-none text-xs text-fg-muted">
                        {member.type === "human" ? "Human" : member.state}
                      </span>
                    ) : null}
                  </button>
                </Fragment>
              );
            })}
          </div>
        ) : null}
        <div className="flex min-w-0 flex-1">
          <Textarea
            ref={input}
            value={body}
            disabled={!controller}
            style={{ paddingLeft: layout.expanded ? undefined : 44 }}
            onPaste={(event) => {
              const images = Array.from(event.clipboardData.files).filter(
                (file) => file.type.startsWith("image/"),
              );
              if (!images.length) return;
              if (!event.clipboardData.getData("text/plain"))
                event.preventDefault();
              controller?.addFiles(images);
            }}
            rows={1}
            variant="composer"
            onScroll={updateFades}
            placeholder={placeholder}
            aria-label="Message"
            role="combobox"
            aria-expanded={suggesting}
            aria-controls={suggesting ? menuId : undefined}
            aria-autocomplete="list"
            aria-activedescendant={
              suggesting ? `${menuId}-${candidates[active].id}` : undefined
            }
            onChange={(event) => {
              cancelRecording();
              setBody(event.target.value);
              setCaret(event.target.selectionStart ?? 0);
              setHighlighted(0);
              setDismissed(false);
            }}
            onSelect={(event) =>
              setCaret(event.currentTarget.selectionStart ?? 0)
            }
            onKeyDown={(event) => {
              const action = composerKey(
                {
                  key: event.key,
                  shiftKey: event.shiftKey,
                  ctrlKey: event.ctrlKey,
                  metaKey: event.metaKey,
                  isComposing: event.nativeEvent.isComposing,
                },
                suggesting,
              );
              if (action === null || action === "newline") return;
              event.preventDefault();
              const size = candidates.length;
              switch (action) {
                case "down":
                  setHighlighted((current) => (current + 1) % size);
                  break;
                case "up":
                  setHighlighted((current) => (current - 1 + size) % size);
                  break;
                case "accept":
                  accept(candidates[active]);
                  break;
                case "dismiss":
                  setDismissed(true);
                  break;
                case "send":
                  void submit();
                  break;
              }
            }}
          />
        </div>
        <div
          ref={topFade}
          aria-hidden="true"
          className="pointer-events-none absolute left-4 right-12 top-0 z-[2] h-8 bg-gradient-to-b from-(--composer-card) via-(--composer-card)/90 to-transparent"
          style={{ opacity: 0 }}
        />
        <div
          ref={bottomFade}
          aria-hidden="true"
          className="pointer-events-none absolute left-4 right-12 z-[2] h-8 bg-gradient-to-t from-(--composer-card) via-(--composer-card)/90 to-transparent"
          style={{ opacity: 0 }}
        />
        <div
          aria-hidden={!layout.expanded}
          className={clsx(
            "absolute bottom-2 left-3 right-12 z-10 flex min-w-0 items-center gap-0 transition-all duration-300 ease-[cubic-bezier(0.175,0.885,0.32,1.275)] motion-reduce:transition-none",
            layout.expanded
              ? "opacity-100 blur-0 translate-y-0 pointer-events-auto"
              : "opacity-0 blur-sm translate-y-2 pointer-events-none",
          )}
        >
          {!recordingActive && canSend ? (
            <button
              type="button"
              aria-label="Start voice input"
              onClick={() => void startRecording()}
            >
              <MicIcon />
            </button>
          ) : null}
          {recordingActive ? (
            <div className="flex items-center gap-3 text-xs" role="status">
              <div
                className="flex h-5 items-center gap-1"
                role="img"
                aria-label="Microphone volume"
              >
                {levels.map((level, index) => (
                  <span
                    key={index}
                    className="w-1 rounded-full bg-current"
                    style={{
                      height: `${Math.max(2, Math.min(20, level * 100))}px`,
                    }}
                  />
                ))}
              </div>
              <span>
                {voiceState === "starting"
                  ? "Starting microphone…"
                  : voiceState === "finishing"
                    ? "Transcribing…"
                    : "Listening…"}
              </span>
              <button type="button" onClick={cancelRecording}>
                Cancel
              </button>
            </div>
          ) : null}
          <button
            type="button"
            disabled={!controller}
            onClick={() => fileInput.current?.click()}
            aria-label="Attach files"
            title="Attach files"
            className="ml-auto flex size-7 flex-none items-center justify-center rounded-full text-(--composer-foreground)/50 outline-none cursor-default disabled:opacity-40"
          >
            <PlusIcon />
          </button>
        </div>
        {!layout.expanded ? (
          <button
            type="button"
            aria-label="Attach files"
            title="Attach files"
            disabled={!controller}
            onClick={() => fileInput.current?.click()}
            className="absolute left-2 bottom-2 z-10 flex size-8 items-center justify-center rounded-full hover:bg-white/10"
          >
            <PlusIcon />
          </button>
        ) : null}
        <button
          type="button"
          className="absolute right-2 bottom-2 z-10 flex size-8 items-center justify-center rounded-full bg-(--composer-primary) text-(--composer-primary-foreground) transition-all duration-300 hover:enabled:opacity-90 outline-none focus-visible:ring-2 focus-visible:ring-(--composer-ring) cursor-default motion-reduce:transition-none"
          aria-label={
            recordingActive
              ? "Stop recording"
              : canSend
                ? "Send · Enter"
                : "Start voice input"
          }
          title={
            recordingActive
              ? "Stop recording"
              : canSend
                ? "Send · Enter"
                : "Start voice input"
          }
          disabled={
            !controller ||
            (!recordingActive && sending) ||
            voiceState === "finishing"
          }
          onClick={() => {
            if (recordingActive) recording.current?.stop();
            else if (canSend) void submit();
            else void startRecording();
          }}
        >
          {recordingActive ? (
            <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
              <rect width="12" height="12" rx="2" fill="currentColor" />
            </svg>
          ) : (
            <span
              className="relative flex h-full w-full items-center justify-center"
              aria-hidden="true"
            >
              <span
                className={clsx(
                  "absolute inset-0 flex items-center justify-center transition-all duration-300 ease-[cubic-bezier(0.175,0.885,0.32,1.275)] motion-reduce:transition-none",
                  canSend
                    ? "opacity-100 scale-100 rotate-0 blur-none"
                    : "opacity-0 scale-50 rotate-45 blur-[1px] pointer-events-none",
                )}
              >
                <ArrowUpIcon />
              </span>
              <span
                className={clsx(
                  "absolute inset-0 flex items-center justify-center transition-all duration-300 ease-[cubic-bezier(0.175,0.885,0.32,1.275)] motion-reduce:transition-none",
                  canSend
                    ? "opacity-0 scale-50 -rotate-45 blur-[1px] pointer-events-none"
                    : "opacity-100 scale-100 rotate-0 blur-none",
                )}
              >
                <MicIcon />
              </span>
            </span>
          )}
        </button>
      </fieldset>
    </div>
  );
}
