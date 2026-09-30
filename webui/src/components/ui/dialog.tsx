import * as Dialog from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import {
  type ReactNode,
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
} from "react";
import { Button, Field, IconButton, Input, toast } from "@/components/ui/index";
import { registerToastHost } from "@/components/ui/toast-host";

export function dialogFocusTarget<
  T extends { disabled?: boolean },
  U extends { disabled?: boolean },
>(fields: T[], buttons: U[]): T | U | null {
  const field = fields.find((item) => !item.disabled);
  if (field) return field;
  const enabled = buttons.filter((item) => !item.disabled);
  return enabled.length > 0 ? enabled[enabled.length - 1] : null;
}

function reportFailure(title: string, failure: unknown) {
  toast({
    tone: "danger",
    title,
    description: failure instanceof Error ? failure.message : String(failure),
  });
}

function refocus(target: { focus: () => void } | null) {
  requestAnimationFrame(() => {
    if (document.activeElement === document.body) target?.focus();
  });
}

export function Modal({
  open,
  onOpenChange,
  title,
  description,
  children,
  footer,
  wide = false,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description?: ReactNode;
  children?: ReactNode;
  footer: ReactNode;
  wide?: boolean;
}) {
  const opener = useRef<HTMLElement | null>(null);
  const content = useRef<HTMLDivElement>(null);
  const footerElement = useRef<HTMLDivElement>(null);
  const toastHost = useCallback((node: HTMLDivElement | null) => {
    if (node) return registerToastHost(node);
  }, []);
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-(--layer-overlay) bg-gray-1100/60 animate-fade-in [animation-duration:var(--duration-fast)] [animation-timing-function:var(--ease-out)]" />
        <Dialog.Content
          ref={content}
          className={`fixed top-1/2 left-1/2 z-(--layer-dialog) flex min-h-0 max-h-[calc(100vh-32px)] flex-col gap-4 p-6 border border-line rounded-lg bg-surface-raised shadow-dialog [translate:-50%_-50%] origin-center animate-pop-in ${wide ? "w-[min(960px,calc(100vw-32px))]" : "w-[min(440px,calc(100vw-48px))]"}`}
          onOpenAutoFocus={(event) => {
            opener.current =
              document.activeElement instanceof HTMLElement
                ? document.activeElement
                : null;
            event.preventDefault();
            const root = content.current;
            if (!root) return;
            const target = dialogFocusTarget(
              Array.from(
                root.querySelectorAll<
                  HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement
                >("input, textarea, select"),
              ),
              Array.from(
                footerElement.current?.querySelectorAll<HTMLButtonElement>(
                  "button",
                ) ?? [],
              ),
            );
            (target ?? root).focus();
          }}
          onCloseAutoFocus={(event) => {
            event.preventDefault();
            opener.current?.focus();
          }}
          {...(description ? {} : { "aria-describedby": undefined })}
        >
          <div className="flex items-start justify-between gap-3">
            <Dialog.Title className="min-w-0 wrap-anywhere m-0 text-md font-semibold tracking-title">
              {title}
            </Dialog.Title>
            <Dialog.Close asChild>
              <IconButton label="Close" size="sm">
                <X size={15} />
              </IconButton>
            </Dialog.Close>
          </div>
          {description ? (
            <Dialog.Description className="-mt-4 mx-0 mb-0 text-fg-muted text-sm">
              {description}
            </Dialog.Description>
          ) : null}
          <div ref={toastHost} className="empty:hidden shrink-0" />
          {children ? (
            <div className="min-h-0 flex-1 overflow-y-auto flex flex-col gap-4">
              {children}
            </div>
          ) : null}
          <div ref={footerElement} className="flex justify-end gap-2">
            {footer}
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

export function PromptDialog({
  open,
  onOpenChange,
  title,
  label,
  placeholder,
  description,
  validate,
  trim = true,
  initial = "",
  submitLabel,
  onSubmit,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  label: string;
  placeholder?: string;
  description?: ReactNode;
  validate?: (value: string) => string | undefined;
  trim?: boolean;
  initial?: string;
  submitLabel: string;
  onSubmit: (value: string) => void | Promise<void>;
}) {
  const [value, setValue] = useState(initial);
  const [busy, setBusy] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const inputId = useId();

  useEffect(() => {
    if (open) setValue(initial);
  }, [open, initial]);

  const submitted = trim ? value.trim() : value;
  const error = validate?.(submitted);

  const commit = async () => {
    if (!submitted || error || busy) return;
    setBusy(true);
    try {
      await onSubmit(submitted);
      onOpenChange(false);
    } catch (failure) {
      reportFailure(`Could not ${submitLabel.toLowerCase()}`, failure);
      refocus(input.current);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      open={open}
      onOpenChange={onOpenChange}
      title={title}
      description={description}
      footer={
        <>
          <Button disabled={busy} onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            variant="primary"
            disabled={!submitted || !!error || busy}
            onClick={commit}
          >
            {submitLabel}
          </Button>
        </>
      }
    >
      <Field label={label} htmlFor={inputId}>
        <Input
          ref={input}
          id={inputId}
          value={value}
          placeholder={placeholder}
          aria-invalid={!!error}
          aria-describedby={error ? `${inputId}-error` : undefined}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              void commit();
            }
          }}
        />
        {error ? (
          <p
            id={`${inputId}-error`}
            className="text-sm text-danger"
            role="alert"
          >
            {error}
          </p>
        ) : null}
      </Field>
    </Modal>
  );
}

export function ConfirmDialog({
  open,
  onOpenChange,
  title,
  description,
  confirmLabel,
  onConfirm,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description: ReactNode;
  confirmLabel: string;
  onConfirm: () => void | Promise<void>;
}) {
  const [busy, setBusy] = useState(false);
  const action = useRef<HTMLButtonElement>(null);

  const confirm = async () => {
    if (busy) return;
    setBusy(true);
    try {
      await onConfirm();
      onOpenChange(false);
    } catch (failure) {
      reportFailure(`Could not ${confirmLabel.toLowerCase()}`, failure);
      refocus(action.current);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      open={open}
      onOpenChange={onOpenChange}
      title={title}
      description={description}
      footer={
        <>
          <Button disabled={busy} onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            ref={action}
            variant="danger"
            disabled={busy}
            onClick={confirm}
          >
            {confirmLabel}
          </Button>
        </>
      }
    />
  );
}
