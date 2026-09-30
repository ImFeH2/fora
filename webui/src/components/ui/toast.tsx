import * as ToastPrimitive from "@radix-ui/react-toast";
import { clsx } from "clsx";
import { CircleAlert, CircleCheck, Info, X } from "lucide-react";
import { useSyncExternalStore } from "react";
import { createPortal } from "react-dom";
import { Button, IconButton } from "@/components/ui/index";
import { readToastHost, subscribeToastHost } from "@/components/ui/toast-host";

const toastTones = {
  success: "text-success",
  danger: "text-danger",
  info: "text-primary",
};

export type ToastTone = "success" | "danger" | "info";

export type ToastAction = { label: string; onClick: () => void };

export type ToastOptions = {
  tone: ToastTone;
  title: string;
  description?: string;
  action?: ToastAction;
  duration?: number | null;
  closable?: boolean;
  id?: string;
};

export type ToastItem = {
  id: string;
  tone: ToastTone;
  title: string;
  description?: string;
  action?: ToastAction;
  duration: number | null;
  closable: boolean;
  open: boolean;
  revision: number;
};

const DURATIONS: Record<ToastTone, number> = {
  success: 4000,
  info: 4000,
  danger: 8000,
};

const EXIT_MS = 200;

let items: ToastItem[] = [];
let sequence = 0;
const listeners = new Set<() => void>();
const removals = new Map<string, ReturnType<typeof setTimeout>>();

function publish(next: ToastItem[]) {
  items = next;
  for (const listener of listeners) listener();
}

export function subscribeToasts(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function readToasts(): ToastItem[] {
  return items;
}

export function toast(options: ToastOptions): string {
  const id = options.id ?? `toast-${++sequence}`;
  const pending = removals.get(id);
  if (pending) {
    clearTimeout(pending);
    removals.delete(id);
  }
  const existing = items.find((item) => item.id === id);
  const next: ToastItem = {
    id,
    tone: options.tone,
    title: options.title,
    description: options.description,
    action: options.action,
    duration:
      options.duration === undefined
        ? DURATIONS[options.tone]
        : options.duration,
    closable: options.closable ?? true,
    open: true,
    revision: existing ? existing.revision + 1 : 0,
  };
  publish(
    existing
      ? items.map((item) => (item.id === id ? next : item))
      : [...items, next],
  );
  return id;
}

export function dismissToast(id: string): void {
  const existing = items.find((item) => item.id === id);
  if (!existing?.open) return;
  publish(
    items.map((item) => (item.id === id ? { ...item, open: false } : item)),
  );
  removals.set(
    id,
    setTimeout(() => {
      removals.delete(id);
      publish(items.filter((item) => item.id !== id));
    }, EXIT_MS),
  );
}

export function clearToasts(): void {
  for (const timer of removals.values()) clearTimeout(timer);
  removals.clear();
  publish([]);
}

function ToneIcon({ tone }: { tone: ToastTone }) {
  if (tone === "success") return <CircleCheck size={16} />;
  if (tone === "danger") return <CircleAlert size={16} />;
  return <Info size={16} />;
}

function ToastView({ item }: { item: ToastItem }) {
  return (
    <ToastPrimitive.Root
      className="flex items-start gap-3 p-3 pl-4 border border-line rounded-md bg-surface-raised text-fg text-sm shadow-popover origin-bottom-right data-[state=open]:animate-rise-in data-[state=open]:[animation-timing-function:var(--ease-out)] data-[state=closed]:animate-fade-in data-[state=closed]:[animation-direction:reverse] data-[swipe=move]:[transform:translateX(var(--radix-toast-swipe-move-x))] data-[swipe=cancel]:[transform:translateX(0)] data-[swipe=cancel]:transition-[transform] data-[swipe=cancel]:duration-(--duration-base) data-[swipe=cancel]:ease-out data-[swipe=end]:animate-fade-in data-[swipe=end]:[animation-timing-function:var(--ease-standard)] data-[swipe=end]:[animation-direction:reverse] [&_button]:flex-none"
      type={item.tone === "danger" ? "foreground" : "background"}
      duration={item.duration ?? Infinity}
      open={item.open}
      onOpenChange={(open) => {
        if (!open && item.closable) dismissToast(item.id);
      }}
      onSwipeEnd={(event) => {
        if (!item.closable) event.preventDefault();
      }}
    >
      <span
        className={clsx("flex flex-none pt-[2px]", toastTones[item.tone])}
        aria-hidden="true"
      >
        <ToneIcon tone={item.tone} />
      </span>
      <div className="flex flex-1 flex-col gap-[2px] min-w-0 pt-[3px]">
        <ToastPrimitive.Title className="m-0 font-medium leading-body">
          {item.title}
        </ToastPrimitive.Title>
        {item.description ? (
          <ToastPrimitive.Description className="m-0 text-fg-muted text-xs leading-4 wrap-anywhere">
            {item.description}
          </ToastPrimitive.Description>
        ) : null}
      </div>
      {item.action ? (
        <ToastPrimitive.Action asChild altText={item.action.label}>
          <Button size="sm" onClick={item.action.onClick}>
            {item.action.label}
          </Button>
        </ToastPrimitive.Action>
      ) : null}
      {item.closable ? (
        <ToastPrimitive.Close asChild>
          <IconButton label="Close" size="sm">
            <X size={14} />
          </IconButton>
        </ToastPrimitive.Close>
      ) : null}
    </ToastPrimitive.Root>
  );
}

export function Toaster() {
  const current = useSyncExternalStore(subscribeToasts, readToasts, readToasts);
  const host = useSyncExternalStore(
    subscribeToastHost,
    readToastHost,
    () => null,
  );
  const viewport =
    current.length > 0 || !host ? (
      <ToastPrimitive.Viewport
        className={
          host
            ? "flex flex-col gap-2 w-full m-0 p-0 list-none outline-none"
            : "fixed right-6 bottom-6 z-(--layer-toast) flex flex-col gap-2 w-[min(360px,calc(100vw-48px))] m-0 p-0 list-none outline-none"
        }
        label="Notifications"
      />
    ) : null;
  return (
    <ToastPrimitive.Provider label="Notification" swipeDirection="right">
      {current.map((item) => (
        <ToastView key={`${item.id}:${item.revision}`} item={item} />
      ))}
      {host ? createPortal(viewport, host) : viewport}
    </ToastPrimitive.Provider>
  );
}
