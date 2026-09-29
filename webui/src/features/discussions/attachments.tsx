import { Download, FileIcon, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Modal } from "@/components/ui/dialog";
import { Button, Spinner } from "@/components/ui/index";
import type { DraftFile } from "@/features/discussions/draft";
import { type Attachment, backend } from "@/lib/backend";

export function fileSize(size: number) {
  return size >= 1024 * 1024
    ? `${(size / (1024 * 1024)).toFixed(1)} MiB`
    : `${Math.ceil(size / 1024)} KiB`;
}

function FilePreview({ file }: { file: File }) {
  const [url, setUrl] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) return;
    const next = URL.createObjectURL(file);
    setUrl(next);
    return () => URL.revokeObjectURL(next);
  }, [file]);
  return url && !failed ? (
    <img
      src={url}
      alt={file.name}
      onError={() => setFailed(true)}
      className="size-12 rounded object-cover"
    />
  ) : (
    <FileIcon className="size-8 flex-none" />
  );
}

export function DraftAttachments({
  files,
  onRemove,
}: {
  files: DraftFile[];
  onRemove: (id: string) => void;
}) {
  return (
    <ul
      aria-label="Draft attachments"
      className="m-0 flex max-h-40 list-none flex-wrap gap-2 overflow-y-auto p-3 pb-0"
    >
      {files.map((item) => (
        <li
          key={item.id}
          className="flex min-w-0 max-w-full items-center gap-2 rounded-lg border border-(--composer-border) p-2"
        >
          <FilePreview file={item.file} />
          <div className="min-w-0 max-w-44">
            <div className="truncate text-xs" title={item.file.name}>
              {item.file.name}
            </div>
            <div className="text-xs opacity-60">{fileSize(item.file.size)}</div>
          </div>
          <button
            type="button"
            aria-label={`Remove ${item.file.name}`}
            className="flex size-6 flex-none items-center justify-center rounded hover:bg-white/10"
            onClick={() => onRemove(item.id)}
          >
            <X size={14} />
          </button>
        </li>
      ))}
    </ul>
  );
}

export function MessageAttachment({
  discussionId,
  messageId,
  attachment,
}: {
  discussionId: number;
  messageId: number;
  attachment: Attachment;
}) {
  const root = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  const [url, setUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [opened, setOpened] = useState(false);
  const [loading, setLoading] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const downloadRequest = useRef<AbortController | null>(null);
  const downloadUrls = useRef(new Map<string, ReturnType<typeof setTimeout>>());
  const [attempt, setAttempt] = useState(0);
  useEffect(
    () => () => {
      downloadRequest.current?.abort();
      for (const [objectUrl, timer] of downloadUrls.current) {
        clearTimeout(timer);
        URL.revokeObjectURL(objectUrl);
      }
      downloadUrls.current.clear();
    },
    [],
  );
  const image = attachment.width !== null && attachment.height !== null;
  const shouldLoad = image && (visible || opened);
  useEffect(() => {
    const observer = new IntersectionObserver((entries) =>
      setVisible(entries.some((entry) => entry.isIntersecting)),
    );
    if (root.current) observer.observe(root.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (!shouldLoad) {
      setUrl(null);
      setLoading(false);
      return;
    }
    const abort = new AbortController();
    let objectUrl: string | null = null;
    setLoading(true);
    setError(null);
    void backend
      .attachmentFile(discussionId, messageId, attachment, abort.signal)
      .then(
        (blob) => {
          if (abort.signal.aborted) return;
          objectUrl = URL.createObjectURL(blob);
          setUrl(objectUrl);
          setLoading(false);
        },
        (failure: unknown) => {
          if (!abort.signal.aborted) {
            setError(
              failure instanceof Error ? failure.message : String(failure),
            );
            setLoading(false);
          }
        },
      );
    return () => {
      abort.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [discussionId, messageId, attachment, shouldLoad, attempt]);

  const download = async () => {
    if (downloadRequest.current) return;
    const abort = new AbortController();
    downloadRequest.current = abort;
    setDownloading(true);
    setDownloadError(null);
    try {
      const blob = await backend.attachmentFile(
        discussionId,
        messageId,
        attachment,
        abort.signal,
      );
      if (abort.signal.aborted) return;
      const objectUrl = URL.createObjectURL(blob);
      const timer = setTimeout(() => {
        URL.revokeObjectURL(objectUrl);
        downloadUrls.current.delete(objectUrl);
      }, 30_000);
      downloadUrls.current.set(objectUrl, timer);
      const link = document.createElement("a");
      link.href = objectUrl;
      link.download = attachment.name;
      document.body.append(link);
      link.click();
      link.remove();
    } catch (failure) {
      if (!abort.signal.aborted)
        setDownloadError(
          failure instanceof Error ? failure.message : String(failure),
        );
    } finally {
      downloadRequest.current = null;
      if (!abort.signal.aborted) setDownloading(false);
    }
  };

  return (
    <div
      ref={root}
      className="my-2 max-w-sm overflow-hidden rounded-lg border border-line bg-surface-raised"
    >
      {image ? (
        <button
          type="button"
          disabled={!url}
          aria-label={`View ${attachment.name}`}
          onClick={() => setOpened(true)}
          className="block w-full overflow-hidden bg-black/10"
          style={{
            aspectRatio: `${attachment.width}/${attachment.height}`,
            maxHeight: 280,
          }}
        >
          {url ? (
            <img
              src={url}
              alt={attachment.name}
              className="h-full w-full object-contain"
              onError={() => setError("The image could not be displayed.")}
            />
          ) : loading ? (
            <Spinner label="Loading image" />
          ) : (
            <span className="text-xs text-fg-muted">Image preview</span>
          )}
        </button>
      ) : null}
      <div className="flex items-center gap-2 p-3">
        {image ? null : <FileIcon size={20} />}
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm" title={attachment.name}>
            {attachment.name}
          </div>
          <div className="text-xs text-fg-muted">
            {fileSize(attachment.size)}
          </div>
        </div>
        <Button
          size="sm"
          disabled={downloading}
          onClick={() => void download()}
          aria-label={`Download ${attachment.name}`}
        >
          <Download size={14} />
          Download
        </Button>
      </div>
      {error ? (
        <div role="alert" className="px-3 pb-3 text-sm text-danger">
          {error}
          <Button size="sm" onClick={() => setAttempt((value) => value + 1)}>
            Retry preview
          </Button>
        </div>
      ) : null}
      {downloadError ? (
        <div role="alert" className="px-3 pb-3 text-sm text-danger">
          {downloadError}
          <Button
            size="sm"
            disabled={downloading}
            onClick={() => void download()}
          >
            Retry download
          </Button>
        </div>
      ) : null}
      <Modal
        open={opened}
        onOpenChange={setOpened}
        title={attachment.name}
        footer={
          <Button disabled={downloading} onClick={() => void download()}>
            Download original
          </Button>
        }
      >
        {url ? (
          <img
            src={url}
            alt={attachment.name}
            className="max-h-[65vh] w-full object-contain"
          />
        ) : null}
      </Modal>
    </div>
  );
}
