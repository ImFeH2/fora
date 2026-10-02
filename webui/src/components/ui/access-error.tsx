import { CircleAlertIcon } from "lucide-react";
import { useId } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button, Spinner } from "@/components/ui/index";
import type { BackendError } from "@/lib/backend";

export function AccessError({
  error,
  reconnect,
  waiting = false,
  loadingData = false,
}: {
  error: BackendError;
  reconnect?: () => void;
  waiting?: boolean;
  loadingData?: boolean;
}) {
  const titleId = useId();
  const descriptionId = useId();
  const authentication =
    !loadingData &&
    (error.code === "authentication_missing" ||
      error.code === "authentication_failed");
  const startup = !loadingData && error.code === "startup_failed";
  const preparation = [
    "storage_read_failed",
    "storage_write_failed",
    "url_cleanup_failed",
    "invalid_connection",
  ].includes(error.code);
  const title = loadingData
    ? "Could not load Fora"
    : authentication
      ? "Access to Fora requires authentication"
      : startup
        ? "Unable to start Fora"
        : error.code === "protocol_error"
          ? "Fora sent an invalid response"
          : preparation
            ? "Unable to prepare the connection"
            : "Could not connect to Fora";

  return (
    <main className="flex h-dvh w-full overflow-auto bg-surface p-6">
      <div className="m-auto w-full max-w-[520px] min-w-0">
        <Alert
          aria-labelledby={titleId}
          aria-describedby={descriptionId}
          className="[overflow-wrap:anywhere]"
        >
          <CircleAlertIcon
            className="row-span-2 h-5 w-4 text-danger"
            aria-hidden="true"
          />
          <AlertTitle id={titleId}>{title}</AlertTitle>
          <AlertDescription id={descriptionId}>
            <p className="whitespace-pre-wrap">{error.message}</p>
            {authentication ? (
              <p>
                Open Fora from the application, or use the access link provided
                when Fora starts. That link includes the credentials needed to
                connect.
              </p>
            ) : startup ? (
              <p>Close this window and start Fora again.</p>
            ) : reconnect ? (
              <div className="flex flex-wrap items-center gap-2 pt-1">
                <Button onClick={reconnect} disabled={waiting}>
                  {waiting ? (
                    <Spinner label={loadingData ? "Loading" : "Reconnecting"} />
                  ) : null}
                  {loadingData
                    ? waiting
                      ? "Loading…"
                      : "Retry"
                    : waiting
                      ? "Reconnecting…"
                      : "Reconnect"}
                </Button>
              </div>
            ) : null}
          </AlertDescription>
        </Alert>
      </div>
    </main>
  );
}
