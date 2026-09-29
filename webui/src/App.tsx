import { BookText, MessagesSquare, Plus, Settings, Users } from "lucide-react";
import {
  type ReactNode,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { OrganizationProvider } from "@/app/organization";
import {
  navIdOf,
  pageKeyOf,
  type Route,
  RouterProvider,
  useNavigate,
  useRouter,
} from "@/app/router";
import {
  Nav,
  NavItem,
  NavSubItem,
  PageTransition,
  Shell,
  Sidebar,
  SidebarBrand,
} from "@/components/layout/shell";
import { AccessError } from "@/components/ui/access-error";
import {
  Badge,
  dismissToast,
  IconButton,
  Spinner,
  StateDot,
  Toaster,
  toast,
} from "@/components/ui/index";
import { TooltipProvider } from "@/components/ui/tooltip";
import { CreateDiscussionDialog } from "@/features/discussions/create";
import { DiscussionsPage } from "@/features/discussions/list";
import { ThreadPage } from "@/features/discussions/thread";
import { DocumentPage } from "@/features/library/document";
import { LibraryPage } from "@/features/library/list";
import { MemberPage } from "@/features/members/detail";
import { MembersPage } from "@/features/members/list";
import { SettingsPage } from "@/features/settings/page";
import {
  SettingsSaveProvider,
  useSettingsSaving,
} from "@/features/settings/saver";
import {
  BackendError,
  backend,
  type DiscussionSummary,
  type Member,
} from "@/lib/backend";
import "@/styles/App.css";

type Loaded = {
  members: Member[];
  humanId: number;
  tokenLimit: number | null;
  discussions: DiscussionSummary[];
};

function View({
  route,
  tokenLimit,
}: {
  route: Route;
  tokenLimit: number | null;
}) {
  switch (route.name) {
    case "discussions":
      return <DiscussionsPage />;
    case "discussion":
      return <ThreadPage id={route.id} />;
    case "members":
      return <MembersPage tokenLimit={tokenLimit} />;
    case "member":
      return <MemberPage id={route.id} />;
    case "library":
      return <LibraryPage path={route.path} />;
    case "document":
      return <DocumentPage path={route.path} />;
    case "settings":
      return <SettingsPage section={route.section} />;
  }
}

const CONNECTION_TOAST = "connection";

function reconnect() {
  void backend.reconnect().catch(reportBackendFailure);
}

function reportBackendFailure(error: BackendError) {
  const uncertain = error.code === "unconfirmed";
  const disconnected = backend.disconnected && !uncertain;
  toast({
    id: disconnected ? CONNECTION_TOAST : undefined,
    tone: "danger",
    title: uncertain
      ? "Check operation result"
      : disconnected
        ? "Connection lost"
        : "Request failed",
    description: error.message,
    duration: disconnected || uncertain ? null : undefined,
    closable: !disconnected,
    action: disconnected
      ? { label: "Reconnect", onClick: reconnect }
      : undefined,
  });
}

function Chrome({ loaded }: { loaded: Loaded }) {
  const { route } = useRouter();
  const navigate = useNavigate();
  const [creating, setCreating] = useState(false);
  const settingsSaving = useSettingsSaving();
  const active = navIdOf(route);
  const unlisted =
    route.name === "discussion" &&
    !loaded.discussions.some((item) => item.id === route.id);
  const running = new Set(
    loaded.members
      .filter((member) => member.type === "agent" && member.state === "running")
      .map((member) => member.id),
  );
  const unread = loaded.discussions.reduce(
    (total, item) => total + item.unread,
    0,
  );

  return (
    <Shell
      sidebar={
        <Sidebar
          inert={settingsSaving}
          footer={
            <Nav label="Settings">
              <NavItem
                icon={<Settings size={16} />}
                label="Settings"
                active={active === "settings"}
                onSelect={() => {
                  if (route.name !== "settings")
                    navigate({ name: "settings", section: "model" });
                }}
              />
            </Nav>
          }
        >
          <SidebarBrand />
          <Nav label="Sections">
            <NavItem
              icon={<MessagesSquare size={16} />}
              label="Discussions"
              active={route.name === "discussions" || unlisted}
              badge={
                unread > 0 ? (
                  <Badge key={unread} tone="unread">
                    {unread}
                  </Badge>
                ) : undefined
              }
              trailing={
                <IconButton
                  size="sm"
                  label="New Discussion"
                  onClick={() => setCreating(true)}
                >
                  <Plus size={15} />
                </IconButton>
              }
              onSelect={() => navigate({ name: "discussions" })}
            >
              {loaded.discussions.map((item) => (
                <NavSubItem
                  key={item.id}
                  label={item.topic}
                  active={route.name === "discussion" && route.id === item.id}
                  badge={
                    item.unread > 0 ? (
                      <Badge key={item.unread} tone="unread">
                        {item.unread}
                      </Badge>
                    ) : undefined
                  }
                  indicator={
                    item.member_ids.some((id) => running.has(id)) ? (
                      <StateDot state="running" />
                    ) : undefined
                  }
                  onSelect={() => navigate({ name: "discussion", id: item.id })}
                />
              ))}
            </NavItem>
            <NavItem
              icon={<Users size={16} />}
              label="Members"
              active={active === "members"}
              onSelect={() => navigate({ name: "members" })}
            />
            <NavItem
              icon={<BookText size={16} />}
              label="Library"
              active={active === "library"}
              onSelect={() => navigate({ name: "library" })}
            />
          </Nav>
        </Sidebar>
      }
    >
      <PageTransition id={pageKeyOf(route)}>
        <View route={route} tokenLimit={loaded.tokenLimit} />
      </PageTransition>
      <CreateDiscussionDialog
        open={creating}
        onOpenChange={setCreating}
        onCreated={(id) => navigate({ name: "discussion", id })}
      />
    </Shell>
  );
}

export function useApplication() {
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [failure, setFailure] = useState<BackendError | null>(null);
  const [reconnecting, setReconnecting] = useState(false);
  const [loading, setLoading] = useState(false);
  const booted = useRef(false);
  const live = useRef(false);
  const generation = useRef(0);
  const pending = useRef<Promise<void> | null>(null);
  const dirty = useRef(false);
  const automaticRecovery = useRef(false);

  const refresh = useCallback((): Promise<void> => {
    if (!live.current) return Promise.resolve();
    if (pending.current) return pending.current;
    const epoch = generation.current;
    setLoading(true);
    const operation = Promise.resolve().then(async () => {
      if (!live.current || epoch !== generation.current) return;
      try {
        do {
          dirty.current = false;
          const [organization, discussions] = await Promise.all([
            backend.organization(),
            backend.discussions(),
          ]);
          if (!live.current || epoch !== generation.current) return;
          setLoaded({
            members: organization.members,
            humanId: organization.human_id,
            tokenLimit: organization.token_limit,
            discussions,
          });
          booted.current = true;
          setFailure(null);
          dismissToast("application-data");
        } while (dirty.current);
      } catch (error) {
        if (!live.current || epoch !== generation.current) return;
        const problem =
          error instanceof BackendError
            ? error
            : new BackendError(
                "request_failed",
                error instanceof Error ? error.message : String(error),
              );
        setFailure(problem);
        if (booted.current)
          toast({
            id: "application-data",
            tone: "danger",
            title: "Could not load organization",
            description: problem.message,
            duration: null,
            action: {
              label: "Retry",
              onClick: () => {
                if (backend.disconnected) reconnect();
                else void refresh();
              },
            },
          });
      } finally {
        if (live.current && epoch === generation.current) {
          pending.current = null;
          setLoading(false);
        }
      }
    });
    pending.current = operation;
    return operation;
  }, []);

  const retry = useCallback((): Promise<void> => {
    if (backend.disconnected)
      return backend.reconnect().catch(reportBackendFailure);
    return refresh();
  }, [refresh]);

  useEffect(() => {
    live.current = true;
    const invalidate = () => {
      generation.current += 1;
      pending.current = null;
      dirty.current = false;
    };
    const recover = () => {
      if (document.visibilityState !== "visible" || !navigator.onLine)
        return false;
      automaticRecovery.current = false;
      const operation = backend.reconnect(true);
      if (!automaticRecovery.current) return false;
      void operation.catch(reportBackendFailure);
      return true;
    };
    const offFailure = backend.onFailure((error) => {
      if (error.code === "disconnected") return;
      if (backend.disconnected) setReconnecting(false);
      if (booted.current) reportBackendFailure(error);
      else if (backend.disconnected) setFailure(error);
    });
    const offEvent = backend.onEvent((event) => {
      if (event.type === "connection.closed") {
        invalidate();
        setLoading(false);
        const recovered = recover();
        if (!recovered && event.error instanceof BackendError) {
          if (booted.current) reportBackendFailure(event.error);
          else setFailure(event.error);
        }
        return;
      }
      if (event.type === "connection.reconnecting") {
        automaticRecovery.current = true;
        setReconnecting(true);
        return;
      }
      if (event.type === "connection.restored") {
        dismissToast(CONNECTION_TOAST);
        setReconnecting(false);
        void refresh();
        return;
      }
      if (
        event.type === "organization.changed" ||
        event.type === "settings.updated" ||
        event.type.startsWith("member.") ||
        event.type.startsWith("turn.") ||
        event.type === "message.created" ||
        event.type === "discussion.read_updated" ||
        event.type === "mention.acked" ||
        event.type === "mention.revoked" ||
        event.type === "discussion.created" ||
        event.type === "discussion.updated"
      ) {
        if (pending.current) dirty.current = true;
        else void refresh();
      }
    });
    window.addEventListener("online", recover);
    document.addEventListener("visibilitychange", recover);
    const epoch = generation.current;
    backend.connect().then(
      () => {
        if (live.current && epoch === generation.current) void refresh();
      },
      (error: unknown) => {
        if (live.current && epoch === generation.current)
          setFailure(
            error instanceof BackendError
              ? error
              : new BackendError(
                  "startup_failed",
                  error instanceof Error ? error.message : String(error),
                ),
          );
      },
    );
    return () => {
      live.current = false;
      invalidate();
      offEvent();
      offFailure();
      window.removeEventListener("online", recover);
      document.removeEventListener("visibilitychange", recover);
      dismissToast("application-data");
      dismissToast(CONNECTION_TOAST);
    };
  }, [refresh]);

  return { loaded, failure, reconnecting, loading, refresh, retry };
}

export default function App() {
  const { loaded, failure, reconnecting, loading, refresh, retry } =
    useApplication();
  let body: ReactNode;
  if (failure && !loaded) {
    body = (
      <AccessError
        error={failure}
        reconnect={() => {
          void retry();
        }}
        waiting={reconnecting || loading}
        loadingData={!backend.disconnected && !reconnecting}
      />
    );
  } else if (!loaded) {
    body = (
      <div className="flex flex-col items-center justify-center gap-3 h-screen bg-surface">
        <Spinner label="Starting Huddol" />
        <p className="text-fg-muted">Starting Huddol…</p>
      </div>
    );
  } else {
    body = (
      <RouterProvider>
        <OrganizationProvider
          value={{
            members: loaded.members,
            humanId: loaded.humanId,
            discussions: loaded.discussions,
            refresh,
          }}
        >
          <SettingsSaveProvider>
            <Chrome loaded={loaded} />
          </SettingsSaveProvider>
        </OrganizationProvider>
      </RouterProvider>
    );
  }

  return (
    <TooltipProvider>
      {body}
      <Toaster />
    </TooltipProvider>
  );
}
