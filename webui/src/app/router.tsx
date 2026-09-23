import {
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useMemo,
  useState,
} from "react";

export type SettingsSection =
  | "model"
  | "execution"
  | "agent"
  | "langfuse"
  | "voice";

export type Route =
  | { name: "discussions" }
  | { name: "discussion"; id: number }
  | { name: "members" }
  | { name: "member"; id: number }
  | { name: "library"; path?: string }
  | { name: "document"; path: string }
  | { name: "settings"; section: SettingsSection };

export type NavId = "discussions" | "members" | "library" | "settings";

const NAV_OF: Record<Route["name"], NavId> = {
  discussions: "discussions",
  discussion: "discussions",
  members: "members",
  member: "members",
  library: "library",
  document: "library",
  settings: "settings",
};

export function navIdOf(route: Route): NavId {
  return NAV_OF[route.name];
}

export function pageKeyOf(route: Route): string {
  if ("id" in route) return `${route.name}:${route.id}`;
  if ("path" in route && route.path) return `${route.name}:${route.path}`;
  return route.name;
}

type Router = { route: Route; navigate: (next: Route) => void };

const RouterContext = createContext<Router | null>(null);

export function RouterProvider({ children }: { children: ReactNode }) {
  const [stack, setStack] = useState<Route[]>([{ name: "discussions" }]);

  const navigate = useCallback((next: Route) => {
    setStack((current) => {
      const top = current[current.length - 1];
      if (top.name === next.name && sameParams(top, next)) return current;
      return [...current, next].slice(-24);
    });
  }, []);

  const value = useMemo<Router>(
    () => ({
      route: stack[stack.length - 1],
      navigate,
    }),
    [stack, navigate],
  );

  return (
    <RouterContext.Provider value={value}>{children}</RouterContext.Provider>
  );
}

function paramOf(route: Route): string | number | null {
  if ("id" in route) return route.id;
  if ("path" in route) return route.path ?? null;
  if ("section" in route) return route.section;
  return null;
}

function sameParams(a: Route, b: Route): boolean {
  return paramOf(a) === paramOf(b);
}

export function useRouter(): Router {
  const value = useContext(RouterContext);
  if (!value) throw new Error("useRouter used outside RouterProvider");
  return value;
}

export function useNavigate(): (next: Route) => void {
  return useRouter().navigate;
}
