import { clsx } from "clsx";
import { ArrowLeft } from "lucide-react";
import { Children, type ReactNode, useRef } from "react";
import brandIcon from "../../../../assets/icon.svg";

const crumbClasses =
  "inline-flex items-center gap-1 self-start min-h-6 py-0 pr-2 pl-1 border-0 rounded-xs bg-transparent text-fg-muted text-xs font-medium cursor-pointer transition-[background-color,color] duration-(--duration-fast) ease-linear hover:bg-gray-800 hover:text-fg";

const pageBodyVariants = {
  scroll: "gap-4 overflow-y-auto pt-0 px-8 pb-8 max-[940px]:px-6",
  flush: "gap-0 overflow-hidden",
};

const columnVisibility = {
  md: "max-[1120px]:hidden",
  sm: "max-[940px]:hidden",
};

export function Shell({
  sidebar,
  children,
}: {
  sidebar: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="grid grid-cols-[240px_minmax(0,1fr)] h-full bg-surface">
      {sidebar}
      <div className="flex flex-col min-w-0 min-h-0">{children}</div>
    </div>
  );
}

export function Sidebar({
  children,
  footer,
  inert = false,
}: {
  children: ReactNode;
  footer?: ReactNode;
  inert?: boolean;
}) {
  return (
    <div
      className="flex flex-col min-h-0 border-r border-line bg-app"
      inert={inert}
    >
      <div className="flex flex-col flex-1 min-h-0 overflow-y-auto">
        {children}
      </div>
      {footer ? (
        <div className="flex-none border-t border-line">{footer}</div>
      ) : null}
    </div>
  );
}

export function SidebarBrand() {
  return (
    <div className="flex items-center gap-3 pt-4 px-4 pb-3 min-w-0">
      <img src={brandIcon} alt="" className="size-7 flex-none" />
      <span className="text-sm font-semibold truncate">Fora</span>
    </div>
  );
}

export function Nav({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  const container = useRef<HTMLElement>(null);

  const move = (event: React.KeyboardEvent) => {
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    const items = Array.from(
      container.current?.querySelectorAll<HTMLButtonElement>(
        "[data-nav]:not([disabled])",
      ) ?? [],
    ).filter((item) => item.offsetParent !== null);
    if (items.length === 0) return;
    const index = items.indexOf(document.activeElement as HTMLButtonElement);
    if (index < 0) return;
    event.preventDefault();
    const delta = event.key === "ArrowDown" ? 1 : -1;
    items[(index + delta + items.length) % items.length].focus();
  };

  return (
    <nav
      className="flex flex-col p-2"
      aria-label={label}
      ref={container}
      onKeyDown={move}
    >
      <ul className="flex flex-col gap-px min-h-0">{children}</ul>
    </nav>
  );
}

export function NavItem({
  icon,
  label,
  active,
  badge,
  trailing,
  onSelect,
  children,
}: {
  icon: ReactNode;
  label: string;
  active: boolean;
  badge?: ReactNode;
  trailing?: ReactNode;
  onSelect: () => void;
  children?: ReactNode;
}) {
  const nested = Children.toArray(children);
  return (
    <li>
      <div
        className={clsx(
          "group relative flex items-center rounded-sm transition-[background-color] duration-(--duration-fast) ease-linear hover:bg-gray-800 before:content-[''] before:absolute before:left-0 before:top-1/2 before:w-[2px] before:rounded-r-[9999px] before:bg-blue-300 before:-translate-y-1/2 before:transition-[height] before:duration-(--duration-base) before:ease-transform",
          active ? "bg-gray-800 before:h-4" : "before:h-0",
        )}
        data-active={active}
      >
        <button
          type="button"
          className={clsx(
            "flex-1 min-w-0 flex items-center gap-2 h-[30px] py-0 pr-2 pl-4 border-0 rounded-sm bg-transparent text-sm font-medium text-left cursor-pointer transition-[color] duration-(--duration-fast) ease-linear group-hover:text-fg",
            active ? "text-fg" : "text-fg-muted",
          )}
          data-nav="item"
          aria-current={active ? "page" : undefined}
          onClick={onSelect}
        >
          <span
            className={clsx(
              "flex flex-none",
              active ? "text-blue-300" : "text-fg-muted",
            )}
            aria-hidden="true"
          >
            {icon}
          </span>
          <span className="flex-1 min-w-0 truncate">{label}</span>
          {badge}
        </button>
        {trailing ? (
          <span className="flex flex-none pr-0.5">{trailing}</span>
        ) : null}
      </div>
      {nested.length > 0 ? (
        <ul className="flex flex-col gap-px pt-px px-0 pb-1 *:animate-fade-in">
          {nested}
        </ul>
      ) : null}
    </li>
  );
}

export function NavSubItem({
  label,
  active,
  badge,
  indicator,
  onSelect,
}: {
  label: string;
  active: boolean;
  badge?: ReactNode;
  indicator?: ReactNode;
  onSelect: () => void;
}) {
  return (
    <li>
      <button
        type="button"
        className={clsx(
          "relative flex items-center gap-2 w-full h-[26px] py-0 pr-2 pl-10 border-0 rounded-sm text-sm text-left cursor-pointer transition-[background-color,color] duration-(--duration-fast) ease-linear hover:bg-gray-800 hover:text-fg data-[unread=true]:font-medium data-[unread=true]:text-fg",
          active
            ? "bg-gray-800 text-fg font-medium"
            : "bg-transparent text-fg-muted",
        )}
        data-nav="subitem"
        aria-current={active ? "page" : undefined}
        data-unread={badge ? "true" : undefined}
        onClick={onSelect}
      >
        {indicator ? (
          <span className="absolute left-6 top-1/2 flex -translate-x-1/2 -translate-y-1/2">
            {indicator}
          </span>
        ) : null}
        <span className="flex-1 min-w-0 truncate">{label}</span>
        {badge}
      </button>
    </li>
  );
}

export function Page({ children }: { children: ReactNode }) {
  return (
    <main className="flex-1 flex flex-col min-w-0 min-h-0 bg-surface">
      {children}
    </main>
  );
}

export function PageTransition({
  id,
  children,
}: {
  id: string;
  children: ReactNode;
}) {
  return (
    <div
      className="flex-1 flex flex-col min-w-0 min-h-0 animate-rise-in"
      key={id}
    >
      {children}
    </div>
  );
}

export type Crumb = { label: string; onSelect: () => void };

export function PageHeader({
  title,
  status,
  actions,
  crumb,
  leading,
}: {
  title: ReactNode;
  status?: ReactNode;
  actions?: ReactNode;
  crumb?: Crumb | Crumb[];
  leading?: ReactNode;
}) {
  return (
    <header className="flex flex-col gap-2 pt-6 px-8 pb-4 flex-none max-[940px]:px-6">
      {Array.isArray(crumb) ? (
        <nav aria-label="Breadcrumb">
          <ol className="flex items-center flex-wrap gap-2 text-fg-muted text-xs wrap-anywhere">
            {crumb.map((item, index) => (
              <li
                className="flex items-center flex-wrap gap-2 text-fg-muted text-xs wrap-anywhere"
                key={`${index}:${item.label}`}
              >
                {index > 0 ? <span aria-hidden="true">›</span> : null}
                {index === crumb.length - 1 ? (
                  <span aria-current="page">{item.label}</span>
                ) : (
                  <button
                    type="button"
                    className={clsx(crumbClasses, "m-0")}
                    onClick={item.onSelect}
                  >
                    {item.label}
                  </button>
                )}
              </li>
            ))}
          </ol>
        </nav>
      ) : crumb ? (
        <button
          type="button"
          className={clsx(crumbClasses, "-ml-1")}
          onClick={crumb.onSelect}
        >
          <ArrowLeft size={14} aria-hidden="true" />
          {crumb.label}
        </button>
      ) : null}
      <div className="flex items-start gap-4">
        {leading}
        <div className="flex-1 min-w-0 flex flex-col gap-1">
          <h1 className="wrap-anywhere">{title}</h1>
          {status ? (
            <div className="flex items-center gap-2 text-fg-muted">
              {status}
            </div>
          ) : null}
        </div>
        {actions ? (
          <div className="flex items-center gap-2 flex-none pt-0.5">
            {actions}
          </div>
        ) : null}
      </div>
    </header>
  );
}

export function Toolbar({ children }: { children: ReactNode }) {
  return (
    <div className="flex items-center gap-2 pt-0 px-8 pb-4 flex-none max-[940px]:px-6">
      {children}
    </div>
  );
}

export function PageBody({
  children,
  variant = "scroll",
}: {
  children: ReactNode;
  variant?: "scroll" | "flush";
}) {
  return (
    <div
      className={clsx(
        "flex-1 min-h-0 flex flex-col",
        pageBodyVariants[variant],
      )}
    >
      {children}
    </div>
  );
}

export type Column = {
  key: string;
  label: string;
  align?: "start" | "end";
  width?: string;
  hideBelow?: "sm" | "md";
};

export function Table({
  columns,
  children,
  label,
}: {
  columns: Column[];
  children: ReactNode;
  label: string;
}) {
  return (
    <div className="w-full">
      <table
        className="w-full table-auto [&_td]:p-3 [&_td]:border-b [&_td]:border-line [&_td]:align-middle [&_td[data-align=end]]:text-right max-[1120px]:[&_td[data-hide-below=md]]:hidden max-[940px]:[&_td[data-hide-below=sm]]:hidden [&_tbody_tr]:relative [&_tbody_tr]:transition-[background-color] [&_tbody_tr]:duration-(--duration-fast) [&_tbody_tr]:ease-linear [&_tbody_tr:hover]:bg-gray-800/70 [&_tbody_tr[data-highlight=true]]:bg-blue-500/7"
        aria-label={label}
      >
        <thead>
          <tr>
            {columns.map((column) => (
              <th
                key={column.key}
                scope="col"
                className={clsx(
                  "py-2 px-3 border-b border-line text-fg-muted text-xs font-bold leading-4 tracking-caps uppercase whitespace-nowrap",
                  column.align === "end" ? "text-right" : "text-left",
                  column.hideBelow && columnVisibility[column.hideBelow],
                )}
                data-align={column.align ?? "start"}
                data-hide-below={column.hideBelow}
                style={column.width ? { width: column.width } : undefined}
              >
                {column.label ? (
                  column.label
                ) : (
                  <span className="sr-only">Actions</span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}

export function RowLink({
  primary,
  secondary,
  onSelect,
  expanded,
}: {
  primary: ReactNode;
  secondary?: ReactNode;
  onSelect: () => void;
  expanded?: boolean;
}) {
  return (
    <button
      type="button"
      className="flex flex-col gap-px min-w-0 p-0 border-0 rounded-xs bg-transparent text-inherit text-left cursor-pointer after:content-[''] after:absolute after:inset-0"
      onClick={onSelect}
      aria-expanded={expanded}
    >
      <span className="font-semibold text-fg truncate transition-[color] duration-(--duration-fast) ease-linear [tr:hover_&]:text-blue-100">
        {primary}
      </span>
      {secondary ? (
        <span className="text-xs text-fg-muted truncate">{secondary}</span>
      ) : null}
    </button>
  );
}

export function Section({
  title,
  actions,
  children,
}: {
  title: string;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="flex flex-col gap-3">
      <div className="flex items-end justify-between gap-4">
        <div className="flex flex-col gap-0.5 min-w-0">
          <h2 className="text-sm font-semibold tracking-body">{title}</h2>
        </div>
        {actions ? <div className="flex gap-2 flex-none">{actions}</div> : null}
      </div>
      {children}
    </section>
  );
}
