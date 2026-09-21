import { clsx } from "clsx";
import { motion, useReducedMotion } from "framer-motion";
import { LoaderCircle, type LucideIcon } from "lucide-react";
import { type ReactNode, useId } from "react";
import { Button } from "@/components/ui/index";
import "./state-panel.css";

export type StatePanelAction = {
  label: string;
  icon: ReactNode;
  onClick: () => void;
  busy?: boolean;
  disabled?: boolean;
};

export function StateAction({ action }: { action: StatePanelAction }) {
  const reducedMotion = useReducedMotion();
  const disabled = action.disabled || action.busy;
  return (
    <motion.span
      className="state-action"
      tabIndex={-1}
      whileTap={reducedMotion || disabled ? undefined : { scale: 0.98 }}
    >
      <Button
        onClick={action.onClick}
        disabled={disabled}
        aria-busy={action.busy || undefined}
      >
        <span
          className={clsx("state-action-icon", action.busy && "is-busy")}
          aria-hidden="true"
        >
          {action.busy ? <LoaderCircle size={16} /> : action.icon}
        </span>
        <span>{action.label}</span>
      </Button>
    </motion.span>
  );
}

type StateIcons = readonly [LucideIcon, LucideIcon, LucideIcon];

export function StatePanel({
  icons,
  title,
  description,
  action,
  secondaryAction,
  compact = false,
}: {
  icons: StateIcons;
  title: string;
  description?: string;
  action?: StatePanelAction;
  secondaryAction?: StatePanelAction;
  compact?: boolean;
}) {
  const titleId = useId();
  const descriptionId = useId();
  const [LeftIcon, CenterIcon, RightIcon] = icons;
  return (
    <section
      className={clsx("state-panel", compact && "state-panel-compact")}
      aria-labelledby={titleId}
      aria-describedby={description ? descriptionId : undefined}
    >
      {compact ? (
        <CenterIcon
          size={20}
          className="state-compact-icon"
          aria-hidden="true"
        />
      ) : (
        <div className="state-icons" aria-hidden="true">
          <span>
            <LeftIcon size={24} />
          </span>
          <span>
            <CenterIcon size={24} />
          </span>
          <span>
            <RightIcon size={24} />
          </span>
        </div>
      )}
      <h3 id={titleId}>{title}</h3>
      {description ? (
        <p id={descriptionId} className="state-description">
          {description}
        </p>
      ) : null}
      {action || secondaryAction ? (
        <div className="state-actions">
          {action ? <StateAction action={action} /> : null}
          {secondaryAction ? <StateAction action={secondaryAction} /> : null}
        </div>
      ) : null}
    </section>
  );
}
