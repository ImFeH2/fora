import { Bot, User, Users } from "lucide-react";
import { useId, useState } from "react";
import { useOrganization } from "@/app/organization";
import { Modal } from "@/components/ui/dialog";
import { Avatar, Button, Chip, Input, toast } from "@/components/ui/index";
import { StatePanel } from "@/components/ui/state-panel";
import { backend, type Member } from "@/lib/backend";

export function MemberPicker({
  members,
  selected,
  onChange,
  disabled = false,
}: {
  members: Member[];
  selected: number[];
  onChange: (selected: number[]) => void;
  disabled?: boolean;
}) {
  const id = useId();

  return (
    <fieldset
      className="m-0 flex flex-col gap-2 border-0 p-0"
      disabled={disabled}
    >
      <legend className="p-0 text-xs font-medium text-fg-muted">Members</legend>
      <ul className="flex max-h-[220px] flex-col overflow-y-auto rounded-sm border border-line bg-app">
        {members.length === 0 ? (
          <li>
            <StatePanel
              compact
              icons={[User, Users, Bot]}
              title="No other Members"
              description="Add an Agent from Members to start a Discussion."
            />
          </li>
        ) : null}
        {members.map((member) => (
          <li key={member.id}>
            <label
              className="flex items-center gap-2 px-3 py-2 cursor-pointer transition-[background-color] duration-(--duration-fast) ease-linear hover:bg-surface-hover"
              htmlFor={`${id}-${member.id}`}
            >
              <Input
                id={`${id}-${member.id}`}
                type="checkbox"
                checked={selected.includes(member.id)}
                onChange={(event) =>
                  onChange(
                    event.target.checked
                      ? [...selected, member.id]
                      : selected.filter((id) => id !== member.id),
                  )
                }
              />
              <Avatar memberId={member.id} size="sm" />
              <span className="min-w-0 flex-1 truncate font-medium">
                {member.name}
              </span>
              <Chip tone={member.type === "agent" ? "blue" : "neutral"}>
                {member.type === "agent" ? "Agent" : "Human"}
              </Chip>
            </label>
          </li>
        ))}
      </ul>
    </fieldset>
  );
}

export function DiscussionMembersDialog({
  discussionId,
  memberIds,
  onClose,
  onSaved,
}: {
  discussionId: number;
  memberIds: number[];
  onClose: () => void;
  onSaved: (memberIds: number[]) => void | Promise<void>;
}) {
  const { members } = useOrganization();
  const [selected, setSelected] = useState(memberIds);
  const [busy, setBusy] = useState(false);
  const changed =
    selected.length !== memberIds.length ||
    selected.some((id) => !memberIds.includes(id));

  const save = async () => {
    if (!changed || busy) return;
    setBusy(true);
    try {
      const updated = await backend.setDiscussionMembers(
        discussionId,
        selected,
      );
      await onSaved(updated.member_ids);
      onClose();
    } catch (failure) {
      toast({
        tone: "danger",
        title: "Could not save",
        description:
          failure instanceof Error ? failure.message : String(failure),
      });
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      open
      onOpenChange={(open) => {
        if (!open && !busy) onClose();
      }}
      title="Discussion members"
      footer={
        <>
          <Button disabled={busy} onClick={onClose}>
            Cancel
          </Button>
          <Button variant="primary" disabled={!changed || busy} onClick={save}>
            Save
          </Button>
        </>
      }
    >
      <MemberPicker
        members={members}
        selected={selected}
        onChange={setSelected}
        disabled={busy}
      />
    </Modal>
  );
}
