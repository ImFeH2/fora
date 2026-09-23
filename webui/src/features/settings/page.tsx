import { type SettingsSection, useNavigate } from "@/app/router";
import { Page, PageBody, PageHeader } from "@/components/layout/shell";
import { type TabItem, Tabs } from "@/components/ui/tabs";
import { AgentPanel } from "@/features/settings/agent";
import { ExecutionPanel } from "@/features/settings/execution";
import { LangfusePanel } from "@/features/settings/langfuse";
import { ModelPanel } from "@/features/settings/model";
import { useSettingsSaving } from "@/features/settings/saver";
import { VoicePanel } from "@/features/settings/voice";

const SECTIONS: TabItem<SettingsSection>[] = [
  { id: "model", label: "Model" },
  { id: "execution", label: "Execution" },
  { id: "agent", label: "Agent" },
  { id: "langfuse", label: "Langfuse" },
  { id: "voice", label: "Voice" },
];

function Panel({ section }: { section: SettingsSection }) {
  switch (section) {
    case "model":
      return <ModelPanel />;
    case "execution":
      return <ExecutionPanel />;
    case "agent":
      return <AgentPanel />;
    case "langfuse":
      return <LangfusePanel />;
    case "voice":
      return <VoicePanel />;
  }
}

function SettingsPageContent({ section }: { section: SettingsSection }) {
  const navigate = useNavigate();
  const saving = useSettingsSaving();
  return (
    <Page>
      <PageHeader title="Settings" />
      <PageBody>
        <div className="settings-layout">
          <Tabs
            label="Settings"
            tabs={SECTIONS}
            value={section}
            disabled={saving}
            onChange={(next) => navigate({ name: "settings", section: next })}
          >
            <Panel section={section} />
          </Tabs>
        </div>
      </PageBody>
    </Page>
  );
}

export function SettingsPage({ section }: { section: SettingsSection }) {
  return <SettingsPageContent section={section} />;
}
