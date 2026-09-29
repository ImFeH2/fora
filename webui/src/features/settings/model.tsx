import { Save } from "lucide-react";
import { useCallback, useEffect, useId, useRef, useState } from "react";
import { Combobox } from "@/components/ui/combobox";
import { Modal } from "@/components/ui/dialog";
import {
  Button,
  Chip,
  Field,
  Input,
  Spinner,
  toast,
} from "@/components/ui/index";
import { Segmented } from "@/components/ui/segmented";
import {
  reportLoadFailure,
  useReportSettingsSave,
} from "@/features/settings/saver";
import {
  type AgentModelConfig,
  BackendError,
  backend,
  type ModelCatalog,
  type ProviderConfig,
  type RegisteredModel,
  type Thinking,
} from "@/lib/backend";

const PROVIDERS = [
  { value: "openai-chat", label: "OpenAI Chat" },
  { value: "openai-responses", label: "OpenAI Responses" },
  { value: "anthropic", label: "Anthropic" },
  { value: "google", label: "Google" },
];

const THINKING_OPTIONS: Thinking[] = [
  "default",
  "none",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
];

function supportsTokenBudget(apiType: string): boolean {
  return apiType === "anthropic" || apiType === "google";
}

function thinkingLabel(option: Thinking, budget?: number | null): string {
  return option === "budget" ? `budget · ${budget} tokens` : option;
}

export function modelTestDescription(latency: number, reply: string): string {
  return `${latency.toLocaleString("en-US")} ms · ${reply.trim().replace(/\s+/g, " ").slice(0, 80)}`;
}

const selectClass =
  "w-full rounded-md border border-line bg-surface px-3 py-2 text-sm text-fg";

export function ModelSelection({
  catalog,
  value,
  onChange,
  inherit = true,
}: {
  catalog: ModelCatalog;
  value: AgentModelConfig;
  onChange: (value: AgentModelConfig) => void;
  inherit?: boolean;
}) {
  const id = useId();
  const selected = catalog.models.find(
    (model) =>
      model.id ===
      (value.model_id ?? (inherit ? catalog.default_model_id : null)),
  );
  const options = selected?.thinking_options ?? THINKING_OPTIONS;
  return (
    <div className="flex flex-col gap-4">
      <Field label="Model" htmlFor={`${id}-model`}>
        <select
          id={`${id}-model`}
          className={selectClass}
          value={value.model_id ?? ""}
          onChange={(event) =>
            onChange({ ...value, model_id: event.target.value || null })
          }
        >
          <option value="">
            {inherit ? "Use global default" : "Not configured"}
          </option>
          {catalog.providers.map((provider) => (
            <optgroup
              key={provider.id}
              label={provider.name}
              disabled={!provider.enabled}
            >
              {catalog.models
                .filter((model) => model.provider_id === provider.id)
                .map((model) => (
                  <option
                    key={model.id}
                    value={model.id}
                    disabled={!model.enabled}
                  >
                    {model.name}
                  </option>
                ))}
            </optgroup>
          ))}
        </select>
      </Field>
      <Field label="Thinking effort" htmlFor={`${id}-thinking`}>
        <select
          id={`${id}-thinking`}
          className={selectClass}
          value={value.thinking ?? ""}
          onChange={(event) =>
            onChange({
              ...value,
              thinking: (event.target.value || null) as Thinking | null,
            })
          }
        >
          {inherit ? (
            <option value="">
              Use global default ({catalog.default_thinking})
            </option>
          ) : null}
          {value.thinking && !options.includes(value.thinking) ? (
            <option value={value.thinking} disabled>
              {value.thinking} · unavailable for this model
            </option>
          ) : null}
          {options.map((option) => (
            <option key={option} value={option}>
              {thinkingLabel(option, selected?.thinking_budget_tokens)}
            </option>
          ))}
        </select>
      </Field>
      <p className="text-sm text-fg-muted">
        {selected
          ? `Effective model: ${catalog.providers.find((provider) => provider.id === selected.provider_id)?.name} / ${selected.name}. Changes apply to the next Turn.`
          : "No model selected. Configure a provider and model before this Agent runs."}
      </p>
    </div>
  );
}

export function AgentCreateDialog({
  onClose,
  onCreated,
}: {
  onClose: () => void;
  onCreated: () => Promise<void>;
}) {
  const id = useId();
  const [catalog, setCatalog] = useState<ModelCatalog | null>(null);
  const [name, setName] = useState("");
  const [value, setValue] = useState<AgentModelConfig>({
    model_id: null,
    thinking: null,
  });
  const [busy, setBusy] = useState(false);
  const load = useCallback(async () => {
    try {
      setCatalog(await backend.modelCatalog());
    } catch (error) {
      reportLoadFailure("create-agent-models", error, () => void load());
    }
  }, []);
  useEffect(() => {
    void load();
    return backend.onEvent((event) => {
      if (event.type === "connection.restored") void load();
    });
  }, [load]);
  const create = async () => {
    if (!catalog || !name.trim() || busy) return;
    setBusy(true);
    let created = false;
    try {
      await backend.createAgent(name.trim(), value);
      created = true;
    } catch (error) {
      if (error instanceof BackendError && error.transport)
        backend.reportFailure(error);
      else
        toast({
          tone: "danger",
          title: "Could not create Agent",
          description: error instanceof Error ? error.message : String(error),
        });
    } finally {
      setBusy(false);
    }
    if (!created) return;
    onClose();
    try {
      await onCreated();
    } catch (error) {
      backend.reportFailure(error);
    }
  };
  return (
    <Modal
      open
      onOpenChange={(open) => {
        if (!open && !busy) onClose();
      }}
      title="New Agent"
      footer={
        <>
          <Button disabled={busy} onClick={onClose}>
            Cancel
          </Button>
          <Button
            variant="primary"
            disabled={!catalog || !name.trim() || busy}
            onClick={() => void create()}
          >
            {busy ? <Spinner label="Creating Agent" /> : null}
            Create Agent
          </Button>
        </>
      }
    >
      <fieldset
        className="flex flex-col gap-4 border-0 p-0"
        disabled={busy || !catalog}
      >
        <Field label="Name" htmlFor={id}>
          <Input
            id={id}
            value={name}
            onChange={(event) => setName(event.target.value)}
          />
        </Field>
        {catalog ? (
          <ModelSelection catalog={catalog} value={value} onChange={setValue} />
        ) : (
          <Spinner label="Loading models" />
        )}
      </fieldset>
    </Modal>
  );
}

export function AgentModelPanel({ agentId }: { agentId: number }) {
  const [catalog, setCatalog] = useState<ModelCatalog | null>(null);
  const [value, setValue] = useState<AgentModelConfig>({
    model_id: null,
    thinking: null,
  });
  const [busy, setBusy] = useState(false);
  const dirty = useRef(false);
  useReportSettingsSave(`agent-model-${agentId}`, busy);
  const loadGeneration = useRef(0);
  const load = useCallback(async () => {
    const generation = ++loadGeneration.current;
    try {
      const loaded = await backend.modelCatalog();
      if (generation !== loadGeneration.current) return;
      setCatalog(loaded);
      if (!dirty.current) {
        setValue(
          loaded.agent_configs[String(agentId)] ?? {
            model_id: null,
            thinking: null,
          },
        );
      }
    } catch (error) {
      if (generation === loadGeneration.current) {
        reportLoadFailure(`agent-model:${agentId}`, error, () => void load());
      }
    }
  }, [agentId]);
  useEffect(() => {
    void load();
    const off = backend.onEvent((event) => {
      if (event.type === "connection.restored") void load();
    });
    return () => {
      loadGeneration.current += 1;
      off();
    };
  }, [load]);
  return (
    <form
      className="flex max-w-[560px] flex-col gap-4"
      onChangeCapture={() => {
        dirty.current = true;
      }}
      onSubmit={async (event) => {
        event.preventDefault();
        if (!catalog || busy) return;
        loadGeneration.current += 1;
        setBusy(true);
        try {
          const updated = await backend.configureModel(
            "set_agent",
            agentId,
            value,
          );
          loadGeneration.current += 1;
          setCatalog(updated);
          setValue(updated.agent_configs[String(agentId)] ?? value);
          dirty.current = false;
        } catch (error) {
          if (error instanceof BackendError && error.transport)
            backend.reportFailure(error);
          else
            toast({
              tone: "danger",
              title: "Could not save Agent model settings",
              description:
                error instanceof Error ? error.message : String(error),
            });
        } finally {
          setBusy(false);
        }
      }}
    >
      <h2 className="text-base font-medium">Model settings</h2>
      <fieldset
        className="flex flex-col gap-4 border-0 p-0"
        disabled={!catalog || busy}
      >
        {catalog ? (
          <ModelSelection catalog={catalog} value={value} onChange={setValue} />
        ) : (
          <Spinner label="Loading models" />
        )}
        <Button type="submit" variant="primary">
          {busy ? <Spinner label="Saving Agent model settings" /> : null}
          Save model settings
        </Button>
      </fieldset>
    </form>
  );
}

type SaveModel = (
  action: string,
  id: string | number | null,
  values?: Record<string, unknown>,
) => Promise<boolean>;

export function ModelPanel() {
  const [catalog, setCatalog] = useState<ModelCatalog | null>(null);
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState(false);
  const [providerId, setProviderId] = useState<string | null>(null);
  const [expandedModels, setExpandedModels] = useState<string[]>([]);
  const selectionId = useId();
  useReportSettingsSave("model", busy);
  const loadGeneration = useRef(0);
  const load = useCallback(async () => {
    const generation = ++loadGeneration.current;
    try {
      const loaded = await backend.modelCatalog();
      if (generation !== loadGeneration.current) return;
      setCatalog(loaded);
      setFailed(false);
    } catch (error) {
      if (generation === loadGeneration.current) {
        setFailed(true);
        reportLoadFailure("settings-model", error, () => void load());
      }
    }
  }, []);
  useEffect(() => {
    void load();
    const off = backend.onEvent((event) => {
      if (event.type === "connection.restored") void load();
    });
    return () => {
      loadGeneration.current += 1;
      off();
    };
  }, [load]);
  const save: SaveModel = async (action, id, values = {}) => {
    loadGeneration.current += 1;
    setBusy(true);
    try {
      const updated = await backend.configureModel(action, id, values);
      loadGeneration.current += 1;
      setCatalog(updated);
      setFailed(false);
      if (action === "save_provider" && id === null) {
        const created = updated.providers[updated.providers.length - 1];
        if (!created) throw new Error("Saved provider is missing");
        setProviderId(created.id);
      }
      return true;
    } catch (error) {
      if (error instanceof BackendError && error.transport)
        backend.reportFailure(error);
      else
        toast({
          tone: "danger",
          title: "Could not save model settings",
          description: error instanceof Error ? error.message : String(error),
        });
      return false;
    } finally {
      setBusy(false);
    }
  };
  const provider = catalog?.providers.find((item) => item.id === providerId);
  const models =
    catalog?.models.filter((item) => item.provider_id === provider?.id) ?? [];
  const chooseProvider = (id: string | null) => {
    if (busy) return;
    setProviderId(id);
    setExpandedModels([]);
  };
  const toggleModel = (id: string) => {
    if (busy) return;
    setExpandedModels((current) =>
      current.includes(id)
        ? current.filter((item) => item !== id)
        : [...current, id],
    );
  };
  return (
    <fieldset
      className="model-settings m-0 flex min-w-0 flex-col gap-6 border-0 p-0"
      aria-label="Model settings"
      disabled={!catalog || busy || failed}
    >
      {!catalog ? (
        <Spinner label="Loading model settings" />
      ) : (
        <>
          <DefaultModelForm catalog={catalog} save={save} busy={busy} />
          <section className="provider-card">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h2 className="text-base font-medium">Provider connection</h2>
              <Field label="Provider" htmlFor={`${selectionId}-provider`}>
                <select
                  id={`${selectionId}-provider`}
                  className={selectClass}
                  value={provider?.id ?? ""}
                  onChange={(event) =>
                    chooseProvider(event.target.value || null)
                  }
                >
                  <option value="">Add provider</option>
                  {catalog.providers.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.name}
                      {item.enabled ? "" : " · disabled"}
                    </option>
                  ))}
                </select>
              </Field>
            </div>
            <ProviderForm
              key={provider?.id ?? "new"}
              provider={provider}
              save={save}
              busy={busy}
              onDeleted={() => chooseProvider(null)}
            />
          </section>
          {provider ? (
            <section className="models-section">
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <h2 className="text-base font-medium">Models</h2>
                <p className="text-sm text-fg-muted">{provider.name}</p>
              </div>
              {models.length === 0 ? (
                <p className="text-sm text-fg-muted">No models configured.</p>
              ) : null}
              {models.map((model) => {
                const expanded = expandedModels.includes(model.id);
                return (
                  <section className="model-item" key={model.id}>
                    <button
                      className="model-summary"
                      type="button"
                      aria-expanded={expanded}
                      aria-controls={`${selectionId}-model-${model.id}`}
                      onClick={() => toggleModel(model.id)}
                    >
                      <span className="min-w-0 flex flex-col gap-1">
                        <span className="font-medium wrap-anywhere">
                          {model.name}
                        </span>
                        <span className="text-xs text-fg-muted wrap-anywhere">
                          {model.model}
                        </span>
                      </span>
                      <span className="flex items-center gap-2 flex-none">
                        <Chip>{model.enabled ? "Enabled" : "Disabled"}</Chip>
                        <span>{expanded ? "Close" : "Edit"}</span>
                      </span>
                    </button>
                    <div
                      id={`${selectionId}-model-${model.id}`}
                      className="model-editor"
                      hidden={!expanded}
                      inert={!expanded}
                    >
                      <ModelForm
                        provider={provider}
                        model={model}
                        save={save}
                        busy={busy}
                      />
                    </div>
                  </section>
                );
              })}
              <section className="model-item">
                <button
                  className="model-add"
                  type="button"
                  aria-expanded={expandedModels.includes("new")}
                  aria-controls={`${selectionId}-model-new`}
                  onClick={() => toggleModel("new")}
                >
                  Add model
                </button>
                <div
                  id={`${selectionId}-model-new`}
                  className="model-editor"
                  hidden={!expandedModels.includes("new")}
                  inert={!expandedModels.includes("new")}
                >
                  <ModelForm
                    key={provider.id}
                    provider={provider}
                    save={save}
                    busy={busy}
                  />
                </div>
              </section>
            </section>
          ) : null}
        </>
      )}
    </fieldset>
  );
}

function DefaultModelForm({
  catalog,
  save,
  busy,
}: {
  catalog: ModelCatalog;
  save: SaveModel;
  busy: boolean;
}) {
  const [value, setValue] = useState<AgentModelConfig>({
    model_id: catalog.default_model_id,
    thinking: catalog.default_thinking,
  });
  const dirty = useRef(false);
  useEffect(() => {
    if (!dirty.current) {
      setValue({
        model_id: catalog.default_model_id,
        thinking: catalog.default_thinking,
      });
    }
  }, [catalog.default_model_id, catalog.default_thinking]);
  return (
    <form
      className="default-model-card flex flex-col gap-4"
      onChangeCapture={() => {
        dirty.current = true;
      }}
      onSubmit={async (event) => {
        event.preventDefault();
        if (await save("set_defaults", null, value)) dirty.current = false;
      }}
    >
      <h2 className="text-base font-medium">Global defaults</h2>
      <ModelSelection
        catalog={catalog}
        value={value}
        onChange={setValue}
        inherit={false}
      />
      <Button type="submit" variant="primary">
        {busy ? <Spinner label="Saving model defaults" /> : <Save size={16} />}
        Save defaults
      </Button>
    </form>
  );
}

function ProviderForm({
  provider,
  save,
  busy,
  onDeleted,
}: {
  provider?: ProviderConfig;
  save: SaveModel;
  busy: boolean;
  onDeleted: () => void;
}) {
  const id = useId();
  const [name, setName] = useState(provider?.name ?? "");
  const [apiType, setApiType] = useState(provider?.api_type ?? "openai-chat");
  const [baseUrl, setBaseUrl] = useState(provider?.base_url ?? "");
  const [apiKey, setApiKey] = useState("");
  const [enabled, setEnabled] = useState(provider?.enabled ?? true);
  const [clearKey, setClearKey] = useState(false);
  const dirty = useRef(false);
  useEffect(() => {
    if (!dirty.current) {
      setName(provider?.name ?? "");
      setApiType(provider?.api_type ?? "openai-chat");
      setBaseUrl(provider?.base_url ?? "");
      setEnabled(provider?.enabled ?? true);
    }
  }, [provider]);
  return (
    <form
      className="provider-form flex flex-col gap-4"
      onChangeCapture={() => {
        dirty.current = true;
      }}
      onSubmit={async (event) => {
        event.preventDefault();
        if (
          await save("save_provider", provider?.id ?? null, {
            name: name.trim(),
            api_type: apiType,
            base_url: baseUrl.trim(),
            api_key: apiKey.trim(),
            enabled,
            ...(provider ? { clear_key: clearKey } : {}),
          })
        ) {
          dirty.current = false;
          setApiKey("");
          setClearKey(false);
        }
      }}
    >
      <Field label="Provider name" htmlFor={`${id}-name`}>
        <Input
          id={`${id}-name`}
          value={name}
          required
          onChange={(event) => setName(event.target.value)}
        />
      </Field>
      <Field label="API type">
        <Segmented
          label="API type"
          value={apiType}
          options={PROVIDERS}
          onChange={(value) => {
            dirty.current = true;
            setApiType(value);
          }}
        />
      </Field>
      <Field label="Base URL" htmlFor={`${id}-url`}>
        <Input
          id={`${id}-url`}
          value={baseUrl}
          placeholder="https://api.example.com/v1"
          onChange={(event) => setBaseUrl(event.target.value)}
        />
      </Field>
      <Field
        label="API key"
        htmlFor={`${id}-key`}
        hint={
          provider?.api_key_set
            ? "Leave blank to keep the stored key."
            : undefined
        }
      >
        <Input
          id={`${id}-key`}
          type="password"
          autoComplete="off"
          value={apiKey}
          onChange={(event) => setApiKey(event.target.value)}
        />
      </Field>
      {provider?.api_key_set ? (
        <>
          <Chip tone="success">Key stored</Chip>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={clearKey}
              onChange={(event) => setClearKey(event.target.checked)}
            />
            Clear stored key
          </label>
        </>
      ) : null}
      <label className="flex items-center gap-2 text-sm">
        <input
          type="checkbox"
          checked={enabled}
          onChange={(event) => setEnabled(event.target.checked)}
        />
        Enabled
      </label>
      <div className="flex gap-2">
        <Button type="submit" variant="primary">
          {busy ? <Spinner label="Saving provider" /> : null}
          {provider ? "Save provider" : "Create provider"}
        </Button>
        {provider ? (
          <Button
            onClick={async () => {
              if (await save("delete_provider", provider.id)) onDeleted();
            }}
          >
            Delete provider
          </Button>
        ) : null}
      </div>
    </form>
  );
}

function ModelForm({
  provider,
  model,
  save,
  busy,
}: {
  provider: ProviderConfig;
  model?: RegisteredModel;
  save: SaveModel;
  busy: boolean;
}) {
  const id = useId();
  const [name, setName] = useState(model?.name ?? "");
  const [remote, setRemote] = useState(model?.model ?? "");
  const [enabled, setEnabled] = useState(model?.enabled ?? true);
  const [budgetTokens, setBudgetTokens] = useState(
    model?.thinking_budget_tokens?.toString() ?? "",
  );
  const [options, setOptions] = useState<string[]>([]);
  const [listing, setListing] = useState(false);
  const [testing, setTesting] = useState(false);
  const [thinking, setThinking] = useState<Thinking>("default");
  const budgetSupported = supportsTokenBudget(provider.api_type);
  const parsedBudget = budgetTokens.trim() ? Number(budgetTokens) : null;
  const validBudget =
    parsedBudget === null ||
    (Number.isSafeInteger(parsedBudget) && parsedBudget > 0);
  const budgetChanged =
    model !== undefined && parsedBudget !== model.thinking_budget_tokens;
  const validTestSelection =
    validBudget &&
    (thinking !== "budget" || (budgetSupported && parsedBudget !== null));
  const testThinkingOptions: Thinking[] =
    budgetSupported && parsedBudget !== null && validBudget
      ? [...THINKING_OPTIONS, "budget"]
      : THINKING_OPTIONS;
  const dirty = useRef(false);
  const listRequest = useRef(0);
  useEffect(() => {
    if (!dirty.current) {
      setName(model?.name ?? "");
      setRemote(model?.model ?? "");
      setEnabled(model?.enabled ?? true);
      setBudgetTokens(model?.thinking_budget_tokens?.toString() ?? "");
    }
  }, [model]);
  useEffect(() => {
    listRequest.current += 1;
    setOptions([]);
    setListing(false);
    return () => {
      listRequest.current += 1;
    };
  }, [provider]);
  const listModels = async () => {
    const request = ++listRequest.current;
    setListing(true);
    try {
      const result = await backend.call<{ models: string[] }>(
        "settings.list_models",
        { provider_id: provider.id },
      );
      if (request === listRequest.current) setOptions(result.models);
    } catch (error) {
      if (request === listRequest.current) {
        if (error instanceof BackendError && error.transport)
          backend.reportFailure(error);
        else
          toast({
            tone: "danger",
            title: "Could not load models",
            description: error instanceof Error ? error.message : String(error),
          });
      }
    } finally {
      if (request === listRequest.current) setListing(false);
    }
  };
  const testModel = async () => {
    if (!validTestSelection || !remote.trim()) return;
    setTesting(true);
    try {
      const result = await backend.call<{ latency_ms: number; reply: string }>(
        "settings.test_model",
        model
          ? { model_id: model.id, thinking }
          : {
              provider_id: provider.id,
              model: remote.trim(),
              thinking,
              thinking_budget_tokens: budgetSupported ? parsedBudget : null,
            },
      );
      toast({
        tone: "success",
        title: "Model responded",
        description: modelTestDescription(result.latency_ms, result.reply),
      });
    } catch (error) {
      if (error instanceof BackendError && error.transport)
        backend.reportFailure(error);
      else
        toast({
          tone: "danger",
          title: "Model test failed",
          description: error instanceof Error ? error.message : String(error),
        });
    } finally {
      setTesting(false);
    }
  };
  return (
    <form
      className="model-form flex flex-col gap-4"
      onChangeCapture={() => {
        dirty.current = true;
      }}
      onSubmit={async (event) => {
        event.preventDefault();
        if (!validBudget) {
          toast({
            tone: "danger",
            title: "Invalid thinking budget",
            description: "Enter a positive whole number.",
          });
          return;
        }
        if (
          await save("save_model", model?.id ?? null, {
            name: name.trim() || remote.trim(),
            model: remote.trim(),
            provider_id: provider.id,
            enabled,
            thinking_budget_tokens: budgetSupported ? parsedBudget : null,
          })
        ) {
          dirty.current = false;
          if (!model) {
            setName("");
            setRemote("");
            setBudgetTokens("");
            setThinking("default");
          }
        }
      }}
    >
      <Field label="Model name" htmlFor={`${id}-name`}>
        <Input
          id={`${id}-name`}
          value={name}
          onChange={(event) => setName(event.target.value)}
        />
      </Field>
      <Field label="Remote model ID" htmlFor={`${id}-remote`}>
        <Combobox
          id={`${id}-remote`}
          label="Remote model ID"
          value={remote}
          onChange={(value) => {
            dirty.current = true;
            setRemote(value);
          }}
          options={options}
          chooseLabel="Choose model"
          loadLabel="Load models"
          onLoad={() => void listModels()}
          loading={listing}
        />
      </Field>
      {budgetSupported ? (
        <Field
          label="Thinking token budget"
          htmlFor={`${id}-budget`}
          hint={
            validBudget
              ? "Used only when budget is selected."
              : "Enter a positive whole number."
          }
        >
          <Input
            id={`${id}-budget`}
            type="number"
            min="1"
            step="1"
            aria-invalid={!validBudget}
            value={budgetTokens}
            onChange={(event) => setBudgetTokens(event.target.value)}
          />
        </Field>
      ) : null}
      <label className="flex items-center gap-2 text-sm">
        <input
          type="checkbox"
          checked={enabled}
          onChange={(event) => setEnabled(event.target.checked)}
        />
        Enabled
      </label>
      <Field label="Test thinking effort" htmlFor={`${id}-thinking`}>
        <select
          id={`${id}-thinking`}
          className={selectClass}
          value={thinking}
          onChange={(event) => setThinking(event.target.value as Thinking)}
        >
          {testThinkingOptions.map((option) => (
            <option key={option} value={option}>
              {thinkingLabel(option, parsedBudget)}
            </option>
          ))}
        </select>
      </Field>
      <div className="flex flex-wrap gap-2">
        <Button type="submit" variant="primary">
          {busy ? <Spinner label="Saving model" /> : null}
          {model ? "Save model" : "Add model"}
        </Button>
        <Button
          disabled={
            testing ||
            !provider.api_key_set ||
            !remote.trim() ||
            !validTestSelection ||
            budgetChanged ||
            Boolean(model && remote !== model.model)
          }
          onClick={() => void testModel()}
        >
          {testing ? <Spinner label="Testing" /> : null}Test
        </Button>
        {model ? (
          <Button onClick={() => void save("delete_model", model.id)}>
            Delete model
          </Button>
        ) : null}
      </div>
    </form>
  );
}
