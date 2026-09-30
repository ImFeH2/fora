const hosts: HTMLElement[] = [];
const listeners = new Set<() => void>();

function publish() {
  for (const listener of listeners) listener();
}

export function registerToastHost(host: HTMLElement): () => void {
  hosts.push(host);
  publish();
  return () => {
    const index = hosts.indexOf(host);
    if (index >= 0) hosts.splice(index, 1);
    publish();
  };
}

export function readToastHost(): HTMLElement | null {
  return hosts[hosts.length - 1] ?? null;
}

export function subscribeToastHost(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
