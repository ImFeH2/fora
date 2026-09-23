import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";
import { describe, expect, it } from "vitest";

function processor() {
  const messages: { type: string; samples?: Float32Array; level?: number }[] =
    [];
  class Processor {
    port = {
      onmessage: null as ((event: { data: string }) => void) | null,
      postMessage: (message: (typeof messages)[number]) =>
        messages.push(message),
    };
    process(_inputs: Float32Array[][]): boolean {
      return true;
    }
  }
  let registered!: typeof Processor;
  runInNewContext(
    readFileSync(new URL("./voice-worklet.js", import.meta.url), "utf8"),
    {
      AudioWorkletProcessor: Processor,
      Float32Array,
      registerProcessor: (_name: string, value: typeof Processor) => {
        registered = value;
      },
    },
  );
  return { instance: new registered(), messages };
}

describe("voice AudioWorklet", () => {
  it("mixes channels and flushes partial PCM before acknowledging stop", () => {
    const { instance, messages } = processor();
    instance.process([[new Float32Array([1, -1]), new Float32Array([0, 0])]]);
    expect(messages).toHaveLength(0);
    instance.port.onmessage?.({ data: "stop" });
    expect(messages).toEqual([
      { type: "audio", samples: new Float32Array([0.5, -0.5]), level: 0.5 },
      { type: "stopped" },
    ]);
    expect(instance.process([[new Float32Array([1])]])).toBe(false);
    expect(messages).toHaveLength(2);
  });

  it("bounds batches and reports the RMS of actual samples", () => {
    const { instance, messages } = processor();
    const input = new Float32Array(4097).fill(0.25);
    instance.process([[input]]);
    expect(messages.map((message) => message.samples?.length)).toEqual([
      2048, 2048,
    ]);
    instance.port.onmessage?.({ data: "stop" });
    expect(messages.map((message) => message.samples?.length)).toEqual([
      2048,
      2048,
      1,
      undefined,
    ]);
    expect(
      messages.slice(0, 3).every((message) => message.level === 0.25),
    ).toBe(true);
  });
});
