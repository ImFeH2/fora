import { BackendError, backend } from "@/lib/backend";
import workletUrl from "./voice-worklet.js?url";

export type VoiceState = "starting" | "recording" | "finishing" | "closed";
export type VoiceEvent =
  | { type: "state"; state: VoiceState }
  | { type: "level"; level: number }
  | { type: "transcript"; text: string }
  | { type: "error"; message: string };

export class VoiceRecording {
  #state: VoiceState = "starting";
  #socket: WebSocket | null = null;
  #stream: MediaStream | null = null;
  #context: AudioContext | null = null;
  #source: MediaStreamAudioSourceNode | null = null;
  #processor: AudioWorkletNode | null = null;
  #timer: ReturnType<typeof setTimeout> | null = null;
  #off: (() => void) | null = null;

  constructor(private readonly emit: (event: VoiceEvent) => void) {}

  get state(): VoiceState {
    return this.#state;
  }

  #isClosed(): boolean {
    return this.#state === "closed";
  }

  #setState(state: VoiceState): void {
    this.#state = state;
    this.emit({ type: "state", state });
  }

  #deadline(message: string): void {
    if (this.#timer !== null) clearTimeout(this.#timer);
    this.#timer = setTimeout(() => this.#fail(new Error(message)), 60_000);
  }

  #fail(error: unknown): void {
    if (this.#isClosed()) return;
    this.emit({
      type: "error",
      message: error instanceof Error ? error.message : String(error),
    });
    this.cancel();
  }

  async start(): Promise<void> {
    this.emit({ type: "state", state: "starting" });
    this.#deadline("Voice recording could not start in time.");
    this.#off = backend.onEvent((event) => {
      if (event.type === "connection.closed")
        this.#fail(new Error("Connection lost. Recording stopped."));
    });
    try {
      if (!navigator.mediaDevices?.getUserMedia)
        throw new Error("Microphone access requires a secure browser page.");
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      if (this.#isClosed()) {
        for (const track of stream.getTracks()) track.stop();
        return;
      }
      this.#stream = stream;
      for (const track of stream.getAudioTracks())
        track.onended = () => this.#fail(new Error("Microphone disconnected."));
      const context = new AudioContext();
      this.#context = context;
      await context.audioWorklet.addModule(workletUrl);
      if (this.#isClosed()) {
        this.#releaseAudio();
        return;
      }
      const url = await backend.voiceUrl();
      if (this.#isClosed()) {
        this.#releaseAudio();
        return;
      }
      const socket = new WebSocket(url);
      this.#socket = socket;
      socket.onopen = () => {
        if (this.#isClosed()) return;
        socket.send(
          JSON.stringify({
            type: "start",
            sample_rate: context.sampleRate,
            channels: 1,
            format: "f32le",
          }),
        );
      };
      socket.onerror = () => this.#fail(new Error("Voice connection failed."));
      socket.onclose = () => this.#fail(new Error("Voice connection closed."));
      socket.onmessage = (event) => {
        if (this.#isClosed()) return;
        try {
          const message = JSON.parse(event.data);
          if (message.type === "ready") {
            if (this.#state !== "starting")
              throw new Error("Unexpected voice ready event.");
            void this.#capture(context, stream).catch((error) =>
              this.#fail(error),
            );
          } else if (
            message.type === "transcript" ||
            message.type === "finished"
          ) {
            if (typeof message.text !== "string")
              throw new Error("Invalid voice transcript.");
            this.emit({ type: "transcript", text: message.text });
            if (message.type === "finished") this.cancel();
          } else if (message.type === "error") {
            throw new BackendError(message.code, message.message);
          } else if (message.type !== "finishing") {
            throw new Error("Unexpected voice event.");
          }
        } catch (error) {
          this.#fail(error);
        }
      };
    } catch (error) {
      this.#fail(error);
    }
  }

  async #capture(context: AudioContext, stream: MediaStream): Promise<void> {
    const processor = new AudioWorkletNode(context, "huddol-voice", {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });
    this.#processor = processor;
    processor.onprocessorerror = () =>
      this.#fail(new Error("Audio capture failed."));
    processor.port.onmessage = (event) => {
      if (this.#isClosed()) return;
      try {
        const socket = this.#socket;
        if (!socket || socket.readyState !== WebSocket.OPEN)
          throw new Error("Voice connection is unavailable.");
        if (event.data.type === "stopped") {
          socket.send(JSON.stringify({ type: "stop" }));
          this.#releaseAudio();
          return;
        }
        const samples: Float32Array = event.data.samples;
        if (
          socket.bufferedAmount + samples.byteLength >
          context.sampleRate * 4 * 30
        )
          throw new Error(
            "Audio connection cannot keep up. Recording stopped.",
          );
        socket.send(samples);
        this.emit({ type: "level", level: event.data.level });
      } catch (error) {
        this.#fail(error);
      }
    };
    this.#source = context.createMediaStreamSource(stream);
    this.#source.connect(processor);
    processor.connect(context.destination);
    await context.resume();
    if (this.#isClosed()) return;
    if (this.#timer !== null) clearTimeout(this.#timer);
    this.#timer = null;
    this.#setState("recording");
  }

  stop(): void {
    if (this.#state === "starting") {
      this.cancel();
      return;
    }
    if (this.#state !== "recording") return;
    const processor = this.#processor;
    if (!processor) throw new Error("Audio capture processor is unavailable");
    this.#setState("finishing");
    this.#deadline("Voice transcription did not finish in time.");
    processor.port.postMessage("stop");
  }

  #releaseAudio(): void {
    for (const track of this.#stream?.getTracks() ?? []) {
      track.onended = null;
      track.stop();
    }
    this.#stream = null;
    this.#source?.disconnect();
    this.#source = null;
    this.#processor?.disconnect();
    this.#processor?.port.close();
    this.#processor = null;
    const context = this.#context;
    this.#context = null;
    if (context && context.state !== "closed")
      void context.close().catch(backend.reportFailure);
    this.emit({ type: "level", level: 0 });
  }

  cancel(): void {
    if (this.#isClosed()) return;
    this.#setState("closed");
    if (this.#timer !== null) clearTimeout(this.#timer);
    this.#timer = null;
    this.#off?.();
    this.#off = null;
    this.#releaseAudio();
    this.#socket?.close();
    this.#socket = null;
  }
}
