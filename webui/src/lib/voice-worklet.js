class VoiceProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.active = true;
    this.samples = new Float32Array(2048);
    this.count = 0;
    this.energy = 0;
    this.port.onmessage = (event) => {
      if (event.data === "stop") {
        this.active = false;
        this.flush();
        this.port.postMessage({ type: "stopped" });
      }
    };
  }

  process(inputs) {
    if (!this.active) return false;
    const channels = inputs[0];
    if (!channels?.length) return true;
    for (let index = 0; index < channels[0].length; index += 1) {
      let sample = 0;
      for (const channel of channels)
        sample += channel[index] / channels.length;
      this.samples[this.count] = sample;
      this.energy += sample * sample;
      this.count += 1;
      if (this.count === this.samples.length) this.flush();
    }
    return true;
  }

  flush() {
    if (this.count === 0) return;
    const samples = this.samples.slice(0, this.count);
    this.port.postMessage(
      { type: "audio", samples, level: Math.sqrt(this.energy / this.count) },
      [samples.buffer],
    );
    this.count = 0;
    this.energy = 0;
  }
}

registerProcessor("huddol-voice", VoiceProcessor);
