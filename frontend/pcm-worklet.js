// AudioWorklet: runs on the browser's audio thread.
// Takes mic audio at the context's native rate (usually 48 kHz), downsamples
// to 16 kHz mono, converts to int16 PCM, and posts ~100 ms chunks to the
// main thread, which forwards them over the WebSocket.

class PcmDownsampler extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.targetRate = options.processorOptions?.targetRate ?? 16000;
    this.ratio = sampleRate / this.targetRate; // `sampleRate` is a worklet global
    this.chunkSamples = Math.round(this.targetRate * 0.1); // 100 ms
    this.out = new Int16Array(this.chunkSamples);
    this.outIndex = 0;
    this.carry = new Float32Array(0); // input samples not yet consumed
    this.frac = 0;
  }

  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input) return true;

    // Join leftover samples from the previous call with the new block.
    const buf = new Float32Array(this.carry.length + input.length);
    buf.set(this.carry);
    buf.set(input, this.carry.length);

    // Box-filter decimation: each output sample is the mean of the `ratio`
    // input samples it covers. The averaging acts as a cheap low-pass filter
    // that reduces aliasing compared with simply dropping samples.
    let pos = this.frac;
    while (pos + this.ratio <= buf.length) {
      const start = Math.floor(pos);
      const end = Math.floor(pos + this.ratio);
      let sum = 0;
      for (let i = start; i < end; i++) sum += buf[i];
      const sample = Math.max(-1, Math.min(1, sum / (end - start)));
      this.out[this.outIndex++] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
      if (this.outIndex === this.chunkSamples) {
        const chunk = this.out.buffer.slice(0);
        this.port.postMessage(chunk, [chunk]);
        this.outIndex = 0;
      }
      pos += this.ratio;
    }
    // Keep unconsumed samples and the fractional read position, so
    // non-integer ratios (e.g. 44.1 kHz -> 16 kHz) don't drift.
    this.carry = buf.slice(Math.floor(pos));
    this.frac = pos - Math.floor(pos);
    return true;
  }
}

registerProcessor("pcm-downsampler", PcmDownsampler);
