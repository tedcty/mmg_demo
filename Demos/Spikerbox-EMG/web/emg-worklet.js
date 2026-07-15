// EMG RMS-envelope processor for the browser game (web/index.html).
//
// Loaded as a real served module (audioWorklet.addModule('emg-worklet.js'))
// rather than from a blob: URL — iOS/WebKit is unreliable loading worklets
// from blob URLs, so a served file is what makes the game work on iPad.
//
// It maintains a moving sum-of-squares over an RMS window (default 0.02 s,
// overridable via processorOptions.rmsWin) and posts the instantaneous RMS
// envelope back to the main thread on every processing block.
class EmgProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        const opts = (options && options.processorOptions) || {};
        const rmsWin = opts.rmsWin || 0.02;
        this.win = Math.max(1, Math.floor(rmsWin * sampleRate));
        this.sq = new Float32Array(this.win);
        this.idx = 0;
        this.sum = 0;
    }
    process(inputs) {
        const input = inputs[0];
        if (input && input[0]) {
            const ch = input[0];
            for (let i = 0; i < ch.length; i++) {
                const s2 = ch[i] * ch[i];
                this.sum += s2 - this.sq[this.idx];
                this.sq[this.idx] = s2;
                this.idx = (this.idx + 1) % this.win;
            }
            const rms = Math.sqrt(Math.max(0, this.sum / this.win));
            this.port.postMessage(rms);
        }
        return true;
    }
}
registerProcessor('emg-processor', EmgProcessor);
