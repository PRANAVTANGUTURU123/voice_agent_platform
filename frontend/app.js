// Real-Time Voice Agent — browser client (vanilla JS).
//
// Mic  -> AudioWorklet (downsample to 16 kHz int16) -> WebSocket (binary)
// WebSocket JSON events -> transcript / streamed reply / status
// WebSocket binary (4-byte turn id + MP3 chunk) -> streaming playback
//
// Barge-in: the server's VAD sends "speech_start" when the user talks; we stop
// all queued audio immediately and ignore late audio from the cancelled turn.

const sessionId = crypto.randomUUID().replace(/-/g, "").slice(0, 16);
const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ UI helpers
function setStatus(state, text) {
  $("status").dataset.state = state;
  $("status").textContent = text;
}

function addMsg(kind, text = "") {
  const el = document.createElement("div");
  el.className = `msg ${kind}`;
  el.textContent = text;
  $("log").appendChild(el);
  $("log").scrollTop = $("log").scrollHeight;
  return el;
}

let agentEl = null;   // bubble currently receiving streamed tokens
let agentText = null; // text node inside it

function startAgentBubble() {
  agentEl = addMsg("agent");
  const tools = document.createElement("div");
  tools.className = "tools";
  agentEl.appendChild(tools);
  agentText = document.createTextNode("");
  agentEl.appendChild(agentText);
}

// ------------------------------------------------------------------ WebSocket
let ws = null;
let currentTurn = 0;   // audio frames from other turns are dropped
let micOn = false;

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/${sessionId}`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { if (!micOn) setStatus("idle", "Connected — mic off"); };
  ws.onclose = () => {
    setStatus("idle", "Disconnected — reconnecting…");
    setTimeout(connect, 1500);
  };
  ws.onmessage = (e) => (typeof e.data === "string" ? onEvent(JSON.parse(e.data)) : onAudio(e.data));
}

function onEvent(ev) {
  switch (ev.type) {
    case "speech_start":
      player.stop();               // barge-in: silence the agent right away
      setStatus("speaking-user", "Listening to you…");
      break;
    case "turn_start":
      currentTurn = ev.turn;
      player.stop();
      agentEl = null;
      setStatus("thinking", "Thinking…");
      break;
    case "transcript":
      if (ev.text) addMsg("user", ev.text);
      break;
    case "tool":
      if (!agentEl) startAgentBubble();
      agentEl.querySelector(".tools").textContent += `🔧 ${ev.name}  `;
      break;
    case "agent_token":
      if (!agentEl) startAgentBubble();
      agentText.textContent += ev.text;
      $("log").scrollTop = $("log").scrollHeight;
      break;
    case "audio_sentence_end":
      player.sentenceEnd();
      break;
    case "agent_done": {
      player.finish();
      const m = ev.metrics;
      const parts = [
        m.stt_ms != null && `STT ${m.stt_ms} ms`,
        m.first_token_ms != null && `first token ${m.first_token_ms} ms`,
        m.first_audio_ms != null && `first audio ${m.first_audio_ms} ms`,
      ].filter(Boolean);
      if (agentEl && parts.length) {
        const d = document.createElement("div");
        d.className = "metrics";
        d.textContent = parts.join(" · ");
        agentEl.appendChild(d);
      }
      if (!player.playing && !player.ms) idleStatus();
      break;
    }
    case "info":
      addMsg("sys", ev.message);
      if (!player.playing) idleStatus();
      break;
    case "error":
      addMsg("sys", ev.message);
      player.finish();
      if (!player.playing) idleStatus();
      break;
  }
}

function onAudio(buf) {
  const turn = new DataView(buf).getUint32(0); // big-endian turn id header
  if (turn !== currentTurn) return;            // stale audio from a cancelled turn
  player.push(buf.slice(4));
}

// ------------------------------------------------------------------ Playback
// The server streams MP3 chunks as edge-tts produces them. Two strategies,
// same interface: push(chunk) / sentenceEnd() / finish() / stop() / playing.
const idleStatus = () => setStatus("idle", micOn ? "Listening…" : "Mic off");

// 1) Streaming (Chrome, Edge, Firefox): append every chunk to one MediaSource
//    per turn. Playback starts on the FIRST chunk, and sentences play back to
//    back because the SourceBuffer is in "sequence" mode.
const streamingPlayer = {
  audio: null, ms: null, sb: null, queue: [], finished: false, playing: false,

  start() {
    const audio = new Audio();
    const ms = new MediaSource();
    Object.assign(this, { audio, ms, sb: null, queue: [], finished: false });
    audio.src = URL.createObjectURL(ms);
    ms.addEventListener("sourceopen", () => {
      if (this.ms !== ms) return; // stopped before it opened
      URL.revokeObjectURL(audio.src);
      this.sb = ms.addSourceBuffer("audio/mpeg");
      this.sb.mode = "sequence";
      this.sb.addEventListener("updateend", () => this.pump());
      this.pump();
    }, { once: true });
    audio.onplaying = () => { this.playing = true; setStatus("speaking-agent", "Speaking…"); };
    audio.onended = () => { this.playing = false; idleStatus(); };
    audio.play().catch(() => {}); // starts as soon as the first chunk is buffered
  },

  push(chunk) {
    if (!this.ms) this.start();
    this.queue.push(chunk);
    this.pump();
  },

  // Feed queued chunks into the SourceBuffer one append at a time.
  pump() {
    const { sb, ms } = this;
    if (!sb || sb.updating) return;
    if (this.queue.length) {
      try { sb.appendBuffer(this.queue.shift()); } catch (e) { console.warn("appendBuffer", e); }
    } else if (this.finished && ms.readyState === "open") {
      ms.endOfStream(); // lets the <audio> element fire "ended"
    }
  },

  sentenceEnd() {},
  finish() { if (this.ms) { this.finished = true; this.pump(); } },

  stop() {
    if (this.audio) {
      this.audio.pause();
      this.audio.removeAttribute("src");
      this.audio.load();
    }
    Object.assign(this, { audio: null, ms: null, sb: null, queue: [], finished: false, playing: false });
  },
  unlock() {},
};

// 2) Fallback (e.g. iOS Safari, no MP3 MediaSource): collect each sentence's
//    chunks, decode the complete MP3, and schedule sentences gaplessly on the
//    AudioContext clock.
const sentencePlayer = {
  ctx: null, nextTime: 0, sources: new Set(), parts: [],
  chain: Promise.resolve(), // keeps async decodes in sentence order
  generation: 0,            // bumped by stop(); pending decodes become no-ops

  get playing() { return this.sources.size > 0; },
  push(chunk) { this.parts.push(chunk); },

  sentenceEnd() {
    if (!this.parts.length) return;
    const blob = new Blob(this.parts);
    this.parts = [];
    const gen = this.generation;
    this.chain = this.chain.then(async () => {
      let audio;
      try { audio = await this.ctx.decodeAudioData(await blob.arrayBuffer()); } catch { return; }
      if (gen !== this.generation) return;
      const src = this.ctx.createBufferSource();
      src.buffer = audio;
      src.connect(this.ctx.destination);
      const start = Math.max(this.ctx.currentTime + 0.02, this.nextTime);
      src.start(start);
      this.nextTime = start + audio.duration;
      this.sources.add(src);
      setStatus("speaking-agent", "Speaking…");
      src.onended = () => { this.sources.delete(src); if (!this.playing) idleStatus(); };
    });
  },

  finish() { this.sentenceEnd(); },

  stop() {
    this.generation++;
    for (const s of this.sources) { try { s.stop(); } catch {} }
    this.sources.clear();
    this.parts = [];
    this.nextTime = 0;
  },

  // AudioContext must be created/resumed inside a user gesture.
  unlock() { this.ctx ??= new AudioContext(); this.ctx.resume(); },
};

const player = window.MediaSource?.isTypeSupported?.("audio/mpeg") ? streamingPlayer : sentencePlayer;

// ------------------------------------------------------------------ Microphone
let micCtx = null, micStream = null, workletNode = null;

async function startMic() {
  // Echo cancellation trade-off (measured in a browser test):
  //  - Speakers: AEC is required, or the agent's own voice triggers barge-in.
  //    But while the agent talks, AEC also mutes the first ~1 s of the user's
  //    speech, so interruptions can lose their first words.
  //  - Headphones: no echo exists, so AEC off gives clean, fast barge-in.
  const headphones = $("headphones").checked;
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: !headphones, noiseSuppression: true, autoGainControl: true },
    });
  } catch (err) {
    addMsg("sys", `Microphone unavailable (${err.name}). You can still type questions below.`);
    return;
  }
  micCtx = new AudioContext();
  await micCtx.audioWorklet.addModule("/static/pcm-worklet.js");
  const source = micCtx.createMediaStreamSource(micStream);
  workletNode = new AudioWorkletNode(micCtx, "pcm-downsampler", { processorOptions: { targetRate: 16000 } });
  workletNode.port.onmessage = (e) => { if (ws?.readyState === WebSocket.OPEN) ws.send(e.data); };
  source.connect(workletNode); // not connected to destination: we don't want to hear ourselves
  player.unlock(); // enable audio output inside this user gesture
  micOn = true;
  $("mic").classList.add("on");
  setStatus("idle", "Listening…");
}

function stopMic() {
  workletNode?.disconnect();
  micStream?.getTracks().forEach((t) => t.stop());
  micCtx?.close();
  micCtx = micStream = workletNode = null;
  micOn = false;
  $("mic").classList.remove("on");
  setStatus("idle", "Mic off");
}

$("mic").onclick = () => (micOn ? stopMic() : startMic());
// Re-open the mic with the new echo-cancellation setting if it's already on.
$("headphones").onchange = async () => { if (micOn) { stopMic(); await startMic(); } };

// ------------------------------------------------------------------ Typed input
$("askForm").onsubmit = (e) => {
  e.preventDefault();
  const text = $("askInput").value.trim();
  if (!text || ws?.readyState !== WebSocket.OPEN) return;
  player.unlock();
  ws.send(JSON.stringify({ type: "text", text }));
  $("askInput").value = "";
};

// ------------------------------------------------------------------ PDF upload
$("uploadBtn").onclick = async () => {
  const files = $("pdfInput").files;
  if (!files.length) { $("uploadStatus").textContent = "Choose one or more PDFs first."; return; }
  const form = new FormData();
  for (const f of files) form.append("files", f);
  $("uploadBtn").disabled = true;
  $("uploadStatus").textContent = "Indexing…";
  try {
    const res = await fetch(`/upload/${sessionId}`, { method: "POST", body: form });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || res.statusText);
    const added = body.added.map((a) => `${a.file} (${a.chunks} chunks)`).join(", ");
    $("uploadStatus").textContent = `Indexed ${added}` + (body.errors.length ? ` · Skipped: ${body.errors.join("; ")}` : "");
    $("fileList").textContent = `PDFs in this session: ${body.files.join(", ")}`;
    $("pdfInput").value = "";
  } catch (err) {
    $("uploadStatus").textContent = `Upload failed: ${err.message}`;
  } finally {
    $("uploadBtn").disabled = false;
  }
};

connect();
