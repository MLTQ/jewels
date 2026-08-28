import "../styles.css";
import { loadJewelField } from "./field-loader.js";
import { SliceRenderer } from "./slice-renderer.js";
import { VolumeScene } from "./volume-scene.js";

const DATA_URL = "/data/singer-field.json";

const canvas = document.getElementById("volume-canvas");
const loadingState = document.getElementById("loading-state");
const errorState = document.getElementById("error-state");
const playButton = document.getElementById("play-button");
const playIcon = playButton.querySelector(".play-icon");
const playLabel = playButton.querySelector(".play-label");
const frameSlider = document.getElementById("frame-slider");
const frameReadout = document.getElementById("frame-readout");

let playing = false;
let frameIndex = 0;
let lastFrameAt = 0;
let field;
let sliceRenderer;
let volume;

function normalizedTime(index) {
  const frames = field.field.shape[0];
  const fraction = frames <= 1 ? 0 : index / (frames - 1);
  return -field.field.t_scale + 2 * field.field.t_scale * fraction;
}

function updateTransport() {
  const frameCount = field.field.shape[0];
  frameSlider.value = String(frameIndex);
  frameReadout.textContent = `frame ${frameIndex + 1} / ${frameCount}`;
  playButton.setAttribute("aria-pressed", String(playing));
  playIcon.textContent = playing ? "Ⅱ" : "▶";
  playLabel.textContent = playing ? "Pause" : "Play";
}

function setFrame(nextFrame) {
  frameIndex = Math.max(0, Math.min(field.field.shape[0] - 1, nextFrame));
  const time = normalizedTime(frameIndex);
  sliceRenderer.render(volume.renderer, time);
  volume.setTime(time);
  updateTransport();
}

function setPlaying(nextPlaying) {
  playing = nextPlaying;
  lastFrameAt = performance.now();
  updateTransport();
}

function animate(now) {
  if (playing) {
    const frameDuration = 1000 / 12;
    if (now - lastFrameAt >= frameDuration) {
      const steps = Math.max(1, Math.floor((now - lastFrameAt) / frameDuration));
      setFrame((frameIndex + steps) % field.field.shape[0]);
      lastFrameAt += steps * frameDuration;
    }
  }
  volume.render();
  requestAnimationFrame(animate);
}

async function start() {
  try {
    field = await loadJewelField(DATA_URL);
    const videoAspect = field.field.shape[2] / field.field.shape[1];
    sliceRenderer = new SliceRenderer(field, videoAspect);
    volume = new VolumeScene(canvas, field, sliceRenderer.target.texture);
    frameSlider.max = String(field.field.shape[0] - 1);
    frameSlider.disabled = false;
    playButton.disabled = false;

    const resizeObserver = new ResizeObserver(([entry]) => {
      volume.resize(entry.contentRect.width, entry.contentRect.height);
    });
    resizeObserver.observe(canvas.parentElement);

    playButton.addEventListener("click", () => setPlaying(!playing));
    frameSlider.addEventListener("input", () => {
      setPlaying(false);
      setFrame(Number(frameSlider.value));
    });
    window.addEventListener("keydown", (event) => {
      if (event.code === "Space" && event.target === document.body) {
        event.preventDefault();
        setPlaying(!playing);
      }
    });

    setFrame(0);
    loadingState.hidden = true;
    requestAnimationFrame(animate);
  } catch (error) {
    loadingState.hidden = true;
    errorState.hidden = false;
    errorState.textContent = error instanceof Error ? error.message : String(error);
  }
}

start();
