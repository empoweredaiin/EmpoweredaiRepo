/**
 * Application wiring. This is the only module that knows about every part:
 * it builds the dataset, the generator, the heuristic evaluator, the loop
 * controller, the renderer and the panel, and passes data between them.
 */
import { DEFAULT_WEIGHTS, EVENTS, LIMITS, NEIGHBOUR_K, TERMINATION } from './contracts.js';
import { semanticNeighbours } from './core/knn.js';
import { createDataset } from './data/dataset.js';
import { createGenerator } from './generator/generator.js';
import { CRITERION_DESCRIPTIONS, createDiscriminator } from './discriminator/discriminator.js';
import { createLoopController } from './loop/controller.js';
import { MAX_ZOOM } from './render/camera.js';
import { PALETTE } from './render/particles.js';
import { createRenderer } from './render/renderer.js';
import { createHud } from './ui/hud.js';

/** Pause between iterations: long enough to watch each step, short enough for 100 iterations inside the 10 s limit. */
const STEP_MS = 90;
const NARROW_QUERY = '(max-width: 860px)';
const PANEL_GAP = 16;
/** One press of a zoom button. */
const ZOOM_STEP = 1.6;
/** Space kept free at the top of narrow screens, where the zoom control sits. */
const ZOOM_CONTROL_HEIGHT = 48;

const canvas = document.getElementById('stage');
const panel = document.getElementById('panel');

const dataset = createDataset();
const discriminator = createDiscriminator({ dataset });

/** Keeps the layout clear of the panel: beside it on wide screens, above it on narrow ones. */
function getInsets() {
  const box = panel.getBoundingClientRect();
  if (window.matchMedia(NARROW_QUERY).matches) {
    return { top: ZOOM_CONTROL_HEIGHT, right: 0, bottom: window.innerHeight - box.top + PANEL_GAP / 2, left: 0 };
  }
  return { top: 0, right: 0, bottom: 0, left: box.right + PANEL_GAP };
}

const renderer = createRenderer({
  canvas,
  dataset,
  neighbours: semanticNeighbours(dataset, NEIGHBOUR_K),
  getInsets,
  onHover: (index) => hud.setHover(index === null ? null : dataset.points[index]),
  onViewChange: (zoom) => hud.setZoom(zoom),
});

/** @type {import('./contracts.js').LoopController} */
let controller;
let seed = -1;
let unsubscribe = () => {};

const hud = createHud({
  document,
  dataset,
  limits: LIMITS,
  weights: DEFAULT_WEIGHTS,
  criterionDescriptions: CRITERION_DESCRIPTIONS,
  clusterColors: PALETTE.map(([r, g, b]) => `rgb(${r} ${g} ${b})`),
  maxZoom: MAX_ZOOM,
  handlers: {
    start() {
      const wanted = hud.getSeed();
      if (wanted === null) {
        hud.setError('Seed must be a whole number of 0 or more.');
        return;
      }
      hud.setError(null);
      if (wanted !== seed) build(wanted);
      renderer.setLayout(null);
      controller.start().catch(reportFailure);
    },
    pause: () => controller.pause(),
    resume: () => controller.resume(),
    stop: () => controller.stop(),
    reset() {
      hud.setError(null);
      controller.reset();
    },
    zoomIn: () => renderer.zoomBy(ZOOM_STEP),
    zoomOut: () => renderer.zoomBy(1 / ZOOM_STEP),
    zoomReset: () => renderer.resetView(),
  },
});

function reportFailure(error) {
  console.error(error);
  hud.setError(`The run failed: ${error instanceof Error ? error.message : String(error)}`);
}

/** @param {import('./contracts.js').LoopEvent} event */
function onLoopEvent(event) {
  const { type, state, record } = event;
  if (type === EVENTS.ITERATION) {
    if (record.accepted) renderer.setLayout(state.best.candidate.positions);
    renderer.pulse(record.accepted ? 'accepted' : 'rejected');
  } else if (type === EVENTS.TERMINATE && state.terminationReason !== TERMINATION.CANCELLED) {
    renderer.pulse('terminated');
  } else if (type === EVENTS.RESET) {
    renderer.setLayout(null);
  }
  hud.update(state);
}

/** Replaces the generator and controller with ones for a new seed. */
function build(nextSeed) {
  controller?.reset();
  unsubscribe();
  seed = nextSeed;
  controller = createLoopController({
    generator: createGenerator({ dataset, seed }),
    discriminator,
    limits: LIMITS,
    yieldToHost: () => new Promise((resolve) => setTimeout(resolve, STEP_MS)),
  });
  unsubscribe = controller.subscribe(onLoopEvent);
  hud.update(controller.getState());
}

build(hud.getSeed() ?? dataset.seed);
renderer.start();
window.matchMedia(NARROW_QUERY).addEventListener('change', () => renderer.resize());
// The panel grows and shrinks with its content, so the layout is refitted to the space beside it.
new ResizeObserver(() => renderer.resize()).observe(panel);
