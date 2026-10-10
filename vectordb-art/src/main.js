/**
 * Application wiring. This is the only module that knows about every part:
 * it builds the dataset, the generator, the heuristic evaluator, the loop
 * controller, the renderer and the panel, and passes data between them.
 */
import { DEFAULT_WEIGHTS, EVENTS, LIMITS, NEIGHBOUR_K, TERMINATION } from './contracts.js';
import { cosine, semanticNeighbours } from './core/knn.js';
import { createDataset } from './data/dataset.js';
import { createGenerator } from './generator/generator.js';
import { CRITERION_DESCRIPTIONS, createDiscriminator } from './discriminator/discriminator.js';
import { createLoopController } from './loop/controller.js';
import { MAX_ZOOM } from './render/camera.js';
import { PALETTE } from './render/particles.js';
import { createRenderer } from './render/renderer.js';
import { shapeOf, shapeSvg } from './render/shapes.js';
import { createHud } from './ui/hud.js';

/**
 * Pause between iterations. Each one is shown as: candidate proposed, then (PROPOSE_MS later) kept or
 * rejected. At 200 ms, 60 seeds checked end for the same reason as with no pause at all; from about
 * 240 ms the longest runs start reaching the 10 s time limit instead.
 */
const STEP_MS = 200;
/** How long a proposed candidate is shown before its verdict. Generating and scoring take under 3 ms. */
const PROPOSE_MS = 90;
const NARROW_QUERY = '(max-width: 860px)';
/** Clear space between the artwork and the chrome around it, in CSS pixels. */
const CHROME_GAP = 24;
/** Height of the row of view controls: along the bottom on wide screens, under the bar on narrow ones. */
const VIEW_ROW = 44;
/** One press of a zoom button. */
const ZOOM_STEP = 1.6;

const canvas = document.getElementById('stage');
const bar = document.getElementById('bar');
const panel = document.getElementById('panel');
const inspector = document.getElementById('inspector');

const dataset = createDataset();
const discriminator = createDiscriminator({ dataset });

/**
 * Keeps the artwork clear of the chrome. Wide screens: below the bar, between the rail and the
 * inspector (when open). Narrow screens: between the bar and the bottom sheet.
 */
function getInsets() {
  const top = bar.getBoundingClientRect().bottom;
  const rail = panel.getBoundingClientRect();
  const inspectorBox = inspector.hidden ? null : inspector.getBoundingClientRect();
  if (window.matchMedia(NARROW_QUERY).matches) {
    const sheetTop = inspectorBox ? inspectorBox.top : rail.top;
    return { top: top + VIEW_ROW, right: 0, bottom: window.innerHeight - sheetTop + CHROME_GAP / 2, left: 0 };
  }
  return {
    top,
    right: inspectorBox ? window.innerWidth - inspectorBox.left + CHROME_GAP : 0,
    bottom: VIEW_ROW,
    left: rail.right + CHROME_GAP,
  };
}

/** Cosine similarity of two stored vectors: the one measure used for neighbours and for the scores shown. */
const similarity = (i, j) => cosine(dataset.points[i].vector, dataset.points[j].vector);

const renderer = createRenderer({
  canvas,
  dataset,
  // Neighbour lists come from the vectors themselves, closest first; the renderer only draws them.
  neighbours: semanticNeighbours(dataset, NEIGHBOUR_K),
  similarity,
  getInsets,
  onSelect: (index, nearest) =>
    hud.setSelection(
      index === null ? null : dataset.points[index],
      nearest.map((j) => ({ point: dataset.points[j], similarity: similarity(index, j) })),
    ),
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
  clusterMarks: PALETTE.map(([r, g, b], cluster) => shapeSvg(shapeOf(cluster), `rgb(${r} ${g} ${b})`, 11)),
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

/** The iteration whose candidate is on screen and whose verdict has not been shown yet. */
let pending = null;
let verdictTimer = 0;

/** Shows the verdict of the pending iteration: the layout moves only if the candidate was accepted. */
function showVerdict() {
  if (!pending) return;
  clearTimeout(verdictTimer);
  const { state, record } = pending;
  pending = null;
  if (record.accepted) renderer.setLayout(state.best.candidate.positions);
  renderer.pulse(record.accepted ? 'accepted' : 'rejected');
  hud.update(state);
  hud.setPhase({ kind: record.accepted ? 'accepted' : 'rejected', record });
}

/** @param {import('./contracts.js').LoopEvent} event */
function onLoopEvent(event) {
  const { type, state, record } = event;
  // Anything still waiting is settled first, so there is never more than one timer or one overlay.
  showVerdict();
  if (type === EVENTS.ITERATION) {
    renderer.showCandidate(state.last.candidate.positions);
    hud.setPhase({ kind: 'evaluating', iteration: record.iteration });
    pending = { state, record };
    verdictTimer = setTimeout(showVerdict, PROPOSE_MS);
    return;
  }
  if (type === EVENTS.TERMINATE && state.terminationReason !== TERMINATION.CANCELLED) {
    renderer.pulse('terminated');
  } else if (type === EVENTS.RESET) {
    renderer.showCandidate(null);
    renderer.setLayout(null);
  }
  if (type === EVENTS.START || type === EVENTS.TERMINATE || type === EVENTS.RESET) hud.setPhase(null);
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

hud.setSelection(null);
build(hud.getSeed() ?? dataset.seed);
renderer.start();
window.matchMedia(NARROW_QUERY).addEventListener('change', () => renderer.resize());
// The chrome grows, shrinks, opens and closes, so the artwork is refitted to the space it leaves.
const chromeObserver = new ResizeObserver(() => {
  // On narrow screens the inspector stacks directly above the bottom sheet.
  document.documentElement.style.setProperty('--rail-h', `${Math.ceil(panel.getBoundingClientRect().height)}px`);
  renderer.resize();
});
for (const element of [bar, panel, inspector]) chromeObserver.observe(element);
