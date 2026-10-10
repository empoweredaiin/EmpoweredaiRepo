/**
 * The generator: proposes 2D layouts of the dataset's embeddings.
 *
 * Every layout is derived from the vectors themselves. The first proposal is a
 * seeded random projection of the embeddings; every later proposal refines the
 * best layout so far with a few force-directed steps whose forces come from the
 * cosine-similarity structure (semantic k-nearest neighbours and the pairwise
 * similarity matrix). Cluster labels are never read.
 *
 * `generate` keeps no state between calls: its random stream is rebuilt from
 * hashSeed(seed, iteration) on every call.
 */
import { LAYOUT_MAX, LAYOUT_MIN, NEIGHBOUR_K, POINT_RADIUS, clamp01 } from '../contracts.js';
import { cosine, semanticNeighbours } from '../core/knn.js';
import { createRng, gaussian, hashSeed } from '../core/rng.js';
import { collide, forceStep } from './forces.js';

/**
 * Tunable options of the generator. Pass overrides as `options` to
 * {@link createGenerator}; anything omitted falls back to these values.
 * Distances are in layout units (the unit square).
 *
 * @typedef {Object} GeneratorOptions
 * @property {number} margin          Gap kept between every point and the edge of the unit square.
 * @property {number} neighbours      Size k of the semantic neighbourhood each point is attracted to.
 * @property {number} steps           Force-directed steps performed per `generate` call.
 * @property {number} stepMax         Largest move of one point in one step, used on the first refinement.
 * @property {number} stepMin         Smallest such move; the step size anneals towards it.
 * @property {number} stepHalfLife    Iterations after which the annealed part of the step size has halved.
 * @property {number} attract         Weight of the pull towards semantic nearest neighbours.
 * @property {number} cohesion        Weight of the pull towards every similar point (similarity-weighted).
 * @property {number} repel           Weight of the push away from dissimilar points.
 * @property {number} attractRange    Distance over which a pull grows to its full strength.
 * @property {number} cohesionRest    Distance below which similar points no longer pull on each other,
 *                                    so a group of similar points is not squeezed into overlap.
 * @property {number} repelRange      Distance scale over which the push from dissimilar points fades.
 * @property {number} jitter          Standard deviation of the random nudge, as a fraction of the step size.
 * @property {number} collisionPad    Extra clearance, as a fraction of the point diameter, enforced
 *                                    between any two points.
 * @property {number} collisionPasses Overlap-resolution passes after each step.
 * @property {number} feedbackGain    How strongly the weakest criterion of the feedback is favoured
 *                                    (0 ignores feedback).
 * @property {number} shakeChance     Probability that a refinement uses the 'shake' strategy.
 * @property {number} relaxChance     Probability that a refinement uses the 'relax' strategy.
 */

/** @type {Readonly<GeneratorOptions>} */
export const GENERATOR_DEFAULTS = Object.freeze({
  margin: 0.04,
  neighbours: NEIGHBOUR_K,
  steps: 2,
  stepMax: 0.012,
  stepMin: 0.0025,
  stepHalfLife: 80,
  attract: 5,
  cohesion: 3,
  repel: 20,
  attractRange: 0.12,
  cohesionRest: 0.15,
  repelRange: 0.2,
  jitter: 0.4,
  collisionPad: 0.1,
  collisionPasses: 3,
  feedbackGain: 1,
  shakeChance: 0.1,
  relaxChance: 0.2,
});

/**
 * How each refinement strategy scales the base forces.
 *   refine  local neighbourhood refinement, the default move
 *   relax   global relaxation: similar points gather, dissimilar ones spread out
 *   shake   a larger, noisier move that can escape a local arrangement
 */
const STRATEGIES = Object.freeze({
  refine: Object.freeze({ step: 1, attract: 1, cohesion: 0.5, repel: 0.6, jitter: 1 }),
  relax: Object.freeze({ step: 1.25, attract: 0.4, cohesion: 1.2, repel: 1.2, jitter: 0.5 }),
  shake: Object.freeze({ step: 1.6, attract: 1, cohesion: 0.5, repel: 0.6, jitter: 4 }),
});

/**
 * Width of the similarity band over which a pair turns from "gathers" into
 * "spreads", as a fraction of the gap between a typical nearest neighbour's
 * similarity and the average pair's similarity.
 */
const AFFINITY_WIDTH = 0.3;

/** Name recorded in meta.strategy for a layout projected straight from the vectors. */
const PROJECT = 'project';

/**
 * Creates a generator bound to one dataset and one seed.
 *
 * Construction precomputes, once, everything that depends only on the vectors:
 * the semantic k-nearest neighbours of every point and two N x N weight
 * matrices derived from pairwise cosine similarity (how strongly two points
 * gather, how strongly they push apart). The dataset is only read.
 *
 * @param {Object} config
 * @param {import('../contracts.js').Dataset} config.dataset  Source embeddings; never mutated.
 * @param {number} config.seed                                Integer. Different seeds give different,
 *                                                            individually reproducible, layouts.
 * @param {Partial<GeneratorOptions>} [config.options]        Overrides for {@link GENERATOR_DEFAULTS}.
 * @returns {import('../contracts.js').Generator}
 * @throws {TypeError}  When the dataset is missing or malformed, the seed is not an integer or an
 *                      option is not a finite number.
 * @throws {RangeError} When an option is outside its meaningful range.
 */
export function createGenerator({ dataset, seed, options = {} } = {}) {
  assertDataset(dataset);
  if (!Number.isSafeInteger(seed)) throw new TypeError(`createGenerator: seed must be an integer, got ${seed}`);
  const settings = resolveOptions(options);
  const count = dataset.points.length;
  const model = buildModel(dataset, settings.neighbours);

  /**
   * Produces one candidate layout.
   *
   * Pure: the result depends only on the dataset, the seed, the options and
   * `input`. Calling it again with the same input, on this or on any other
   * generator created with the same arguments, in any order, returns identical
   * positions. Nothing passed in is mutated; `positions` is always a new array.
   *
   * - `parent === null`: the vectors are projected onto a seeded random plane
   *   and fitted into the unit square (strategy 'project').
   * - otherwise: a copy of `parent.positions` is refined by `steps`
   *   force-directed steps (strategy 'refine', or occasionally 'relax' or
   *   'shake', chosen by the seeded stream of this iteration). The step size
   *   anneals with `iteration`, and `feedback.criteria` shifts weight towards
   *   the forces that serve the weakest criterion.
   *
   * @param {import('../contracts.js').GenerateInput} input
   * @returns {import('../contracts.js').CandidateLayout}
   * @throws {TypeError}  When `input`, `iteration` or `parent` has the wrong type or shape.
   * @throws {RangeError} When `iteration` is below 1.
   */
  function generate(input) {
    if (input === null || typeof input !== 'object') throw new TypeError('generate: input must be an object');
    const { iteration, parent = null, feedback = null } = input;
    if (!Number.isSafeInteger(iteration)) {
      throw new TypeError(`generate: iteration must be an integer, got ${iteration}`);
    }
    if (iteration < 1) throw new RangeError(`generate: iteration must be >= 1, got ${iteration}`);
    if (parent !== null) assertParent(parent, count);

    const rng = createRng(hashSeed(seed, iteration));
    const { positions, strategy, stepSize } =
      parent === null
        ? project(dataset, settings, rng)
        : refine(parent.positions, model, settings, iteration, feedback, rng);

    return {
      id: `${seed}:${iteration}`,
      seed,
      iteration,
      parentId: parent === null ? null : parent.id,
      positions,
      meta: { strategy, stepSize },
    };
  }

  return Object.freeze({ seed, generate });
}

/* ------------------------------------------------------------------ */
/* Initial layout                                                     */
/* ------------------------------------------------------------------ */

/**
 * Projects every vector onto a random orthonormal pair of directions and fits
 * the result, with one common scale, inside the margins of the unit square.
 */
function project(dataset, settings, rng) {
  const { points, dim } = dataset;
  const count = points.length;
  const axisX = Array.from({ length: dim }, () => gaussian(rng));
  const axisY = Array.from({ length: dim }, () => gaussian(rng));
  const along = cosine(axisX, axisY) / cosine(axisX, axisX);
  for (let d = 0; d < dim; d++) axisY[d] -= along * axisX[d];

  const positions = new Float64Array(2 * count);
  const scaleX = 1 / Math.sqrt(cosine(axisX, axisX));
  const scaleY = 1 / Math.sqrt(cosine(axisY, axisY));
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < count; i++) {
    const x = cosine(points[i].vector, axisX) * scaleX;
    const y = cosine(points[i].vector, axisY) * scaleY;
    positions[2 * i] = x;
    positions[2 * i + 1] = y;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }

  const lo = LAYOUT_MIN + settings.margin;
  const hi = LAYOUT_MAX - settings.margin;
  const centre = (lo + hi) / 2;
  const extent = Math.max(maxX - minX, maxY - minY);
  const scale = extent > 0 ? (hi - lo) / extent : 0;
  for (let i = 0; i < count; i++) {
    positions[2 * i] = clampTo(centre + (positions[2 * i] - (minX + maxX) / 2) * scale, lo, hi);
    positions[2 * i + 1] = clampTo(centre + (positions[2 * i + 1] - (minY + maxY) / 2) * scale, lo, hi);
  }
  return { positions, strategy: PROJECT, stepSize: 0 };
}

/* ------------------------------------------------------------------ */
/* Refinement                                                         */
/* ------------------------------------------------------------------ */

/** Moves a copy of the parent positions by a few force-directed steps. */
function refine(parentPositions, model, settings, iteration, feedback, rng) {
  const strategy = pickStrategy(settings, rng);
  const shape = STRATEGIES[strategy];
  const emphasis = emphasisFrom(feedback, settings.feedbackGain);
  const annealed =
    settings.stepMin + (settings.stepMax - settings.stepMin) * 0.5 ** ((iteration - 1) / settings.stepHalfLife);
  const stepSize = annealed * shape.step * emphasis.step;

  const lo = LAYOUT_MIN + settings.margin;
  const hi = LAYOUT_MAX - settings.margin;
  const positions = Float64Array.from(parentPositions, (v) => clampTo(v, lo, hi));
  const params = {
    stepSize,
    attract: settings.attract * shape.attract * emphasis.neighbourhood,
    cohesion: settings.cohesion * shape.cohesion * emphasis.separation,
    repel: settings.repel * shape.repel * emphasis.separation,
    attractRange: settings.attractRange,
    repelRange: settings.repelRange,
    jitter: settings.jitter * shape.jitter,
    rest: 2 * POINT_RADIUS * (1 + settings.collisionPad),
    cohesionRest: settings.cohesionRest,
    minDistance: 2 * POINT_RADIUS * (1 + settings.collisionPad),
    collisionPasses: Math.max(1, Math.round(settings.collisionPasses * emphasis.overlap)),
    lo,
    hi,
  };
  for (let step = 0; step < settings.steps; step++) forceStep(positions, model, params, rng);
  // A last pass so the returned layout is as overlap-free as the step allows.
  collide(positions, params.minDistance, 3 * params.collisionPasses, lo, hi);
  return { positions, strategy, stepSize };
}

/** Seeded choice of the refinement strategy for this iteration. */
function pickStrategy(settings, rng) {
  const roll = rng();
  if (roll < settings.shakeChance) return 'shake';
  if (roll < settings.shakeChance + settings.relaxChance) return 'relax';
  return 'refine';
}

/**
 * Turns the evaluation of the parent into multipliers for the forces. Each
 * criterion's shortfall (1 - score) is compared with the total shortfall, so the
 * weakest criterion receives the largest boost. A parent that moved a lot
 * (low stability) shortens the next step.
 */
function emphasisFrom(feedback, gain) {
  const shortfall = (key) => {
    const value = feedback?.criteria?.[key];
    return typeof value === 'number' && Number.isFinite(value) ? 1 - clamp01(value) : 0;
  };
  const neighbourhood = shortfall('neighbourhood');
  const separation = shortfall('separation');
  const overlap = shortfall('overlap');
  const total = neighbourhood + separation + overlap + 0.1;
  return {
    neighbourhood: 1 + (gain * neighbourhood) / total,
    separation: 1 + (gain * separation) / total,
    overlap: 1 + (gain * overlap) / total,
    step: 1 - 0.5 * Math.min(1, gain) * shortfall('stability'),
  };
}

/* ------------------------------------------------------------------ */
/* Precomputation                                                     */
/* ------------------------------------------------------------------ */

/**
 * Everything the forces need that depends only on the vectors.
 * `gather[i * N + j]` grows with the similarity of i and j, `spread[i * N + j]`
 * with their dissimilarity; both are scaled so a point's row sums to about 1.
 */
function buildModel(dataset, k) {
  const points = dataset.points;
  const count = points.length;
  const neighbours = semanticNeighbours(dataset, k);
  const similarity = new Float64Array(count * count);
  let total = 0;
  for (let i = 0; i < count; i++) {
    for (let j = i + 1; j < count; j++) {
      const value = cosine(points[i].vector, points[j].vector);
      similarity[i * count + j] = similarity[j * count + i] = value;
      total += 2 * value;
    }
  }
  // Two reference levels taken from the data: how similar a typical nearest
  // neighbour is ("near") and how similar an average pair is ("far").
  let near = 0;
  for (let i = 0; i < count; i++) {
    for (const j of neighbours[i]) near += similarity[i * count + j];
  }
  near /= count * neighbours[0].length;
  const far = total / (count * (count - 1));
  const band = Math.max(near - far, 1e-9) * AFFINITY_WIDTH;

  const gather = new Float64Array(count * count);
  const spread = new Float64Array(count * count);
  let gatherSum = 0;
  let spreadSum = 0;
  for (let i = 0; i < count; i++) {
    for (let j = i + 1; j < count; j++) {
      const gap = Math.max(0, near - similarity[i * count + j]) / band;
      const g = Math.exp(-gap * gap);
      const s = 1 - g;
      gather[i * count + j] = gather[j * count + i] = g;
      spread[i * count + j] = spread[j * count + i] = s;
      gatherSum += 2 * g;
      spreadSum += 2 * s;
    }
  }
  const gatherScale = gatherSum > 0 ? count / gatherSum : 0;
  const spreadScale = spreadSum > 0 ? count / spreadSum : 0;
  for (let n = 0; n < gather.length; n++) {
    gather[n] *= gatherScale;
    spread[n] *= spreadScale;
  }
  return { count, neighbours, gather, spread };
}

/* ------------------------------------------------------------------ */
/* Argument checks                                                    */
/* ------------------------------------------------------------------ */

function assertDataset(dataset) {
  if (dataset === null || typeof dataset !== 'object' || !Array.isArray(dataset.points)) {
    throw new TypeError('createGenerator: dataset must be a Dataset with a points array');
  }
  if (dataset.points.length < 2) throw new RangeError('createGenerator: dataset must contain at least 2 points');
  if (!Number.isInteger(dataset.dim) || dataset.dim < 2) {
    throw new RangeError(`createGenerator: dataset.dim must be an integer >= 2, got ${dataset.dim}`);
  }
  for (const point of dataset.points) {
    if (!point || !point.vector || point.vector.length !== dataset.dim) {
      throw new TypeError(`createGenerator: every point needs a vector of length ${dataset.dim}`);
    }
  }
}

function assertParent(parent, count) {
  if (typeof parent !== 'object' || !(parent.positions instanceof Float64Array)) {
    throw new TypeError('generate: parent must be a CandidateLayout with Float64Array positions, or null');
  }
  if (parent.positions.length !== 2 * count) {
    throw new TypeError(
      `generate: parent.positions must have length ${2 * count}, got ${parent.positions.length}`,
    );
  }
  for (let n = 0; n < parent.positions.length; n++) {
    if (!Number.isFinite(parent.positions[n])) {
      throw new TypeError(`generate: parent.positions[${n}] is not finite`);
    }
  }
}

function resolveOptions(options) {
  if (options === null || typeof options !== 'object') {
    throw new TypeError('createGenerator: options must be an object');
  }
  const settings = { ...GENERATOR_DEFAULTS };
  for (const [key, value] of Object.entries(options)) {
    if (!(key in GENERATOR_DEFAULTS)) throw new TypeError(`createGenerator: unknown option "${key}"`);
    if (value === undefined) continue;
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      throw new TypeError(`createGenerator: option "${key}" must be a finite number, got ${value}`);
    }
    if (value < 0) throw new RangeError(`createGenerator: option "${key}" must be >= 0, got ${value}`);
    settings[key] = value;
  }
  for (const key of ['neighbours', 'steps', 'collisionPasses']) {
    if (!Number.isInteger(settings[key]) || settings[key] < 1) {
      throw new RangeError(`createGenerator: option "${key}" must be an integer >= 1, got ${settings[key]}`);
    }
  }
  if (settings.margin >= 0.5) throw new RangeError('createGenerator: option "margin" must be below 0.5');
  if (settings.stepMin > settings.stepMax) {
    throw new RangeError('createGenerator: option "stepMin" must not exceed "stepMax"');
  }
  for (const key of ['stepHalfLife', 'attractRange', 'repelRange']) {
    if (settings[key] <= 0) throw new RangeError(`createGenerator: option "${key}" must be > 0`);
  }
  if (settings.shakeChance + settings.relaxChance > 1) {
    throw new RangeError('createGenerator: "shakeChance" + "relaxChance" must not exceed 1');
  }
  return Object.freeze(settings);
}

function clampTo(value, lo, hi) {
  return value < lo ? lo : value > hi ? hi : value;
}
