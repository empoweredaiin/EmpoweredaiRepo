/**
 * Heuristic evaluator ("discriminator") for candidate layouts.
 *
 * Every score here is a transparent, hand-written geometric measure. Nothing
 * is learned or trained: the same layout always gets the same score, and each
 * criterion is exported as its own pure function so it can be inspected and
 * tested in isolation.
 *
 * All criteria work on interleaved positions [x0, y0, x1, y1, ...] in the unit
 * square, return a finite number in [0, 1] and treat higher as better.
 */
import {
  CRITERIA,
  DEFAULT_WEIGHTS,
  NEIGHBOUR_K,
  POINT_RADIUS,
  candidateIssues,
  clamp01,
} from '../contracts.js';
import { semanticNeighbours } from '../core/knn.js';

/**
 * Defaults of every option accepted by {@link createDiscriminator}.
 *
 * - `weights`         Weight of each criterion in the overall score.
 * - `k`               Neighbourhood size used by the neighbourhood criterion.
 * - `pointRadius`     Radius of a point; centres closer than twice this overlap.
 * - `silhouetteFull`  Mean silhouette at which separation reaches 1.
 * - `stabilityScale`  Mean per-point displacement at which stability reaches 0.
 */
export const DISCRIMINATOR_DEFAULTS = Object.freeze({
  weights: DEFAULT_WEIGHTS,
  k: NEIGHBOUR_K,
  pointRadius: POINT_RADIUS,
  silhouetteFull: 0.6,
  stabilityScale: 0.25,
});

/** One-sentence plain-English description of each criterion, keyed like CRITERIA. */
export const CRITERION_DESCRIPTIONS = Object.freeze({
  neighbourhood:
    'How many of each point’s closest matches in the vector database are still among its closest neighbours on screen.',
  separation:
    'How clearly the labelled topics form their own groups, with each point nearer to its own topic than to any other.',
  overlap: 'How little the points sit on top of one another; it is 1 when no two points touch.',
  stability:
    'How little the points moved compared with the best layout so far; it is 1 when nothing moved.',
});

/**
 * Semantic-neighbourhood preservation.
 *
 * For each point i with semantic neighbour list S_i (size k_i), let L_i be its
 * k_i nearest neighbours in the layout (Euclidean, ties broken on the lower
 * index). The score is the mean over points of |S_i ∩ L_i| / k_i. Points with
 * an empty neighbour list are skipped; if every list is empty the score is 1,
 * because there is nothing to preserve.
 *
 * @param {ArrayLike<number>} positions               Interleaved, length 2 * N.
 * @param {ReadonlyArray<ArrayLike<number>>} semantic  Per point, indices of its semantic neighbours.
 * @returns {number} In [0, 1].
 */
export function scoreNeighbourhood(positions, semantic) {
  const count = positions.length >> 1;
  const dist = new Float64Array(count);
  let maxK = 0;
  for (let i = 0; i < count; i++) maxK = Math.max(maxK, Math.min(semantic[i].length, count - 1));
  // The k smallest distances seen so far, ascending, with their point indices.
  const nearDist = new Float64Array(maxK);
  const nearIndex = new Int32Array(maxK);

  let total = 0;
  let scored = 0;
  for (let i = 0; i < count; i++) {
    const wanted = semantic[i];
    const k = Math.min(wanted.length, count - 1);
    if (k === 0) continue;
    const xi = positions[2 * i];
    const yi = positions[2 * i + 1];

    // Partial selection: keep only the k nearest, by insertion. Visiting j in
    // ascending order and inserting on strictly smaller distances keeps the
    // lower index first among equals.
    let filled = 0;
    for (let j = 0; j < count; j++) {
      if (j === i) continue;
      const dx = positions[2 * j] - xi;
      const dy = positions[2 * j + 1] - yi;
      const d = dx * dx + dy * dy;
      dist[j] = d;
      if (filled === k && d >= nearDist[k - 1]) continue;
      let slot = filled < k ? filled++ : k - 1;
      while (slot > 0 && nearDist[slot - 1] > d) {
        nearDist[slot] = nearDist[slot - 1];
        nearIndex[slot] = nearIndex[slot - 1];
        slot--;
      }
      nearDist[slot] = d;
      nearIndex[slot] = j;
    }

    // j is one of the k nearest iff (distance, index) is no later than the k-th entry.
    const cutDist = nearDist[k - 1];
    const cutIndex = nearIndex[k - 1];
    let hits = 0;
    for (let n = 0; n < wanted.length; n++) {
      const j = wanted[n];
      if (dist[j] < cutDist || (dist[j] === cutDist && j <= cutIndex)) hits++;
    }
    total += hits / wanted.length;
    scored++;
  }
  return scored === 0 ? 1 : clamp01(total / scored);
}

/**
 * Mean silhouette coefficient of the labelled clusters in the layout, in [-1, 1].
 *
 * For point i: a = mean distance to the other members of its own cluster,
 * b = smallest mean distance to the members of any other cluster, and
 * s_i = (b - a) / max(a, b). By the usual convention s_i = 0 when the point is
 * alone in its cluster, when there is no other cluster, or when a = b = 0
 * (everything coincident). The result is the mean of s_i over all points, 0
 * for an empty layout.
 *
 * @param {ArrayLike<number>} positions  Interleaved, length 2 * N.
 * @param {ArrayLike<number>} clusters   Per point, a non-negative integer cluster index.
 * @returns {number}
 */
export function meanSilhouette(positions, clusters) {
  const count = positions.length >> 1;
  if (count === 0) return 0;
  let clusterCount = 0;
  for (let i = 0; i < count; i++) clusterCount = Math.max(clusterCount, clusters[i] + 1);
  const sizes = new Int32Array(clusterCount);
  for (let i = 0; i < count; i++) sizes[clusters[i]]++;
  const sums = new Float64Array(clusterCount);

  let total = 0;
  for (let i = 0; i < count; i++) {
    const own = clusters[i];
    if (sizes[own] < 2) continue;
    sums.fill(0);
    const xi = positions[2 * i];
    const yi = positions[2 * i + 1];
    for (let j = 0; j < count; j++) {
      const dx = positions[2 * j] - xi;
      const dy = positions[2 * j + 1] - yi;
      sums[clusters[j]] += Math.sqrt(dx * dx + dy * dy);
    }
    const a = sums[own] / (sizes[own] - 1);
    let b = Infinity;
    for (let c = 0; c < clusterCount; c++) {
      if (c === own || sizes[c] === 0) continue;
      const mean = sums[c] / sizes[c];
      if (mean < b) b = mean;
    }
    const larger = Math.max(a, b);
    if (b !== Infinity && larger > 0) total += (b - a) / larger;
  }
  return total / count;
}

/**
 * Cluster separation: the mean silhouette s mapped to [0, 1] by
 *
 *   separation = clamp01(0.5 + 0.5 * s / silhouetteFull)
 *
 * so s = 0 (topics fully mixed, as in a random scatter) gives 0.5, layouts
 * that are worse than random fall towards 0, and s >= silhouetteFull gives 1.
 * The default silhouetteFull is 0.6 rather than the theoretical maximum of 1:
 * points have a physical radius, so a cluster cannot shrink to a dot, and the
 * dataset's topics come in related pairs that belong next to each other. Tidy,
 * non-overlapping blobs with modest gaps reach s of about 0.6 to 0.75, which
 * this mapping scores as 1; s of about 0.5 already scores above 0.9.
 *
 * @param {ArrayLike<number>} positions  Interleaved, length 2 * N.
 * @param {ArrayLike<number>} clusters   Per point, a non-negative integer cluster index.
 * @param {number} [silhouetteFull]      Mean silhouette that maps to 1. Must be > 0.
 * @returns {number} In [0, 1].
 */
export function scoreSeparation(positions, clusters, silhouetteFull = DISCRIMINATOR_DEFAULTS.silhouetteFull) {
  return clamp01(0.5 + (0.5 * meanSilhouette(positions, clusters)) / silhouetteFull);
}

/**
 * Freedom from overlap, measured as nearest-neighbour penetration.
 *
 * For each point, d is the distance to its nearest other point and its
 * penetration is max(0, 1 - d / (2 * pointRadius)): 0 when the nearest point
 * does not touch it, 1 when the two are exactly stacked. The score is
 * 1 - mean penetration, so it is exactly 1 when no two centres are closer
 * than 2 * pointRadius, 0 when every point is stacked on another, and changes
 * smoothly in between. Fewer than two points cannot overlap and score 1.
 *
 * @param {ArrayLike<number>} positions  Interleaved, length 2 * N.
 * @param {number} [pointRadius]         Must be > 0.
 * @returns {number} In [0, 1].
 */
export function scoreOverlap(positions, pointRadius = DISCRIMINATOR_DEFAULTS.pointRadius) {
  const count = positions.length >> 1;
  if (count < 2) return 1;
  const diameter = 2 * pointRadius;
  const diameterSquared = diameter * diameter;
  let penetration = 0;
  for (let i = 0; i < count; i++) {
    const xi = positions[2 * i];
    const yi = positions[2 * i + 1];
    let nearest = diameterSquared;
    for (let j = 0; j < count; j++) {
      if (j === i) continue;
      const dx = positions[2 * j] - xi;
      const dy = positions[2 * j + 1] - yi;
      const d = dx * dx + dy * dy;
      if (d < nearest) nearest = d;
    }
    if (nearest < diameterSquared) penetration += 1 - Math.sqrt(nearest) / diameter;
  }
  return clamp01(1 - penetration / count);
}

/**
 * Stability relative to a previous layout:
 *
 *   stability = clamp01(1 - meanDisplacement / stabilityScale)
 *
 * where meanDisplacement is the mean Euclidean distance each point moved. With
 * the default stabilityScale of 0.25, an unchanged layout scores 1 and a mean
 * move of a quarter of the unit square's side or more scores 0. A missing
 * previous layout, or one that cannot be compared (different length,
 * non-finite values), scores 1.
 *
 * @param {ArrayLike<number>} positions                 Interleaved, length 2 * N.
 * @param {ArrayLike<number>|null|undefined} previous   Positions of the previous layout.
 * @param {number} [stabilityScale]                     Mean displacement that maps to 0. Must be > 0.
 * @returns {number} In [0, 1].
 */
export function scoreStability(positions, previous, stabilityScale = DISCRIMINATOR_DEFAULTS.stabilityScale) {
  if (!previous || previous.length !== positions.length || positions.length < 2) return 1;
  const count = positions.length >> 1;
  let moved = 0;
  for (let i = 0; i < count; i++) {
    const dx = positions[2 * i] - previous[2 * i];
    const dy = positions[2 * i + 1] - previous[2 * i + 1];
    moved += Math.sqrt(dx * dx + dy * dy);
  }
  if (!Number.isFinite(moved)) return 1;
  return clamp01(1 - moved / count / stabilityScale);
}

/**
 * Validates criterion weights and scales them to sum to 1.
 * @param {unknown} weights
 * @returns {Readonly<Record<string, number>>}
 */
function normaliseWeights(weights) {
  if (weights === null || typeof weights !== 'object') {
    throw new TypeError('createDiscriminator: options.weights must be an object');
  }
  const unknown = Object.keys(weights).filter((key) => !CRITERIA.includes(key));
  if (unknown.length > 0) {
    throw new TypeError(`createDiscriminator: unknown weight(s): ${unknown.join(', ')}`);
  }
  let sum = 0;
  for (const key of CRITERIA) {
    const value = weights[key];
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
      throw new TypeError(`createDiscriminator: options.weights.${key} must be a finite number >= 0`);
    }
    sum += value;
  }
  if (!(sum > 0) || !Number.isFinite(sum)) {
    throw new TypeError('createDiscriminator: options.weights must have a finite, positive sum');
  }
  // Weights that already sum to 1 up to rounding are kept exactly as given, so
  // the defaults are reported as 0.4, not 0.4000000000000001.
  const divisor = Math.abs(sum - 1) < 1e-9 ? 1 : sum;
  const normalised = {};
  for (const key of CRITERIA) normalised[key] = weights[key] / divisor;
  return Object.freeze(normalised);
}

/**
 * @param {string} name
 * @param {unknown} value
 */
function requirePositive(name, value) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    throw new TypeError(`createDiscriminator: options.${name} must be a finite number > 0`);
  }
}

/**
 * Checks the dataset and returns each point's cluster as a dense index.
 * @param {unknown} dataset
 * @returns {Int32Array}
 */
function clusterIndices(dataset) {
  if (dataset === null || typeof dataset !== 'object' || !Array.isArray(dataset.points)) {
    throw new TypeError('createDiscriminator: dataset must be a Dataset with a points array');
  }
  const { points } = dataset;
  if (points.length === 0) throw new TypeError('createDiscriminator: dataset.points must not be empty');
  const dense = new Map();
  const clusters = new Int32Array(points.length);
  for (let i = 0; i < points.length; i++) {
    const point = points[i];
    if (point === null || typeof point !== 'object' || typeof point.vector?.length !== 'number') {
      throw new TypeError(`createDiscriminator: dataset.points[${i}] must have a vector`);
    }
    if (!Number.isInteger(point.cluster) || point.cluster < 0) {
      throw new TypeError(`createDiscriminator: dataset.points[${i}].cluster must be an integer >= 0`);
    }
    if (!dense.has(point.cluster)) dense.set(point.cluster, dense.size);
    clusters[i] = dense.get(point.cluster);
  }
  return clusters;
}

/**
 * Builds the heuristic evaluator for one dataset. The semantic neighbours and
 * cluster indices are computed once here; `evaluate` then only does geometry.
 *
 * @param {Object} config
 * @param {import('../contracts.js').Dataset} config.dataset
 * @param {Object} [config.options]
 * @param {Record<string, number>} [config.options.weights]  All four criteria, finite and >= 0;
 *        scaled to sum to 1. Default DEFAULT_WEIGHTS.
 * @param {number} [config.options.k]               Integer >= 1. Default NEIGHBOUR_K.
 * @param {number} [config.options.pointRadius]     > 0. Default POINT_RADIUS.
 * @param {number} [config.options.silhouetteFull]  > 0. Default 0.6. See {@link scoreSeparation}.
 * @param {number} [config.options.stabilityScale]  > 0. Default 0.25. See {@link scoreStability}.
 * @returns {import('../contracts.js').Discriminator}
 * @throws {TypeError} When the dataset or an option is invalid.
 */
export function createDiscriminator({ dataset, options = {} } = {}) {
  const clusters = clusterIndices(dataset);
  if (options === null || typeof options !== 'object') {
    throw new TypeError('createDiscriminator: options must be an object');
  }
  const { weights: rawWeights, k, pointRadius, silhouetteFull, stabilityScale } = {
    ...DISCRIMINATOR_DEFAULTS,
    ...options,
  };
  const weights = normaliseWeights(rawWeights);
  if (!Number.isInteger(k) || k < 1) {
    throw new TypeError('createDiscriminator: options.k must be an integer >= 1');
  }
  requirePositive('pointRadius', pointRadius);
  requirePositive('silhouetteFull', silhouetteFull);
  requirePositive('stabilityScale', stabilityScale);

  const pointCount = clusters.length;
  const semantic = semanticNeighbours(dataset, k);

  /**
   * @param {import('../contracts.js').EvaluateInput} input
   * @returns {import('../contracts.js').Evaluation}
   */
  function evaluate(input) {
    const candidate = input?.candidate;
    const issues = candidateIssues(candidate, pointCount);
    if (issues.length > 0) {
      const id = candidate?.id;
      return {
        candidateId: typeof id === 'string' && id !== '' ? id : 'invalid',
        kind: 'heuristic',
        criteria: { neighbourhood: 0, separation: 0, overlap: 0, stability: 0 },
        weights: { ...weights },
        overall: 0,
        valid: false,
        issues,
      };
    }

    const { positions } = candidate;
    const previous = input.previous?.positions;
    const criteria = {
      neighbourhood: scoreNeighbourhood(positions, semantic),
      separation: scoreSeparation(positions, clusters, silhouetteFull),
      overlap: scoreOverlap(positions, pointRadius),
      stability: scoreStability(positions, previous instanceof Float64Array ? previous : null, stabilityScale),
    };
    let overall = 0;
    for (const key of CRITERIA) overall += weights[key] * criteria[key];
    return {
      candidateId: candidate.id,
      kind: 'heuristic',
      criteria,
      weights: { ...weights },
      overall: clamp01(overall),
      valid: true,
      issues: [],
    };
  }

  return { evaluate };
}
