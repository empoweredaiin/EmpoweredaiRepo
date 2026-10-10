/**
 * Force-directed steps used by the generator to refine a layout in place.
 * Both functions only ever write to the `positions` array they are given, which
 * the generator has already copied from the parent.
 */
import { gaussian } from '../core/rng.js';

/** Golden angle, used to pick a separation direction for coincident points. */
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

/**
 * @typedef {Object} ForceModel
 * @property {number} count               Number of points N.
 * @property {ArrayLike<number>[]} neighbours  Semantic nearest neighbours per point.
 * @property {Float64Array} gather        N x N weights, larger for similar pairs.
 * @property {Float64Array} spread        N x N weights, larger for dissimilar pairs.
 */

/**
 * @typedef {Object} ForceParams
 * @property {number} stepSize       Largest move of one point caused by the forces.
 * @property {number} attract        Weight of the pull towards semantic neighbours.
 * @property {number} cohesion       Weight of the pull between similar points.
 * @property {number} repel          Weight of the push between dissimilar points.
 * @property {number} attractRange   Distance at which a pull saturates.
 * @property {number} repelRange     Distance scale of the push.
 * @property {number} jitter         Random nudge, as a fraction of stepSize.
 * @property {number} rest           Distance at which semantic neighbours stop pulling.
 * @property {number} cohesionRest   Distance at which similar points stop pulling.
 * @property {number} minDistance    Clearance enforced between any two points.
 * @property {number} collisionPasses
 * @property {number} lo             Lower bound of both coordinates.
 * @property {number} hi             Upper bound of both coordinates.
 */

/**
 * One step: accumulate the three similarity-derived forces, move every point by
 * at most `stepSize` along its net force, add seeded jitter, then resolve
 * overlaps and keep everything inside [lo, hi].
 *
 * @param {Float64Array} positions  Interleaved [x0, y0, ...]; modified in place.
 * @param {ForceModel} model
 * @param {ForceParams} params
 * @param {() => number} rng
 */
export function forceStep(positions, model, params, rng) {
  const { count, neighbours, gather, spread } = model;
  const { stepSize, attract, cohesion, repel, attractRange, repelRange, rest, cohesionRest, lo, hi } = params;
  const force = new Float64Array(2 * count);
  const repelRangeSq = repelRange * repelRange;

  // Similar points gather, dissimilar points spread: every pair, once.
  for (let i = 0; i < count; i++) {
    const xi = positions[2 * i];
    const yi = positions[2 * i + 1];
    const row = i * count;
    for (let j = i + 1; j < count; j++) {
      const dx = positions[2 * j] - xi;
      const dy = positions[2 * j + 1] - yi;
      const distSq = dx * dx + dy * dy;
      const dist = Math.sqrt(distSq) + 1e-12;
      const pull =
        dist > cohesionRest ? cohesion * gather[row + j] * Math.min(1, (dist - cohesionRest) / attractRange) : 0;
      const push = (repel * spread[row + j] * repelRangeSq) / (repelRangeSq + distSq);
      const fx = ((pull - push) * dx) / dist;
      const fy = ((pull - push) * dy) / dist;
      force[2 * i] += fx;
      force[2 * i + 1] += fy;
      force[2 * j] -= fx;
      force[2 * j + 1] -= fy;
    }
  }

  // Each point and its semantic nearest neighbours pull on each other.
  for (let i = 0; i < count; i++) {
    const near = neighbours[i];
    const share = attract / near.length;
    for (let n = 0; n < near.length; n++) {
      const j = near[n];
      const dx = positions[2 * j] - positions[2 * i];
      const dy = positions[2 * j + 1] - positions[2 * i + 1];
      const dist = Math.sqrt(dx * dx + dy * dy) + 1e-12;
      if (dist <= rest) continue;
      const pull = share * Math.min(1, (dist - rest) / attractRange);
      const fx = (pull * dx) / dist;
      const fy = (pull * dy) / dist;
      force[2 * i] += fx;
      force[2 * i + 1] += fy;
      force[2 * j] -= fx;
      force[2 * j + 1] -= fy;
    }
  }

  const noise = stepSize * params.jitter;
  for (let i = 0; i < count; i++) {
    const fx = force[2 * i];
    const fy = force[2 * i + 1];
    const magnitude = Math.sqrt(fx * fx + fy * fy);
    const scale = magnitude > 1 ? stepSize / magnitude : stepSize;
    positions[2 * i] += fx * scale + noise * gaussian(rng);
    positions[2 * i + 1] += fy * scale + noise * gaussian(rng);
  }

  collide(positions, params.minDistance, params.collisionPasses, lo, hi);
}

/**
 * Pushes apart every pair of points closer than `minDistance`, each by half the
 * shortfall, then clamps all coordinates into [lo, hi]. Repeated `passes` times
 * because resolving one overlap can create another.
 *
 * @param {Float64Array} positions  Interleaved [x0, y0, ...]; modified in place.
 * @param {number} minDistance
 * @param {number} passes
 * @param {number} lo
 * @param {number} hi
 */
export function collide(positions, minDistance, passes, lo, hi) {
  const count = positions.length / 2;
  const minSq = minDistance * minDistance;
  for (let pass = 0; pass < passes; pass++) {
    for (let i = 0; i < count; i++) {
      for (let j = i + 1; j < count; j++) {
        let dx = positions[2 * j] - positions[2 * i];
        if (dx >= minDistance || dx <= -minDistance) continue;
        let dy = positions[2 * j + 1] - positions[2 * i + 1];
        const distSq = dx * dx + dy * dy;
        if (distSq >= minSq) continue;
        let dist = Math.sqrt(distSq);
        if (dist < 1e-9) {
          const angle = GOLDEN_ANGLE * (i + j + 1);
          dx = Math.cos(angle);
          dy = Math.sin(angle);
          dist = 1;
        }
        const shift = (minDistance - Math.sqrt(distSq)) / 2 / dist;
        positions[2 * i] -= dx * shift;
        positions[2 * i + 1] -= dy * shift;
        positions[2 * j] += dx * shift;
        positions[2 * j + 1] += dy * shift;
      }
    }
    for (let n = 0; n < positions.length; n++) {
      const v = positions[n];
      positions[n] = v < lo ? lo : v > hi ? hi : v;
    }
  }
}
