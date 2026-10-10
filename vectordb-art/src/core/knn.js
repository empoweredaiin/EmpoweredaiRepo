/**
 * Nearest-neighbour helpers shared by the generator, the discriminator and the
 * renderer. Brute force: N is a few hundred points.
 */

/** @type {WeakMap<object, Map<number, Int32Array[]>>} */
const semanticCache = new WeakMap();

/**
 * Cosine similarity of two unit vectors.
 * @param {ArrayLike<number>} a
 * @param {ArrayLike<number>} b
 */
export function cosine(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

/**
 * For each of `count` items, the indices of its k nearest others, closest
 * first. Ties break on the lower index, so the result is deterministic.
 * @param {number} count
 * @param {(i: number, j: number) => number} distance  Smaller means closer.
 * @param {number} k
 * @returns {Int32Array[]}
 */
export function nearestBy(count, distance, k) {
  const size = Math.max(0, Math.min(k, count - 1));
  const result = new Array(count);
  const order = new Int32Array(Math.max(0, count - 1));
  const dist = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    let n = 0;
    for (let j = 0; j < count; j++) {
      if (j === i) continue;
      dist[j] = distance(i, j);
      order[n++] = j;
    }
    const sorted = Array.from(order).sort((a, b) => dist[a] - dist[b] || a - b);
    result[i] = Int32Array.from(sorted.slice(0, size));
  }
  return result;
}

/**
 * k nearest neighbours of every dataset point in embedding space (cosine).
 * Cached per dataset and k.
 * @param {import('../contracts.js').Dataset} dataset
 * @param {number} k
 * @returns {Int32Array[]}
 */
export function semanticNeighbours(dataset, k) {
  let byK = semanticCache.get(dataset);
  if (!byK) semanticCache.set(dataset, (byK = new Map()));
  let neighbours = byK.get(k);
  if (!neighbours) {
    const points = dataset.points;
    neighbours = nearestBy(points.length, (i, j) => 1 - cosine(points[i].vector, points[j].vector), k);
    byK.set(k, neighbours);
  }
  return neighbours;
}

/**
 * k nearest neighbours of every point in a 2D layout (Euclidean).
 * @param {Float64Array} positions  Interleaved [x0, y0, x1, y1, ...].
 * @param {number} k
 * @returns {Int32Array[]}
 */
export function layoutNeighbours(positions, k) {
  return nearestBy(
    positions.length / 2,
    (i, j) => {
      const dx = positions[2 * i] - positions[2 * j];
      const dy = positions[2 * i + 1] - positions[2 * j + 1];
      return dx * dx + dy * dy;
    },
    k,
  );
}
