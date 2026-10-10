/**
 * Seeded pseudo-random numbers. Nothing in this project calls Math.random for
 * anything that affects a layout or a score.
 */

/**
 * Mixes any number of integers into one unsigned 32-bit seed, so that a stream
 * can be derived from e.g. (seed, iteration) without carrying state around.
 * @param {...number} parts
 * @returns {number}
 */
export function hashSeed(...parts) {
  let h = 0x811c9dc5;
  for (const part of parts) {
    let k = Math.imul(part | 0, 0xcc9e2d51);
    k = (k << 15) | (k >>> 17);
    h ^= Math.imul(k, 0x1b873593);
    h = (h << 13) | (h >>> 19);
    h = (Math.imul(h, 5) + 0xe6546b64) | 0;
  }
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return h >>> 0;
}

/**
 * mulberry32: returns a function producing floats in [0, 1).
 * @param {number} seed
 * @returns {() => number}
 */
export function createRng(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Standard normal sample (Box-Muller) drawn from `rng`.
 * @param {() => number} rng
 * @returns {number}
 */
export function gaussian(rng) {
  let u = 0;
  while (u === 0) u = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}
