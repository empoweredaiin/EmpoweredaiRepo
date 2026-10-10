/**
 * Pure particle simulation for the renderer: palette, layout -> screen mapping,
 * particle state, the per-frame step, hit testing and the ambient dust field.
 *
 * Nothing here touches the DOM, so the module runs unchanged in Node. All
 * per-particle state lives in typed arrays and the step functions allocate
 * nothing.
 */
import { clamp01 } from '../contracts.js';
import { createRng, hashSeed } from '../core/rng.js';

/** Canvas background colour. */
export const BACKGROUND = '#07080d';

/**
 * One [r, g, b] colour per cluster, tuned to sit on near-black under additive
 * blending without tipping into neon.
 * @type {ReadonlyArray<ReadonlyArray<number>>}
 */
export const PALETTE = Object.freeze([
  Object.freeze([57, 135, 229]), // tides: blue
  Object.freeze([217, 89, 38]), // circuits: orange
  Object.freeze([25, 158, 112]), // orchards: aqua
  Object.freeze([201, 133, 0]), // archives: yellow
  Object.freeze([213, 81, 129]), // weather: magenta
  Object.freeze([0, 131, 0]), // cities: green
]);

/** Largest simulation step in seconds; longer frames are clamped to this. */
export const MAX_DT = 1 / 30;

/** Step used when no previous frame time is known. */
export const DEFAULT_DT = 1 / 60;

/** Radius in CSS pixels within which the pointer displaces particles. */
export const POINTER_RADIUS = 110;

/** Largest displacement in CSS pixels caused by the pointer. */
export const POINTER_PUSH = 16;

/** Angular frequencies (rad/s) of the critically damped springs. */
const OMEGA_LAYOUT = 9;
const OMEGA_IDLE = 2.5;
const OMEGA_REDUCED = 30;

/** Idle flow: differential rotation (rad/s) and wander amplitude (field radii). */
const IDLE_ROTATION = 0.03;
const IDLE_WANDER = 0.055;
const IDLE_MAX_RADIUS = 0.88;

/** Share of a canvas dimension that insets may take before they are scaled down. */
const MAX_INSET_SHARE = 0.75;

const ZERO_INSETS = Object.freeze({ top: 0, right: 0, bottom: 0, left: 0 });

/**
 * @typedef {Object} Insets  CSS pixels covered by overlays on each canvas edge.
 * @property {number} [top]
 * @property {number} [right]
 * @property {number} [bottom]
 * @property {number} [left]
 */

/**
 * Where things go on the canvas, in CSS pixels.
 * @typedef {Object} Viewport
 * @property {number} x       Left edge of the square the layout maps into.
 * @property {number} y       Top edge of that square.
 * @property {number} size    Side of that square.
 * @property {number} fieldX  Left edge of the free rectangle (canvas minus insets).
 * @property {number} fieldY  Top edge of the free rectangle.
 * @property {number} fieldW  Width of the free rectangle.
 * @property {number} fieldH  Height of the free rectangle.
 * @property {number} pad     Margin kept clear inside the free rectangle.
 */

/**
 * Per-frame inputs of the simulation. One object is reused across frames.
 * @typedef {Object} Frame
 * @property {Viewport} viewport
 * @property {boolean} pointerActive
 * @property {number} pointerX       CSS pixels.
 * @property {number} pointerY       CSS pixels.
 * @property {number} exempt         Particle index the pointer does not push, or -1.
 * @property {boolean} reducedMotion Disables drift and pointer displacement, eases fast.
 */

/**
 * @typedef {Object} ParticleSystem
 * @property {number} count
 * @property {number} time           Accumulated simulation time in seconds.
 * @property {boolean} hasTargets    False while idle.
 * @property {Float64Array} targets  Interleaved layout targets in the unit square.
 * @property {Float64Array} x        Position, CSS pixels.
 * @property {Float64Array} y
 * @property {Float64Array} vx       Velocity, CSS pixels per second.
 * @property {Float64Array} vy
 * @property {Float64Array} gx       Goal the particle eases towards this frame.
 * @property {Float64Array} gy
 * @property {Float64Array} homeR    Idle anchor, polar radius in [0, IDLE_MAX_RADIUS].
 * @property {Float64Array} homeA    Idle anchor, polar angle.
 * @property {Float64Array} phase    Per-particle phase in [0, 2π).
 * @property {Float64Array} depth    Pseudo-depth in [0, 1]; 1 is nearest.
 */

/**
 * @typedef {Object} Dust
 * @property {number} count
 * @property {Float64Array} x      Normalised [0, 1) across the canvas width.
 * @property {Float64Array} y      Normalised [0, 1) across the canvas height.
 * @property {Float64Array} depth  In [0, 1].
 * @property {Float64Array} phase  In [0, 2π).
 */

/**
 * Colour of a cluster. Indices beyond the palette wrap around.
 * @param {number} cluster
 * @returns {ReadonlyArray<number>} [r, g, b]
 */
export function clusterColour(cluster) {
  const n = PALETTE.length;
  return PALETTE[(((cluster | 0) % n) + n) % n];
}

/**
 * Clamps a frame delta to a safe simulation step, so a throttled or background
 * tab cannot make the simulation jump.
 * @param {number} dt Seconds.
 * @returns {number}
 */
export function clampDt(dt) {
  if (!Number.isFinite(dt) || dt <= 0) return DEFAULT_DT;
  return dt > MAX_DT ? MAX_DT : dt;
}

function nonNegative(value) {
  return Number.isFinite(value) && value > 0 ? value : 0;
}

/**
 * Computes the free rectangle left by the insets and the centred, padded
 * square that the unit layout square maps into. Insets that would swallow the
 * canvas are scaled down so a usable area always remains.
 * @param {number} width   Canvas width in CSS pixels.
 * @param {number} height  Canvas height in CSS pixels.
 * @param {Insets} [insets]
 * @param {Viewport} [out] Written in place when given, to avoid an allocation.
 * @returns {Viewport}
 */
export function computeViewport(width, height, insets, out) {
  const viewport = out ?? /** @type {Viewport} */ ({});
  const w = nonNegative(width);
  const h = nonNegative(height);
  const source = insets ?? ZERO_INSETS;
  let left = nonNegative(source.left);
  let right = nonNegative(source.right);
  let top = nonNegative(source.top);
  let bottom = nonNegative(source.bottom);

  const maxX = w * MAX_INSET_SHARE;
  if (left + right > maxX) {
    const scale = maxX / (left + right);
    left *= scale;
    right *= scale;
  }
  const maxY = h * MAX_INSET_SHARE;
  if (top + bottom > maxY) {
    const scale = maxY / (top + bottom);
    top *= scale;
    bottom *= scale;
  }

  const fieldW = w - left - right;
  const fieldH = h - top - bottom;
  const shortest = Math.min(fieldW, fieldH);
  const pad = Math.min(Math.max(shortest * 0.08, 12), 56, shortest * 0.25);
  const size = Math.max(shortest - 2 * pad, 0);

  viewport.x = left + (fieldW - size) / 2;
  viewport.y = top + (fieldH - size) / 2;
  viewport.size = size;
  viewport.fieldX = left;
  viewport.fieldY = top;
  viewport.fieldW = fieldW;
  viewport.fieldH = fieldH;
  viewport.pad = pad;
  return viewport;
}

/**
 * Maps a point of the unit layout square to CSS pixels.
 * @param {Viewport} viewport
 * @param {number} lx
 * @param {number} ly
 * @param {{ x: number, y: number }} [out]
 * @returns {{ x: number, y: number }}
 */
export function layoutToScreen(viewport, lx, ly, out) {
  const point = out ?? { x: 0, y: 0 };
  point.x = viewport.x + lx * viewport.size;
  point.y = viewport.y + ly * viewport.size;
  return point;
}

/**
 * @param {Viewport} [viewport]
 * @returns {Frame} A frame with no pointer and full motion.
 */
export function createFrame(viewport) {
  return {
    viewport: viewport ?? computeViewport(0, 0),
    pointerActive: false,
    pointerX: 0,
    pointerY: 0,
    exempt: -1,
    reducedMotion: false,
  };
}

/**
 * Creates the particle state. Idle anchors, phases and depths are drawn from a
 * seeded stream, so the same (count, seed) always gives the same first frame.
 * Positions start at the origin: call `snapParticles` once a viewport is known.
 * @param {{ count: number, seed?: number }} options
 * @returns {ParticleSystem}
 */
export function createParticles({ count, seed = 0 }) {
  const rng = createRng(hashSeed(seed, 0x9a17));
  const system = {
    count,
    time: 0,
    hasTargets: false,
    targets: new Float64Array(count * 2),
    x: new Float64Array(count),
    y: new Float64Array(count),
    vx: new Float64Array(count),
    vy: new Float64Array(count),
    gx: new Float64Array(count),
    gy: new Float64Array(count),
    homeR: new Float64Array(count),
    homeA: new Float64Array(count),
    phase: new Float64Array(count),
    depth: new Float64Array(count),
  };
  for (let i = 0; i < count; i++) {
    system.homeR[i] = Math.sqrt(rng()) * IDLE_MAX_RADIUS;
    system.homeA[i] = rng() * Math.PI * 2;
    system.phase[i] = rng() * Math.PI * 2;
    system.depth[i] = rng();
  }
  return system;
}

/**
 * Sets the layout the particles ease towards, or returns them to idle drift.
 * The input is copied (and clamped to the unit square); it is never retained.
 * @param {ParticleSystem} system
 * @param {ArrayLike<number>|null} positions Interleaved [x0, y0, x1, y1, ...] or null for idle.
 */
export function setTargets(system, positions) {
  if (positions === null || positions === undefined) {
    system.hasTargets = false;
    return;
  }
  if (positions.length !== system.count * 2) {
    throw new RangeError(`positions must have length ${system.count * 2}, got ${positions.length}`);
  }
  for (let i = 0; i < positions.length; i++) system.targets[i] = clamp01(positions[i]);
  system.hasTargets = true;
}

/**
 * Writes this frame's goal of every particle into `system.gx` / `system.gy`:
 * the layout target (or the drifting idle anchor), displaced away from the
 * pointer when it is close.
 * @param {ParticleSystem} system
 * @param {Frame} frame
 */
export function computeGoals(system, frame) {
  const { count, gx, gy, targets, homeR, homeA, phase, depth, time } = system;
  const view = frame.viewport;
  const still = frame.reducedMotion;

  if (system.hasTargets) {
    for (let i = 0; i < count; i++) {
      gx[i] = view.x + targets[2 * i] * view.size;
      gy[i] = view.y + targets[2 * i + 1] * view.size;
    }
  } else {
    const cx = view.fieldX + view.fieldW / 2;
    const cy = view.fieldY + view.fieldH / 2;
    const rx = view.fieldW / 2 - view.pad;
    const ry = view.fieldH / 2 - view.pad;
    for (let i = 0; i < count; i++) {
      const angle = still ? homeA[i] : homeA[i] + time * IDLE_ROTATION * (0.4 + depth[i]);
      let ux = homeR[i] * Math.cos(angle);
      let uy = homeR[i] * Math.sin(angle);
      if (!still) {
        const p = phase[i];
        ux += IDLE_WANDER * (Math.sin(uy * 5.1 + time * 0.21 + p) + 0.5 * Math.sin(ux * 9.3 - time * 0.13 + p * 1.7));
        uy += IDLE_WANDER * (Math.cos(ux * 4.7 + time * 0.19 + p * 0.6) + 0.5 * Math.cos(uy * 8.9 + time * 0.11 - p));
      }
      gx[i] = cx + rx * ux;
      gy[i] = cy + ry * uy;
    }
  }

  if (!frame.pointerActive || still) return;
  const px = frame.pointerX;
  const py = frame.pointerY;
  const exempt = frame.exempt;
  const r2 = POINTER_RADIUS * POINTER_RADIUS;
  for (let i = 0; i < count; i++) {
    if (i === exempt) continue;
    const dx = gx[i] - px;
    const dy = gy[i] - py;
    const d2 = dx * dx + dy * dy;
    if (d2 >= r2 || d2 < 1e-6) continue;
    const d = Math.sqrt(d2);
    const falloff = 1 - d / POINTER_RADIUS;
    const push = (POINTER_PUSH * falloff * falloff) / d;
    gx[i] += dx * push;
    gy[i] += dy * push;
  }
}

/**
 * Advances the simulation by `dt` seconds: every particle follows its goal as
 * a critically damped spring, integrated in closed form so it is stable for
 * any step. The caller is expected to pass a clamped dt (see `clampDt`).
 * @param {ParticleSystem} system
 * @param {number} dt Seconds.
 * @param {Frame} frame
 */
export function stepParticles(system, dt, frame) {
  system.time += dt;
  computeGoals(system, frame);
  const { count, x, y, vx, vy, gx, gy } = system;
  const omega = frame.reducedMotion ? OMEGA_REDUCED : system.hasTargets ? OMEGA_LAYOUT : OMEGA_IDLE;
  const decay = Math.exp(-omega * dt);
  for (let i = 0; i < count; i++) {
    const dx = x[i] - gx[i];
    const kx = (vx[i] + omega * dx) * dt;
    x[i] = gx[i] + (dx + kx) * decay;
    vx[i] = (vx[i] - omega * kx) * decay;

    const dy = y[i] - gy[i];
    const ky = (vy[i] + omega * dy) * dt;
    y[i] = gy[i] + (dy + ky) * decay;
    vy[i] = (vy[i] - omega * ky) * decay;
  }
}

/**
 * Places every particle on its goal with zero velocity, without animation.
 * @param {ParticleSystem} system
 * @param {Frame} frame
 */
export function snapParticles(system, frame) {
  computeGoals(system, frame);
  system.x.set(system.gx);
  system.y.set(system.gy);
  system.vx.fill(0);
  system.vy.fill(0);
}

/**
 * Finds the particle nearest to a point.
 * @param {ParticleSystem} system
 * @param {number} px     CSS pixels.
 * @param {number} py     CSS pixels.
 * @param {number} radius Hit radius in CSS pixels.
 * @returns {number|null} Index of the nearest particle within `radius`, else null.
 */
export function hitTest(system, px, py, radius) {
  const { count, x, y } = system;
  let best = -1;
  let bestD2 = radius * radius;
  for (let i = 0; i < count; i++) {
    const dx = x[i] - px;
    const dy = y[i] - py;
    const d2 = dx * dx + dy * dy;
    if (d2 <= bestD2) {
      bestD2 = d2;
      best = i;
    }
  }
  return best === -1 ? null : best;
}

/**
 * Creates the ambient dust field, in canvas-normalised coordinates so it
 * survives a resize untouched.
 * @param {{ count: number, seed?: number }} options
 * @returns {Dust}
 */
export function createDust({ count, seed = 0 }) {
  const rng = createRng(hashSeed(seed, 0xd057));
  const dust = {
    count,
    x: new Float64Array(count),
    y: new Float64Array(count),
    depth: new Float64Array(count),
    phase: new Float64Array(count),
  };
  for (let i = 0; i < count; i++) {
    dust.x[i] = rng();
    dust.y[i] = rng();
    dust.depth[i] = rng();
    dust.phase[i] = rng() * Math.PI * 2;
  }
  return dust;
}

/**
 * Drifts the dust along a slow flow field, wrapping at the edges.
 * @param {Dust} dust
 * @param {number} dt   Seconds.
 * @param {number} time Simulation time in seconds.
 */
export function stepDust(dust, dt, time) {
  const { count, x, y, depth, phase } = dust;
  for (let i = 0; i < count; i++) {
    const angle = 1.7 * Math.sin(x[i] * 6.1 + time * 0.05 + phase[i]) + 1.7 * Math.cos(y[i] * 5.3 - time * 0.04);
    const speed = (0.004 + 0.01 * depth[i]) * dt;
    const nx = x[i] + Math.cos(angle) * speed;
    const ny = y[i] + Math.sin(angle) * speed;
    x[i] = nx - Math.floor(nx);
    y[i] = ny - Math.floor(ny);
  }
}
