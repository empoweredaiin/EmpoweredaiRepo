import assert from 'node:assert/strict';
import { test } from 'node:test';

import { createDataset } from '../src/data/dataset.js';
import {
  MAX_DT,
  PALETTE,
  POINTER_PUSH,
  POINTER_RADIUS,
  clampDt,
  clusterColour,
  computeGoals,
  computeViewport,
  createDust,
  createFrame,
  createParticles,
  hitTest,
  layoutToScreen,
  setTargets,
  snapParticles,
  stepDust,
  stepParticles,
} from '../src/render/particles.js';
import { TERMINATED_DURATION, createPulseState, stepPulses, triggerPulse } from '../src/render/pulses.js';
import { createRenderer } from '../src/render/renderer.js';

const DT = 1 / 60;

/** A frame over an 800 x 600 canvas with no insets. */
function frameFor(width = 800, height = 600, insets) {
  return createFrame(computeViewport(width, height, insets));
}

/** Deterministic layout: point i sits on a diagonal-ish lattice inside the unit square. */
function latticeLayout(count) {
  const positions = new Float64Array(count * 2);
  for (let i = 0; i < count; i++) {
    positions[2 * i] = ((i * 37) % 101) / 100;
    positions[2 * i + 1] = ((i * 53) % 97) / 96;
  }
  return positions;
}

/* ------------------------------------------------------------------ */
/* Palette and time step                                              */
/* ------------------------------------------------------------------ */

test('palette has one distinct colour per dataset cluster', () => {
  const dataset = createDataset();
  assert.ok(PALETTE.length >= dataset.clusterNames.length);
  const seen = new Set(dataset.clusterNames.map((_, c) => clusterColour(c).join(',')));
  assert.equal(seen.size, dataset.clusterNames.length);
  assert.deepEqual(clusterColour(PALETTE.length), clusterColour(0));
  assert.deepEqual(clusterColour(-1), clusterColour(PALETTE.length - 1));
});

test('clampDt bounds long, zero and invalid frame deltas', () => {
  assert.equal(clampDt(5), MAX_DT);
  assert.equal(clampDt(0.01), 0.01);
  for (const bad of [0, -1, NaN, Infinity]) {
    const dt = clampDt(bad);
    assert.ok(dt > 0 && dt <= MAX_DT, `clampDt(${bad}) = ${dt}`);
  }
});

/* ------------------------------------------------------------------ */
/* Layout -> screen mapping                                           */
/* ------------------------------------------------------------------ */

test('viewport is a centred, padded square when there are no insets', () => {
  const v = computeViewport(800, 600);
  assert.ok(v.size > 0 && v.size < 600);
  assert.ok(Math.abs(v.x + v.size / 2 - 400) < 1e-9);
  assert.ok(Math.abs(v.y + v.size / 2 - 300) < 1e-9);
  assert.ok(v.pad >= 12);
});

test('layout maps inside the canvas and clear of the insets', () => {
  const cases = [
    [1280, 800, { left: 340 }],
    [1280, 800, { top: 10, right: 20, bottom: 30, left: 340 }],
    [320, 568, { bottom: 260 }],
    [320, 480, undefined],
    [2560, 1440, { left: 340, right: 0, top: 0, bottom: 0 }],
  ];
  for (const [width, height, insets] of cases) {
    const v = computeViewport(width, height, insets);
    const left = insets?.left ?? 0;
    const right = insets?.right ?? 0;
    const top = insets?.top ?? 0;
    const bottom = insets?.bottom ?? 0;
    for (const [lx, ly] of [[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]]) {
      const p = layoutToScreen(v, lx, ly);
      assert.ok(p.x >= left + v.pad - 1e-9 && p.x <= width - right - v.pad + 1e-9, `x ${p.x} for ${width}x${height}`);
      assert.ok(p.y >= top + v.pad - 1e-9 && p.y <= height - bottom - v.pad + 1e-9, `y ${p.y} for ${width}x${height}`);
    }
    // Aspect is preserved: one layout unit is the same length on both axes.
    const a = layoutToScreen(v, 0, 0);
    const b = layoutToScreen(v, 1, 1);
    assert.ok(Math.abs(b.x - a.x - (b.y - a.y)) < 1e-9);
    // Centred in the free rectangle.
    assert.ok(Math.abs(v.x + v.size / 2 - (left + (width - left - right) / 2)) < 1e-9);
  }
});

test('oversized or invalid insets never push the layout off the canvas', () => {
  for (const insets of [{ left: 5000 }, { left: 300, right: 300 }, { top: NaN, left: -40 }, { bottom: Infinity }]) {
    const v = computeViewport(320, 480, insets);
    assert.ok(v.size > 0, 'a usable square remains');
    for (const [lx, ly] of [[0, 0], [1, 1]]) {
      const p = layoutToScreen(v, lx, ly);
      assert.ok(p.x >= 0 && p.x <= 320 && p.y >= 0 && p.y <= 480, JSON.stringify(p));
    }
  }
  const empty = computeViewport(0, 0);
  assert.equal(empty.size, 0);
  assert.ok(Number.isFinite(empty.x) && Number.isFinite(empty.y));
});

test('computeViewport writes into the object it is given', () => {
  const out = computeViewport(100, 100);
  assert.equal(computeViewport(800, 600, { left: 340 }, out), out);
  assert.equal(out.fieldX, 340);
  assert.equal(out.fieldW, 460);
});

/* ------------------------------------------------------------------ */
/* Particle state                                                     */
/* ------------------------------------------------------------------ */

test('initial state is deterministic for a seed and differs across seeds', () => {
  const frame = frameFor();
  const a = createParticles({ count: 216, seed: 7 });
  const b = createParticles({ count: 216, seed: 7 });
  const c = createParticles({ count: 216, seed: 8 });
  for (const system of [a, b, c]) snapParticles(system, frame);
  assert.deepEqual(a.x, b.x);
  assert.deepEqual(a.y, b.y);
  assert.deepEqual(a.depth, b.depth);
  assert.notDeepEqual(a.x, c.x);

  for (let i = 0; i < 120; i++) {
    stepParticles(a, DT, frame);
    stepParticles(b, DT, frame);
  }
  assert.deepEqual(a.x, b.x);
  assert.deepEqual(a.y, b.y);
});

test('idle particles drift but stay inside the free rectangle', () => {
  const frame = frameFor(900, 500, { left: 340 });
  const system = createParticles({ count: 216, seed: 3 });
  snapParticles(system, frame);
  const startX = Float64Array.from(system.x);
  for (let i = 0; i < 1200; i++) stepParticles(system, DT, frame);
  let moved = 0;
  for (let i = 0; i < system.count; i++) {
    assert.ok(system.x[i] >= 340 && system.x[i] <= 900, `x[${i}] = ${system.x[i]}`);
    assert.ok(system.y[i] >= 0 && system.y[i] <= 500, `y[${i}] = ${system.y[i]}`);
    if (Math.abs(system.x[i] - startX[i]) > 0.5) moved++;
  }
  assert.ok(moved > system.count / 2, `only ${moved} particles drifted`);
});

test('stepping converges every particle to its layout target', () => {
  const frame = frameFor();
  const system = createParticles({ count: 216, seed: 7 });
  snapParticles(system, frame);
  const positions = latticeLayout(system.count);
  setTargets(system, positions);

  stepParticles(system, DT, frame);
  const first = layoutToScreen(frame.viewport, positions[0], positions[1]);
  assert.ok(Math.hypot(system.x[0] - first.x, system.y[0] - first.y) > 1, 'eases rather than jumps');

  for (let i = 0; i < 240; i++) stepParticles(system, DT, frame);
  for (let i = 0; i < system.count; i++) {
    const p = layoutToScreen(frame.viewport, positions[2 * i], positions[2 * i + 1]);
    assert.ok(Math.abs(system.x[i] - p.x) < 0.01, `x[${i}]`);
    assert.ok(Math.abs(system.y[i] - p.y) < 0.01, `y[${i}]`);
    assert.ok(Math.abs(system.vx[i]) < 0.1 && Math.abs(system.vy[i]) < 0.1, `velocity[${i}]`);
  }
});

test('the spring never overshoots wildly, even at the largest clamped step', () => {
  const frame = frameFor();
  const system = createParticles({ count: 4, seed: 1 });
  setTargets(system, new Float64Array(8).fill(0.5));
  snapParticles(system, frame);
  setTargets(system, new Float64Array([0, 0, 1, 0, 0, 1, 1, 1]));
  const { x: vx0, y: vy0, size } = frame.viewport;
  for (let i = 0; i < 300; i++) {
    stepParticles(system, clampDt(10), frame);
    for (let p = 0; p < system.count; p++) {
      assert.ok(system.x[p] >= vx0 - 1 && system.x[p] <= vx0 + size + 1);
      assert.ok(system.y[p] >= vy0 - 1 && system.y[p] <= vy0 + size + 1);
    }
  }
  assert.ok(Math.abs(system.x[3] - (vx0 + size)) < 0.01);
});

test('snapParticles places particles on their targets immediately', () => {
  const frame = frameFor(1024, 768, { left: 340 });
  const system = createParticles({ count: 216, seed: 7 });
  const positions = latticeLayout(system.count);
  setTargets(system, positions);
  snapParticles(system, frame);
  for (let i = 0; i < system.count; i++) {
    const p = layoutToScreen(frame.viewport, positions[2 * i], positions[2 * i + 1]);
    assert.equal(system.x[i], p.x);
    assert.equal(system.y[i], p.y);
    assert.equal(system.vx[i], 0);
    assert.equal(system.vy[i], 0);
  }
});

test('setTargets copies its input and never mutates it', () => {
  const frame = frameFor();
  const system = createParticles({ count: 3, seed: 1 });
  const positions = new Float64Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
  const original = Float64Array.from(positions);
  setTargets(system, positions);
  assert.notEqual(system.targets, positions);
  assert.notEqual(system.targets.buffer, positions.buffer);

  positions.fill(0.99);
  snapParticles(system, frame);
  const p = layoutToScreen(frame.viewport, original[0], original[1]);
  assert.equal(system.x[0], p.x);
  assert.equal(system.y[0], p.y);

  const untouched = Float64Array.from(original);
  setTargets(system, untouched);
  for (let i = 0; i < 30; i++) stepParticles(system, DT, frame);
  assert.deepEqual(untouched, original);
});

test('setTargets rejects a wrong length, clamps stray values and null returns to idle', () => {
  const system = createParticles({ count: 2, seed: 1 });
  assert.throws(() => setTargets(system, new Float64Array(3)), RangeError);
  setTargets(system, new Float64Array([-0.5, 1.5, NaN, 0.25]));
  assert.deepEqual(Array.from(system.targets), [0, 1, 0, 0.25]);
  assert.equal(system.hasTargets, true);
  setTargets(system, null);
  assert.equal(system.hasTargets, false);
});

test('reduced motion removes idle drift', () => {
  const frame = frameFor();
  frame.reducedMotion = true;
  const system = createParticles({ count: 50, seed: 7 });
  snapParticles(system, frame);
  const startX = Float64Array.from(system.x);
  const startY = Float64Array.from(system.y);
  for (let i = 0; i < 300; i++) stepParticles(system, DT, frame);
  for (let i = 0; i < system.count; i++) {
    assert.ok(Math.abs(system.x[i] - startX[i]) < 1e-6);
    assert.ok(Math.abs(system.y[i] - startY[i]) < 1e-6);
  }
});

/* ------------------------------------------------------------------ */
/* Pointer force and hit testing                                      */
/* ------------------------------------------------------------------ */

/** Three particles on a horizontal line: near the pointer, further away, and out of range. */
function pointerScene() {
  const frame = frameFor(1000, 1000);
  const { x, y, size } = frame.viewport;
  const system = createParticles({ count: 3, seed: 1 });
  const px = x + size * 0.5;
  const py = y + size * 0.5;
  const toLayout = (offset) => 0.5 + offset / size;
  setTargets(system, new Float64Array([toLayout(20), 0.5, toLayout(70), 0.5, toLayout(POINTER_RADIUS + 60), 0.5]));
  snapParticles(system, frame);
  return { frame, system, px, py };
}

test('the pointer repels nearby particles, leaves distant ones alone, and they spring back', () => {
  const { frame, system, px, py } = pointerScene();
  const rest = Float64Array.from(system.x);

  frame.pointerActive = true;
  frame.pointerX = px;
  frame.pointerY = py;
  for (let i = 0; i < 180; i++) stepParticles(system, DT, frame);

  const nearShift = system.x[0] - rest[0];
  const midShift = system.x[1] - rest[1];
  assert.ok(nearShift > 1, `near particle moved ${nearShift}`);
  assert.ok(midShift > 0 && midShift < nearShift, 'displacement falls off with distance');
  assert.ok(nearShift <= POINTER_PUSH, 'displacement is bounded');
  assert.equal(system.x[2], rest[2]);
  assert.equal(system.y[2], py);
  assert.ok(Math.abs(system.y[0] - py) < 1e-9, 'pushed straight away from the pointer');

  frame.pointerActive = false;
  for (let i = 0; i < 240; i++) stepParticles(system, DT, frame);
  for (let i = 0; i < 3; i++) assert.ok(Math.abs(system.x[i] - rest[i]) < 0.01, `particle ${i} returned`);
});

test('the exempt (hovered) particle is not pushed, and reduced motion disables the push', () => {
  const { frame, system, px, py } = pointerScene();
  const rest = Float64Array.from(system.x);
  frame.pointerActive = true;
  frame.pointerX = px;
  frame.pointerY = py;
  frame.exempt = 0;
  computeGoals(system, frame);
  assert.equal(system.gx[0], rest[0]);
  assert.ok(system.gx[1] > rest[1]);

  frame.exempt = -1;
  frame.reducedMotion = true;
  computeGoals(system, frame);
  assert.equal(system.gx[0], rest[0]);
  assert.equal(system.gx[1], rest[1]);
});

test('hitTest returns the nearest particle within the radius, otherwise null', () => {
  const system = createParticles({ count: 3, seed: 1 });
  system.x.set([100, 110, 400]);
  system.y.set([100, 100, 400]);
  assert.equal(hitTest(system, 103, 100, 20), 0);
  assert.equal(hitTest(system, 108, 101, 20), 1);
  assert.equal(hitTest(system, 395, 404, 20), 2);
  assert.equal(hitTest(system, 250, 250, 20), null);
  assert.equal(hitTest(system, 100, 125, 20), null);
  assert.equal(hitTest(system, 100, 120, 20), 0, 'the radius is inclusive');
  assert.equal(hitTest(createParticles({ count: 0 }), 0, 0, 50), null);
});

/* ------------------------------------------------------------------ */
/* Dust and pulses                                                    */
/* ------------------------------------------------------------------ */

test('dust is seeded and stays wrapped inside the unit square', () => {
  const a = createDust({ count: 80, seed: 7 });
  const b = createDust({ count: 80, seed: 7 });
  assert.deepEqual(a.x, b.x);
  const start = Float64Array.from(a.x);
  for (let i = 0; i < 5000; i++) stepDust(a, MAX_DT, i * MAX_DT);
  assert.notDeepEqual(a.x, start);
  for (let i = 0; i < a.count; i++) {
    assert.ok(a.x[i] >= 0 && a.x[i] < 1, `x[${i}] = ${a.x[i]}`);
    assert.ok(a.y[i] >= 0 && a.y[i] < 1, `y[${i}] = ${a.y[i]}`);
  }
});

test('pulses saturate instead of accumulating, then decay to silence', () => {
  const state = createPulseState();
  for (let i = 0; i < 500; i++) {
    assert.equal(triggerPulse(state, 'accepted'), true);
    triggerPulse(state, 'rejected');
    triggerPulse(state, 'terminated');
    assert.ok(state.accepted <= 1 && state.rejected <= 1);
    stepPulses(state, 0.001);
  }
  assert.ok(Math.abs(state.terminatedAge - 0.5) < 1e-6, 'a ring in flight is not restarted');

  for (let i = 0; i < 600; i++) stepPulses(state, DT);
  assert.deepEqual(state, { accepted: 0, rejected: 0, terminatedAge: -1 });
  assert.ok(TERMINATED_DURATION < 10);
  assert.equal(triggerPulse(state, 'nonsense'), false);
  assert.deepEqual(state, { accepted: 0, rejected: 0, terminatedAge: -1 });
});

/* ------------------------------------------------------------------ */
/* Renderer wiring, against a minimal fake DOM                        */
/* ------------------------------------------------------------------ */

function eventTarget() {
  const listeners = new Map();
  return {
    listeners,
    addEventListener(type, fn) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type).add(fn);
    },
    removeEventListener(type, fn) {
      listeners.get(type)?.delete(fn);
    },
    emit(type, event = {}) {
      for (const fn of [...(listeners.get(type) ?? [])]) fn(event);
    },
    listenerCount() {
      let total = 0;
      for (const set of listeners.values()) total += set.size;
      return total;
    },
  };
}

/** A 2D context that accepts every call, counts painted paths and records drawn text. */
function fakeContext(stats) {
  const store = {};
  return new Proxy(store, {
    get(target, prop) {
      if (prop in target) return target[prop];
      if (prop === 'fill' || prop === 'stroke') return () => void stats.paints++;
      if (prop === 'fillText') return (text) => void stats.texts.push(text);
      return () => {};
    },
    set(target, prop, value) {
      target[prop] = value;
      return true;
    },
  });
}

function fakeDom({ width = 800, height = 600, reducedMotion = false } = {}) {
  const stats = { paints: 0, observers: 0, texts: [] };
  const frames = new Map();
  let nextId = 1;
  const media = Object.assign(eventTarget(), { matches: reducedMotion });
  const win = Object.assign(eventTarget(), {
    devicePixelRatio: 3,
    requestAnimationFrame(fn) {
      frames.set(nextId, fn);
      return nextId++;
    },
    cancelAnimationFrame(id) {
      frames.delete(id);
    },
    matchMedia: () => media,
    ResizeObserver: class {
      observe() {
        stats.observers++;
      }
      disconnect() {
        stats.observers--;
      }
    },
  });
  const makeCanvas = () =>
    Object.assign(eventTarget(), { width: 300, height: 150, style: { touchAction: '' }, getContext: () => fakeContext(stats) });
  const doc = Object.assign(eventTarget(), { hidden: false, defaultView: win, createElement: makeCanvas });
  const canvas = Object.assign(makeCanvas(), { ownerDocument: doc, clientWidth: width, clientHeight: height });
  let now = 0;
  /** Runs every pending animation frame once, 16 ms later. */
  const flush = () => {
    now += 16;
    const pending = [...frames.entries()];
    frames.clear();
    for (const [, fn] of pending) fn(now);
  };
  return { canvas, doc, win, media, frames, stats, flush };
}

test('renderer exposes exactly the contract methods and paints before start()', () => {
  const dom = fakeDom();
  const renderer = createRenderer({ canvas: dom.canvas, dataset: createDataset() });
  assert.deepEqual(Object.keys(renderer).sort(), [
    'destroy',
    'pulse',
    'resetView',
    'resize',
    'select',
    'setLayout',
    'showCandidate',
    'start',
    'stop',
    'zoomBy',
  ]);
  assert.equal(dom.canvas.width, 1600, 'device pixel ratio is capped at 2');
  assert.equal(dom.canvas.height, 1200);
  assert.ok(dom.stats.paints >= 216, 'first frame is painted on creation');
  assert.equal(dom.frames.size, 0, 'no loop until start()');
  renderer.destroy();
});

test('start and stop are idempotent and the loop pauses while the document is hidden', () => {
  const dom = fakeDom();
  const renderer = createRenderer({ canvas: dom.canvas, dataset: createDataset() });
  renderer.start();
  renderer.start();
  assert.equal(dom.frames.size, 1);
  for (let i = 0; i < 5; i++) dom.flush();
  assert.equal(dom.frames.size, 1);

  dom.doc.hidden = true;
  dom.doc.emit('visibilitychange');
  assert.equal(dom.frames.size, 0);
  dom.doc.hidden = false;
  dom.doc.emit('visibilitychange');
  assert.equal(dom.frames.size, 1);

  renderer.stop();
  renderer.stop();
  assert.equal(dom.frames.size, 0);
  dom.doc.emit('visibilitychange');
  assert.equal(dom.frames.size, 0, 'becoming visible does not restart a stopped renderer');
  renderer.start();
  assert.equal(dom.frames.size, 1);
  renderer.destroy();
});

test('renderer runs layouts, pulses and resizes without mutating the caller layout', () => {
  const dom = fakeDom();
  const dataset = createDataset();
  let insetCalls = 0;
  const renderer = createRenderer({
    canvas: dom.canvas,
    dataset,
    getInsets: () => {
      insetCalls++;
      return { left: 340 };
    },
  });
  assert.equal(insetCalls, 1);
  const positions = latticeLayout(dataset.points.length);
  const original = Float64Array.from(positions);

  renderer.start();
  renderer.setLayout(positions);
  for (let i = 0; i < 30; i++) {
    renderer.pulse(['accepted', 'rejected', 'terminated'][i % 3]);
    dom.flush();
  }
  renderer.setLayout(positions, { immediate: true });
  renderer.setLayout(null);
  dom.canvas.clientWidth = 320;
  dom.canvas.clientHeight = 568;
  renderer.resize();
  assert.equal(insetCalls, 2);
  assert.equal(dom.canvas.width, 640);
  dom.flush();
  assert.deepEqual(positions, original);
  assert.throws(() => renderer.setLayout(new Float64Array(4)), RangeError);
  renderer.destroy();
});

test('onHover fires only when the hovered point changes, and leaving clears it', () => {
  const dom = fakeDom();
  const dataset = createDataset();
  const count = dataset.points.length;
  const neighbours = dataset.points.map((_, i) => Int32Array.of((i + 1) % count, (i + 2) % count));
  const hovers = [];
  const renderer = createRenderer({ canvas: dom.canvas, dataset, neighbours, onHover: (index) => hovers.push(index) });

  // Everything in the bottom-right corner except point 5, alone at the centre.
  const positions = new Float64Array(count * 2).fill(1);
  positions[10] = 0.5;
  positions[11] = 0.5;
  renderer.setLayout(positions, { immediate: true });
  renderer.start();

  const centre = { offsetX: 400, offsetY: 300, pointerType: 'mouse' };
  dom.canvas.emit('pointermove', centre);
  dom.canvas.emit('pointermove', { ...centre, offsetX: 402 });
  for (let i = 0; i < 10; i++) dom.flush();
  assert.deepEqual(hovers, [5]);

  dom.canvas.emit('pointermove', { offsetX: 200, offsetY: 100, pointerType: 'mouse' });
  assert.deepEqual(hovers, [5, null]);
  dom.canvas.emit('pointermove', centre);
  dom.canvas.emit('pointerleave', { pointerType: 'mouse' });
  dom.canvas.emit('pointerleave', { pointerType: 'mouse' });
  for (let i = 0; i < 3; i++) dom.flush();
  assert.deepEqual(hovers, [5, null, 5, null]);

  // A touch keeps its selection after the finger lifts.
  dom.canvas.emit('pointerdown', { ...centre, pointerType: 'touch' });
  dom.canvas.emit('pointerleave', { pointerType: 'touch' });
  assert.deepEqual(hovers, [5, null, 5, null, 5]);
  renderer.destroy();
});

test('reduced motion snaps layouts and ignores pulses', () => {
  const dom = fakeDom({ reducedMotion: true });
  const dataset = createDataset();
  const hovers = [];
  const renderer = createRenderer({ canvas: dom.canvas, dataset, onHover: (index) => hovers.push(index) });
  const positions = new Float64Array(dataset.points.length * 2).fill(1);
  positions[0] = 0.5;
  positions[1] = 0.5;
  renderer.setLayout(positions);
  renderer.pulse('accepted');
  // No frame has run, yet point 0 is already at the centre of the canvas.
  dom.canvas.emit('pointermove', { offsetX: 400, offsetY: 300, pointerType: 'mouse' });
  assert.deepEqual(hovers, [0]);
  renderer.destroy();
});

test('destroy stops the loop and removes every listener and observer', () => {
  const dom = fakeDom();
  const renderer = createRenderer({ canvas: dom.canvas, dataset: createDataset() });
  assert.ok(dom.canvas.listenerCount() >= 3);
  assert.equal(dom.stats.observers, 1);
  assert.equal(dom.canvas.style.touchAction, 'none');
  renderer.start();
  renderer.destroy();
  renderer.destroy();
  assert.equal(dom.frames.size, 0);
  assert.equal(dom.canvas.listenerCount(), 0);
  assert.equal(dom.doc.listenerCount(), 0);
  assert.equal(dom.win.listenerCount(), 0);
  assert.equal(dom.media.listenerCount(), 0);
  assert.equal(dom.stats.observers, 0);
  assert.equal(dom.canvas.style.touchAction, '');
  renderer.start();
  assert.equal(dom.frames.size, 0, 'a destroyed renderer cannot be restarted');
});

test('zooming moves the view, keeps hover accurate, labels particles and reports the zoom level', () => {
  const dom = fakeDom();
  const dataset = createDataset();
  const count = dataset.points.length;
  const hovers = [];
  const zooms = [];
  const renderer = createRenderer({
    canvas: dom.canvas,
    dataset,
    onHover: (index) => hovers.push(index),
    onViewChange: (zoom) => zooms.push(zoom),
  });
  const positions = new Float64Array(count * 2).fill(1);
  positions[10] = 0.5;
  positions[11] = 0.5;
  renderer.setLayout(positions, { immediate: true });
  renderer.start();

  // Zoom 2x about (100, 100): the point at the centre (400, 300) moves to (700, 500).
  let prevented = false;
  dom.canvas.emit('wheel', {
    offsetX: 100,
    offsetY: 100,
    deltaY: -Math.log(2) / 0.0016,
    preventDefault: () => (prevented = true),
  });
  for (let i = 0; i < 60; i++) dom.flush();
  assert.ok(prevented, 'the page must not scroll while zooming');
  assert.equal(zooms.length, 1);
  assert.ok(Math.abs(zooms[0] - 2) < 1e-9);

  dom.canvas.emit('pointermove', { offsetX: 400, offsetY: 300, pointerType: 'mouse' });
  assert.deepEqual(hovers, []);
  dom.canvas.emit('pointermove', { offsetX: 700, offsetY: 500, pointerType: 'mouse' });
  assert.deepEqual(hovers, [5]);
  for (let i = 0; i < 10; i++) dom.flush();
  assert.ok(dom.stats.texts.includes(dataset.points[5].label.toUpperCase()), 'the hovered particle gets a tooltip with its word');

  // Zoomed in far enough, every visible particle is labelled with its word.
  dom.stats.texts.length = 0;
  renderer.zoomBy(2);
  for (let i = 0; i < 60; i++) dom.flush();
  assert.ok(new Set(dom.stats.texts).size > 1, 'labels are drawn when zoomed in');
  renderer.zoomBy(0.5);
  for (let i = 0; i < 60; i++) dom.flush();

  // Dragging while zoomed pans the view instead of hovering.
  dom.canvas.emit('pointerdown', { pointerId: 1, offsetX: 700, offsetY: 500, pointerType: 'mouse' });
  dom.canvas.emit('pointermove', { pointerId: 1, offsetX: 650, offsetY: 450, pointerType: 'mouse' });
  dom.canvas.emit('pointerup', { pointerId: 1, offsetX: 650, offsetY: 450, pointerType: 'mouse' });
  dom.canvas.emit('pointermove', { pointerId: 1, offsetX: 650, offsetY: 450, pointerType: 'mouse' });
  assert.equal(hovers.at(-1), 5, 'the point followed the drag');

  renderer.zoomBy(2);
  renderer.resetView();
  assert.deepEqual(zooms.map((z) => Math.round(z)), [2, 4, 2, 4, 1]);
  renderer.destroy();
  assert.equal(dom.canvas.listenerCount(), 0);
});

test('every topic has its own shape, on canvas and in the legend', async () => {
  const { SHAPES, isStroked, shapeOf, shapeSvg, traceShape } = await import('../src/render/shapes.js');
  const dataset = createDataset();
  const used = dataset.clusterNames.map((_, cluster) => shapeOf(cluster));
  assert.equal(new Set(used).size, dataset.clusterNames.length, 'no two topics share a shape');
  assert.equal(shapeOf(SHAPES.length), SHAPES[0], 'indices wrap');

  for (const shape of SHAPES) {
    const calls = [];
    const ctx = new Proxy({}, { get: (_, method) => (...args) => void calls.push([method, ...args]) });
    traceShape(ctx, shape, 100, 50, 4);
    assert.ok(calls.length > 0, `${shape} adds to the path`);
    for (const [, ...args] of calls) {
      for (const value of args) assert.ok(Number.isFinite(value), `${shape} uses finite coordinates`);
    }
    // Everything stays within a small box around the centre.
    for (const [method, px, py] of calls) {
      if (method !== 'moveTo' && method !== 'lineTo') continue;
      assert.ok(Math.abs(px - 100) <= 6 && Math.abs(py - 50) <= 6, `${shape} stays near its centre`);
    }
    const markup = shapeSvg(shape, 'rgb(1 2 3)');
    assert.ok(markup.startsWith('<svg') && markup.endsWith('</svg>') && markup.includes('aria-hidden="true"'));
    assert.ok(markup.includes(isStroked(shape) ? 'stroke="rgb(1 2 3)"' : 'fill="rgb(1 2 3)"'));
  }
});

/** A renderer over the real dataset with point 5 at the centre and point 9 at the left, everything else in a corner. */
function selectionFixture(extra = {}) {
  const dom = fakeDom();
  const dataset = createDataset();
  const count = dataset.points.length;
  const neighbours = Array.from({ length: count }, (_, i) => Int32Array.from({ length: 8 }, (_, k) => (i + k + 1) % count));
  const selections = [];
  const renderer = createRenderer({
    canvas: dom.canvas,
    dataset,
    neighbours,
    onSelect: (index, list) => selections.push([index, list]),
    ...extra,
  });
  const positions = new Float64Array(count * 2).fill(1);
  positions[10] = 0.5;
  positions[11] = 0.5;
  positions[18] = 0.25;
  positions[19] = 0.5;
  renderer.setLayout(positions, { immediate: true });
  renderer.start();
  const view = computeViewport(800, 600);
  const at = (index) => {
    const p = layoutToScreen(view, positions[2 * index], positions[2 * index + 1]);
    return { offsetX: p.x, offsetY: p.y, pointerType: 'mouse', pointerId: 1 };
  };
  const click = (event) => {
    dom.canvas.emit('pointerdown', event);
    dom.canvas.emit('pointerup', event);
  };
  return { dom, dataset, neighbours, renderer, selections, at, click, positions };
}

test('clicking a mark selects it and reveals exactly its five nearest neighbours, in order', () => {
  const { dom, neighbours, renderer, selections, at, click } = selectionFixture();
  click(at(5));
  assert.equal(selections.length, 1);
  assert.equal(selections[0][0], 5);
  assert.deepEqual(selections[0][1], Array.from(neighbours[5]).slice(0, 5));
  for (let i = 0; i < 5; i++) dom.flush();
  renderer.destroy();
});

test('clicking another mark moves the selection; clicking it again or empty space clears it', () => {
  const { neighbours, renderer, selections, at, click } = selectionFixture();
  click(at(5));
  click(at(9));
  assert.equal(selections.at(-1)[0], 9);
  assert.deepEqual(selections.at(-1)[1], Array.from(neighbours[9]).slice(0, 5));
  click(at(9));
  assert.deepEqual(selections.at(-1), [null, []]);
  click(at(5));
  click({ offsetX: 400, offsetY: 80, pointerType: 'mouse', pointerId: 1 });
  assert.deepEqual(selections.at(-1), [null, []]);
  assert.deepEqual(selections.map(([index]) => index), [5, 9, null, 5, null]);
  renderer.destroy();
});

test('a drag is not a click, and Escape clears the selection', () => {
  const { dom, renderer, selections, at, click } = selectionFixture();
  const start = at(5);
  dom.canvas.emit('pointerdown', start);
  dom.canvas.emit('pointermove', { ...start, offsetX: start.offsetX + 30 });
  dom.canvas.emit('pointerup', { ...start, offsetX: start.offsetX + 30 });
  assert.deepEqual(selections, [], 'dragging selects nothing');

  click(at(5));
  dom.win.emit('keydown', { key: 'a' });
  assert.equal(selections.length, 1);
  dom.win.emit('keydown', { key: 'Escape' });
  assert.deepEqual(selections.at(-1), [null, []]);
  dom.win.emit('keydown', { key: 'Escape' });
  assert.equal(selections.length, 2, 'Escape with nothing selected does nothing');
  renderer.destroy();
  assert.equal(dom.win.listenerCount(), 0);
});

test('select() honours selectionSize and ignores indices that do not exist', () => {
  const { renderer, selections } = selectionFixture({ selectionSize: 3 });
  renderer.select(7);
  assert.equal(selections.at(-1)[1].length, 3);
  renderer.select(99999);
  assert.deepEqual(selections.at(-1), [null, []]);
  renderer.select(null);
  assert.equal(selections.length, 2, 'clearing an empty selection reports nothing');
  renderer.destroy();
});

test('the tooltip shows the caller-supplied similarity only for a revealed neighbour of the selection', () => {
  const calls = [];
  const { dom, neighbours, renderer, at, click, positions } = selectionFixture({
    similarity: (i, j) => {
      calls.push([i, j]);
      return 0.87654;
    },
  });
  const neighbour = neighbours[5][0];
  const unrelated = neighbours[5][7];
  // Put one revealed neighbour and one unrelated point where they can be hovered.
  positions[2 * neighbour] = 0.75;
  positions[2 * neighbour + 1] = 0.5;
  positions[2 * unrelated] = 0.5;
  positions[2 * unrelated + 1] = 0.25;
  renderer.setLayout(positions, { immediate: true });
  const view = computeViewport(800, 600);
  const hover = (index) => {
    const p = layoutToScreen(view, positions[2 * index], positions[2 * index + 1]);
    dom.canvas.emit('pointermove', { offsetX: p.x, offsetY: p.y, pointerType: 'mouse' });
    dom.stats.texts.length = 0;
    for (let i = 0; i < 12; i++) dom.flush();
    return [...dom.stats.texts];
  };

  assert.ok(!hover(neighbour).some((text) => text.startsWith('cos')), 'no similarity without a selection');
  click(at(5));
  const texts = hover(neighbour);
  assert.ok(texts.includes('cos 0.877'), texts.join('|'));
  assert.deepEqual(calls.at(-1), [5, neighbour], 'similarity is asked for the selected point and the hovered one');
  assert.ok(!hover(unrelated).some((text) => text.startsWith('cos')), 'unrelated points show no similarity');
  renderer.destroy();
});

test('showCandidate overlays a proposal without moving the layout, and tolerates bad input', () => {
  const { dom, renderer, selections, at, click, positions } = selectionFixture();
  const candidate = Float64Array.from(positions, (v) => 1 - v);
  renderer.showCandidate(candidate);
  candidate.fill(0.3);
  for (let i = 0; i < 20; i++) dom.flush();
  // Point 5 is still where the accepted layout put it, so a click there still finds it.
  click(at(5));
  assert.equal(selections.at(-1)[0], 5);
  renderer.pulse('rejected');
  for (let i = 0; i < 40; i++) dom.flush();
  renderer.showCandidate(new Float64Array(3));
  renderer.showCandidate(null);
  renderer.pulse('accepted');
  for (let i = 0; i < 5; i++) dom.flush();
  renderer.destroy();
});
