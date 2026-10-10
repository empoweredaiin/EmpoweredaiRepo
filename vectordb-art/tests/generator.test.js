import assert from 'node:assert/strict';
import { test } from 'node:test';

import { NEIGHBOUR_K, POINT_RADIUS, candidateIssues } from '../src/contracts.js';
import { layoutNeighbours, semanticNeighbours } from '../src/core/knn.js';
import { createDataset } from '../src/data/dataset.js';
import { GENERATOR_DEFAULTS, createGenerator } from '../src/generator/generator.js';

const dataset = createDataset();
const N = dataset.points.length;
const SEED = 1234;

/* ------------------------------------------------------------------ */
/* Helpers                                                            */
/* ------------------------------------------------------------------ */

/** Fraction of each point's semantic neighbours that are also its 2D neighbours. */
function neighbourhoodScore(positions) {
  const semantic = semanticNeighbours(dataset, NEIGHBOUR_K);
  const layout = layoutNeighbours(positions, NEIGHBOUR_K);
  let shared = 0;
  for (let i = 0; i < N; i++) {
    const wanted = new Set(semantic[i]);
    for (const j of layout[i]) if (wanted.has(j)) shared++;
  }
  return shared / (N * NEIGHBOUR_K);
}

/** Mean silhouette of the labelled clusters in 2D, mapped from [-1, 1] to [0, 1]. */
function separationScore(positions) {
  const clusters = dataset.clusterNames.length;
  let total = 0;
  for (let i = 0; i < N; i++) {
    const sums = new Array(clusters).fill(0);
    const counts = new Array(clusters).fill(0);
    for (let j = 0; j < N; j++) {
      if (j === i) continue;
      const c = dataset.points[j].cluster;
      sums[c] += Math.hypot(positions[2 * i] - positions[2 * j], positions[2 * i + 1] - positions[2 * j + 1]);
      counts[c]++;
    }
    const own = dataset.points[i].cluster;
    const a = sums[own] / counts[own];
    let b = Infinity;
    for (let c = 0; c < clusters; c++) if (c !== own) b = Math.min(b, sums[c] / counts[c]);
    total += (b - a) / Math.max(a, b);
  }
  return (total / N + 1) / 2;
}

/** 1 minus the fraction of points that overlap at least one other point. */
function overlapScore(positions) {
  const overlapping = new Set();
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      const d = Math.hypot(positions[2 * i] - positions[2 * j], positions[2 * i + 1] - positions[2 * j + 1]);
      if (d < 2 * POINT_RADIUS) overlapping.add(i).add(j);
    }
  }
  return 1 - overlapping.size / N;
}

/** 1 when nothing moved, 0 when points moved 0.05 on average. */
function stabilityScore(positions, previous) {
  if (!previous) return 1;
  let moved = 0;
  for (let i = 0; i < N; i++) {
    moved += Math.hypot(positions[2 * i] - previous[2 * i], positions[2 * i + 1] - previous[2 * i + 1]);
  }
  return Math.max(0, 1 - moved / N / 0.05);
}

/** A stand-in for the real evaluator, with the same criteria and weights. */
function evaluate(candidate, previous) {
  const criteria = {
    neighbourhood: neighbourhoodScore(candidate.positions),
    separation: separationScore(candidate.positions),
    overlap: overlapScore(candidate.positions),
    stability: stabilityScore(candidate.positions, previous?.positions ?? null),
  };
  const overall =
    0.4 * criteria.neighbourhood + 0.3 * criteria.separation + 0.2 * criteria.overlap + 0.1 * criteria.stability;
  return { candidateId: candidate.id, kind: 'heuristic', criteria, overall, valid: true, issues: [] };
}

/** Runs the accept-if-strictly-better loop and returns the best score after every iteration. */
function greedyLoop(generator, iterations) {
  let best = null;
  const trajectory = [];
  for (let iteration = 1; iteration <= iterations; iteration++) {
    const candidate = generator.generate({
      iteration,
      parent: best?.candidate ?? null,
      feedback: best?.evaluation ?? null,
    });
    const evaluation = evaluate(candidate, best?.candidate ?? null);
    if (best === null || evaluation.overall > best.evaluation.overall) best = { candidate, evaluation };
    trajectory.push(best.evaluation.overall);
  }
  return { best, trajectory };
}

/** Always-accept chain: every candidate becomes the next parent. */
function chain(generator, iterations) {
  const layouts = [];
  let parent = null;
  for (let iteration = 1; iteration <= iterations; iteration++) {
    parent = generator.generate({ iteration, parent, feedback: null });
    layouts.push(parent);
  }
  return layouts;
}

const feedbackFixture = () => ({
  candidateId: `${SEED}:1`,
  kind: 'heuristic',
  criteria: { neighbourhood: 0.2, separation: 0.6, overlap: 0.4, stability: 1 },
  weights: { neighbourhood: 0.4, separation: 0.3, overlap: 0.2, stability: 0.1 },
  overall: 0.44,
  valid: true,
  issues: [],
});

/* ------------------------------------------------------------------ */
/* Shape                                                              */
/* ------------------------------------------------------------------ */

test('exposes the Generator contract and documented defaults', () => {
  const generator = createGenerator({ dataset, seed: SEED });
  assert.equal(generator.seed, SEED);
  assert.equal(typeof generator.generate, 'function');
  assert.ok(Object.isFrozen(GENERATOR_DEFAULTS));
  assert.equal(GENERATOR_DEFAULTS.neighbours, NEIGHBOUR_K);
});

test('first layout has correct fields and passes candidateIssues', () => {
  const first = createGenerator({ dataset, seed: SEED }).generate({ iteration: 1, parent: null, feedback: null });
  assert.deepEqual(candidateIssues(first, N), []);
  assert.equal(first.id, `${SEED}:1`);
  assert.equal(first.seed, SEED);
  assert.equal(first.iteration, 1);
  assert.equal(first.parentId, null);
  assert.ok(first.positions instanceof Float64Array);
  assert.equal(first.positions.length, 2 * N);
  assert.equal(first.meta.strategy, 'project');
  assert.equal(typeof first.meta.stepSize, 'number');
});

test('refined layout records its parent, iteration and strategy', () => {
  const generator = createGenerator({ dataset, seed: SEED });
  const first = generator.generate({ iteration: 1, parent: null, feedback: null });
  const second = generator.generate({ iteration: 2, parent: first, feedback: feedbackFixture() });
  assert.equal(second.id, `${SEED}:2`);
  assert.equal(second.iteration, 2);
  assert.equal(second.parentId, first.id);
  assert.notEqual(second.positions, first.positions);
  assert.ok(['refine', 'relax', 'shake'].includes(second.meta.strategy));
  assert.ok(Number.isFinite(second.meta.stepSize) && second.meta.stepSize > 0);
});

test('a chain of 100 refinements stays valid, finite and inside the unit square', () => {
  const layouts = chain(createGenerator({ dataset, seed: SEED }), 100);
  for (const layout of layouts) {
    assert.deepEqual(candidateIssues(layout, N), []);
    for (const value of layout.positions) {
      assert.ok(Number.isFinite(value) && value >= 0 && value <= 1, `out of range: ${value}`);
    }
  }
});

test('parents on the very edge of the square, or fully coincident, still give valid layouts', () => {
  const generator = createGenerator({ dataset, seed: SEED });
  for (const fill of [0, 1, 0.5]) {
    const parent = { id: 'edge', seed: SEED, iteration: 1, parentId: null, positions: new Float64Array(2 * N).fill(fill) };
    const child = generator.generate({ iteration: 2, parent, feedback: null });
    assert.deepEqual(candidateIssues(child, N), []);
  }
});

test('every strategy appears over a run and step size anneals', () => {
  const generator = createGenerator({ dataset, seed: SEED });
  const first = generator.generate({ iteration: 1, parent: null, feedback: null });
  const seen = new Map();
  for (let iteration = 2; iteration <= 120; iteration++) {
    const { meta } = generator.generate({ iteration, parent: first, feedback: null });
    if (!seen.has(meta.strategy)) seen.set(meta.strategy, []);
    seen.get(meta.strategy).push(meta.stepSize);
  }
  assert.deepEqual([...seen.keys()].sort(), ['refine', 'relax', 'shake']);
  const refineSteps = seen.get('refine');
  for (let n = 1; n < refineSteps.length; n++) assert.ok(refineSteps[n] < refineSteps[n - 1]);
  assert.ok(refineSteps.at(-1) >= GENERATOR_DEFAULTS.stepMin);
});

/* ------------------------------------------------------------------ */
/* Determinism                                                        */
/* ------------------------------------------------------------------ */

test('same seed and input give identical positions, also across generators', () => {
  const a = createGenerator({ dataset, seed: SEED });
  const b = createGenerator({ dataset, seed: SEED });
  const firstA = a.generate({ iteration: 1, parent: null, feedback: null });
  const firstB = b.generate({ iteration: 1, parent: null, feedback: null });
  assert.deepEqual(firstA.positions, firstB.positions);
  assert.deepEqual(a.generate({ iteration: 1, parent: null, feedback: null }).positions, firstA.positions);

  const input = { iteration: 5, parent: firstA, feedback: feedbackFixture() };
  const once = a.generate(input);
  assert.deepEqual(a.generate(input).positions, once.positions);
  assert.deepEqual(b.generate(input).positions, once.positions);
  assert.deepEqual(b.generate(input).meta, once.meta);
});

test('the result does not depend on call order or call history', () => {
  const a = createGenerator({ dataset, seed: SEED });
  const b = createGenerator({ dataset, seed: SEED });
  const parent = a.generate({ iteration: 1, parent: null, feedback: null });
  const iterations = [2, 3, 4, 5, 6, 7];
  const forward = new Map(iterations.map((i) => [i, a.generate({ iteration: i, parent, feedback: null })]));
  for (const i of [...iterations].reverse()) {
    assert.deepEqual(b.generate({ iteration: i, parent, feedback: null }).positions, forward.get(i).positions);
  }
  assert.deepEqual(a.generate({ iteration: 5, parent, feedback: null }).positions, forward.get(5).positions);
});

test('two independently built datasets with the same seed give the same layouts', () => {
  const other = createGenerator({ dataset: createDataset(), seed: SEED });
  const mine = createGenerator({ dataset, seed: SEED });
  assert.deepEqual(chain(other, 5).at(-1).positions, chain(mine, 5).at(-1).positions);
});

test('different seeds give different layouts', () => {
  const a = createGenerator({ dataset, seed: 1 });
  const b = createGenerator({ dataset, seed: 2 });
  const firstA = a.generate({ iteration: 1, parent: null, feedback: null });
  const firstB = b.generate({ iteration: 1, parent: null, feedback: null });
  assert.notDeepEqual(firstA.positions, firstB.positions);
  assert.notDeepEqual(
    a.generate({ iteration: 2, parent: firstA, feedback: null }).positions,
    b.generate({ iteration: 2, parent: firstA, feedback: null }).positions,
  );
});

test('different iterations with the same parent give different layouts', () => {
  const generator = createGenerator({ dataset, seed: SEED });
  const parent = generator.generate({ iteration: 1, parent: null, feedback: null });
  const seen = new Set();
  for (let iteration = 2; iteration <= 12; iteration++) {
    seen.add(generator.generate({ iteration, parent, feedback: null }).positions.join(','));
  }
  assert.equal(seen.size, 11);
});

test('feedback changes the proposal', () => {
  const generator = createGenerator({ dataset, seed: SEED });
  const parent = generator.generate({ iteration: 1, parent: null, feedback: null });
  const neutral = generator.generate({ iteration: 2, parent, feedback: null });
  const biased = generator.generate({ iteration: 2, parent, feedback: feedbackFixture() });
  assert.notDeepEqual(biased.positions, neutral.positions);
});

/* ------------------------------------------------------------------ */
/* Purity                                                             */
/* ------------------------------------------------------------------ */

test('parent, feedback and dataset are not mutated', () => {
  const vectorsBefore = dataset.points.map((p) => [...p.vector]);
  const generator = createGenerator({ dataset, seed: SEED });
  const parent = generator.generate({ iteration: 1, parent: null, feedback: null });
  const positionsBefore = Float64Array.from(parent.positions);
  const parentBefore = structuredClone({ ...parent, positions: null });
  const feedback = feedbackFixture();
  const feedbackBefore = structuredClone(feedback);

  for (let iteration = 2; iteration <= 30; iteration++) generator.generate({ iteration, parent, feedback });

  assert.deepEqual(parent.positions, positionsBefore);
  assert.deepEqual({ ...parent, positions: null }, parentBefore);
  assert.deepEqual(feedback, feedbackBefore);
  assert.ok(Object.isFrozen(dataset) && Object.isFrozen(dataset.points));
  assert.ok(dataset.points.every((p) => Object.isFrozen(p) && Object.isFrozen(p.vector)));
  assert.deepEqual(
    dataset.points.map((p) => [...p.vector]),
    vectorsBefore,
  );
});

test('frozen inputs are accepted', () => {
  const generator = createGenerator({ dataset, seed: SEED });
  const parent = Object.freeze(generator.generate({ iteration: 1, parent: null, feedback: null }));
  const feedback = Object.freeze({ ...feedbackFixture(), criteria: Object.freeze(feedbackFixture().criteria) });
  assert.doesNotThrow(() => generator.generate(Object.freeze({ iteration: 2, parent, feedback })));
});

/* ------------------------------------------------------------------ */
/* Quality                                                            */
/* ------------------------------------------------------------------ */

test('initial layout preserves semantic neighbourhoods well above chance', () => {
  const chance = NEIGHBOUR_K / (N - 1);
  for (const seed of [1, 2, 3, 42, SEED]) {
    const first = createGenerator({ dataset, seed }).generate({ iteration: 1, parent: null, feedback: null });
    const score = neighbourhoodScore(first.positions);
    assert.ok(score > 2.5 * chance, `seed ${seed}: ${score.toFixed(3)} vs chance ${chance.toFixed(3)}`);
  }
});

test('refinement uses the vectors, not the labels: relabelled clusters give the same layout', () => {
  const relabelled = Object.freeze({
    ...dataset,
    points: Object.freeze(dataset.points.map((p) => Object.freeze({ ...p, cluster: 0 }))),
  });
  const a = chain(createGenerator({ dataset, seed: SEED }), 6).at(-1);
  const b = chain(createGenerator({ dataset: relabelled, seed: SEED }), 6).at(-1);
  assert.deepEqual(a.positions, b.positions);
});

test('a greedy accept-if-better loop climbs over 100 iterations', () => {
  for (const seed of [1, 42]) {
    const { best, trajectory } = greedyLoop(createGenerator({ dataset, seed }), 100);
    const at = (iteration) => trajectory[iteration - 1];
    assert.ok(at(25) > at(1) + 0.15, `seed ${seed}: ${at(1).toFixed(3)} -> ${at(25).toFixed(3)} after 25`);
    assert.ok(at(100) > at(25), `seed ${seed}: no gain after iteration 25`);
    assert.ok(at(100) > 0.64, `seed ${seed}: final score ${at(100).toFixed(3)}`);
    const { criteria } = best.evaluation;
    const start = evaluate(createGenerator({ dataset, seed }).generate({ iteration: 1, parent: null, feedback: null }), null);
    assert.ok(criteria.neighbourhood > start.criteria.neighbourhood + 0.1);
    assert.ok(criteria.separation > start.criteria.separation + 0.15);
    assert.ok(criteria.overlap > 0.95);
  }
});

test('late refinements are small moves', () => {
  const layouts = chain(createGenerator({ dataset, seed: SEED }), 80);
  assert.ok(stabilityScore(layouts[79].positions, layouts[78].positions) > 0.8);
});

test('one generate call is fast', () => {
  const generator = createGenerator({ dataset, seed: SEED });
  let parent = generator.generate({ iteration: 1, parent: null, feedback: null });
  for (let iteration = 2; iteration <= 20; iteration++) parent = generator.generate({ iteration, parent, feedback: null });
  const started = process.hrtime.bigint();
  const calls = 50;
  for (let iteration = 21; iteration < 21 + calls; iteration++) {
    parent = generator.generate({ iteration, parent, feedback: null });
  }
  const msPerCall = Number(process.hrtime.bigint() - started) / 1e6 / calls;
  // The budget is "well under 5 ms"; the bound here is loose so a busy machine does not fail the suite.
  assert.ok(msPerCall < 15, `${msPerCall.toFixed(2)} ms per call`);
});

/* ------------------------------------------------------------------ */
/* Argument validation                                                */
/* ------------------------------------------------------------------ */

test('createGenerator rejects bad arguments with clear errors', () => {
  assert.throws(() => createGenerator(), { name: 'TypeError', message: /dataset/ });
  assert.throws(() => createGenerator({ seed: 1 }), { name: 'TypeError', message: /dataset/ });
  assert.throws(() => createGenerator({ dataset: { points: 'nope' }, seed: 1 }), { name: 'TypeError', message: /dataset/ });
  assert.throws(() => createGenerator({ dataset }), { name: 'TypeError', message: /seed/ });
  assert.throws(() => createGenerator({ dataset, seed: 1.5 }), { name: 'TypeError', message: /seed/ });
  assert.throws(() => createGenerator({ dataset, seed: '7' }), { name: 'TypeError', message: /seed/ });
  assert.throws(() => createGenerator({ dataset, seed: NaN }), { name: 'TypeError', message: /seed/ });
  assert.throws(() => createGenerator({ dataset, seed: 1, options: { nope: 1 } }), { name: 'TypeError', message: /nope/ });
  assert.throws(() => createGenerator({ dataset, seed: 1, options: { steps: 'x' } }), { name: 'TypeError', message: /steps/ });
  assert.throws(() => createGenerator({ dataset, seed: 1, options: { steps: 0 } }), { name: 'RangeError', message: /steps/ });
  assert.throws(() => createGenerator({ dataset, seed: 1, options: { margin: 0.6 } }), { name: 'RangeError', message: /margin/ });
});

test('generate rejects bad input with clear errors', () => {
  const generator = createGenerator({ dataset, seed: SEED });
  const first = generator.generate({ iteration: 1, parent: null, feedback: null });
  assert.throws(() => generator.generate(), { name: 'TypeError', message: /input/ });
  assert.throws(() => generator.generate({ parent: null, feedback: null }), { name: 'TypeError', message: /iteration/ });
  assert.throws(() => generator.generate({ iteration: 1.5, parent: null, feedback: null }), { name: 'TypeError', message: /iteration/ });
  assert.throws(() => generator.generate({ iteration: 0, parent: null, feedback: null }), { name: 'RangeError', message: /iteration/ });
  assert.throws(() => generator.generate({ iteration: -3, parent: null, feedback: null }), { name: 'RangeError', message: /iteration/ });
  assert.throws(() => generator.generate({ iteration: 2, parent: { id: 'x', positions: [0, 1] }, feedback: null }), {
    name: 'TypeError',
    message: /parent/,
  });
  assert.throws(
    () => generator.generate({ iteration: 2, parent: { ...first, positions: new Float64Array(4) }, feedback: null }),
    { name: 'TypeError', message: /length/ },
  );
  assert.throws(
    () => generator.generate({ iteration: 2, parent: { ...first, positions: new Float64Array(2 * N).fill(NaN) }, feedback: null }),
    { name: 'TypeError', message: /finite/ },
  );
});

test('options override the defaults and stay deterministic', () => {
  const options = { steps: 2, stepMax: 0.02, shakeChance: 0, relaxChance: 0 };
  const a = createGenerator({ dataset, seed: SEED, options });
  const b = createGenerator({ dataset, seed: SEED, options });
  const parent = a.generate({ iteration: 1, parent: null, feedback: null });
  const childA = a.generate({ iteration: 2, parent, feedback: null });
  assert.equal(childA.meta.strategy, 'refine');
  assert.ok(childA.meta.stepSize > GENERATOR_DEFAULTS.stepMax && childA.meta.stepSize < 0.02);
  assert.deepEqual(b.generate({ iteration: 2, parent, feedback: null }).positions, childA.positions);
  const plain = createGenerator({ dataset, seed: SEED }).generate({ iteration: 2, parent, feedback: null });
  assert.notDeepEqual(plain.positions, childA.positions);
});
