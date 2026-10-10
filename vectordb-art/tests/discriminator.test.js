import test from 'node:test';
import assert from 'node:assert/strict';

import { CRITERIA, DEFAULT_WEIGHTS, NEIGHBOUR_K, POINT_RADIUS, evaluationIssues } from '../src/contracts.js';
import { createRng } from '../src/core/rng.js';
import { semanticNeighbours } from '../src/core/knn.js';
import { createDataset } from '../src/data/dataset.js';
import {
  CRITERION_DESCRIPTIONS,
  DISCRIMINATOR_DEFAULTS,
  createDiscriminator,
  meanSilhouette,
  scoreNeighbourhood,
  scoreOverlap,
  scoreSeparation,
  scoreStability,
} from '../src/discriminator/discriminator.js';

const dataset = createDataset();
const N = dataset.points.length;
const clusters = Int32Array.from(dataset.points, (point) => point.cluster);
const EPSILON = 1e-12;

function candidateOf(positions, id = '1:1') {
  return { id, seed: 1, iteration: 1, parentId: null, positions, meta: { strategy: 'test', stepSize: 0 } };
}

function randomPositions(seed, count = N) {
  const rng = createRng(seed);
  return Float64Array.from({ length: 2 * count }, () => rng());
}

/** Each cluster on its own 6-wide grid, spaced wider than a point diameter, far from the others. */
function clusteredPositions(spacing = 0.03) {
  const positions = new Float64Array(2 * N);
  const seen = new Map();
  for (let i = 0; i < N; i++) {
    const cluster = dataset.points[i].cluster;
    const n = seen.get(cluster) ?? 0;
    seen.set(cluster, n + 1);
    positions[2 * i] = 0.2 + 0.3 * (cluster >> 1) + ((n % 6) - 2.5) * spacing;
    positions[2 * i + 1] = (cluster % 2 ? 0.7 : 0.3) + (Math.floor(n / 6) - 2.5) * spacing;
  }
  return positions;
}

/** Embedding coordinates `ax` and `ay`, rescaled to fill the unit square. */
function projectedPositions(ax, ay) {
  const positions = new Float64Array(2 * N);
  for (const [offset, axis] of [[0, ax], [1, ay]]) {
    const values = dataset.points.map((point) => point.vector[axis]);
    const min = Math.min(...values);
    const span = Math.max(...values) - min;
    for (let i = 0; i < N; i++) positions[2 * i + offset] = (values[i] - min) / span;
  }
  return positions;
}

function gridPositions(count, spacing) {
  const side = Math.ceil(Math.sqrt(count));
  const positions = new Float64Array(2 * count);
  for (let i = 0; i < count; i++) {
    positions[2 * i] = 0.05 + (i % side) * spacing;
    positions[2 * i + 1] = 0.05 + Math.floor(i / side) * spacing;
  }
  return positions;
}

/** Four points on the unit circle of a 2D embedding, in two clusters. */
function tinyDataset() {
  const unit = (angle) => [Math.cos(angle), Math.sin(angle)];
  const points = [
    { id: 'a1', vector: unit(0), cluster: 0, label: 'a1' },
    { id: 'a2', vector: unit(0.1), cluster: 0, label: 'a2' },
    { id: 'b1', vector: unit(2), cluster: 1, label: 'b1' },
    { id: 'b2', vector: unit(2.1), cluster: 1, label: 'b2' },
  ];
  return { seed: 0, dim: 2, points, clusterNames: ['a', 'b'] };
}

function assertUnit(value, name) {
  assert.ok(Number.isFinite(value) && value >= 0 && value <= 1, `${name} = ${value} is not in [0, 1]`);
}

function assertWellFormed(evaluation) {
  assert.deepEqual(evaluationIssues(evaluation), []);
  assert.equal(evaluation.kind, 'heuristic');
}

test('exports defaults and a description for every criterion', () => {
  assert.equal(DISCRIMINATOR_DEFAULTS.k, NEIGHBOUR_K);
  assert.equal(DISCRIMINATOR_DEFAULTS.pointRadius, POINT_RADIUS);
  assert.deepEqual(DISCRIMINATOR_DEFAULTS.weights, DEFAULT_WEIGHTS);
  assert.equal(DISCRIMINATOR_DEFAULTS.stabilityScale, 0.25);
  assert.ok(Object.isFrozen(DISCRIMINATOR_DEFAULTS));
  assert.ok(Object.isFrozen(CRITERION_DESCRIPTIONS));
  assert.deepEqual(Object.keys(CRITERION_DESCRIPTIONS), [...CRITERIA]);
  for (const key of CRITERIA) assert.ok(CRITERION_DESCRIPTIONS[key].length > 20);
});

test('every criterion is finite and in [0, 1] on random layouts', () => {
  const discriminator = createDiscriminator({ dataset });
  let previous = null;
  for (let seed = 1; seed <= 25; seed++) {
    const candidate = candidateOf(randomPositions(seed), `1:${seed}`);
    const evaluation = discriminator.evaluate({ candidate, previous });
    assertWellFormed(evaluation);
    assert.equal(evaluation.valid, true);
    assert.deepEqual(evaluation.issues, []);
    assert.equal(evaluation.candidateId, candidate.id);
    for (const key of CRITERIA) assertUnit(evaluation.criteria[key], key);
    assertUnit(evaluation.overall, 'overall');
    previous = candidate;
  }
});

test('neighbourhood: exact on a hand-made dataset', () => {
  const tiny = tinyDataset();
  const semantic = semanticNeighbours(tiny, 1);
  assert.deepEqual(semantic.map((list) => Array.from(list)), [[1], [0], [3], [2]]);
  const paired = Float64Array.of(0.1, 0.1, 0.15, 0.1, 0.9, 0.9, 0.85, 0.9);
  const crossed = Float64Array.of(0.1, 0.1, 0.9, 0.9, 0.15, 0.1, 0.85, 0.9);
  const half = Float64Array.of(0.1, 0.1, 0.15, 0.1, 0.5, 0.1, 0.9, 0.9);
  assert.equal(scoreNeighbourhood(paired, semantic), 1);
  assert.equal(scoreNeighbourhood(crossed, semantic), 0);
  // Points 0 and 1 keep each other; 2 is nearest to 1 (miss); 3 is nearest to 2 (hit).
  assert.equal(scoreNeighbourhood(half, semantic), 0.75);
});

test('neighbourhood: an embedding projection beats a random layout', () => {
  const semantic = semanticNeighbours(dataset, NEIGHBOUR_K);
  const projected = scoreNeighbourhood(projectedPositions(0, 1), semantic);
  for (let seed = 1; seed <= 5; seed++) {
    const random = scoreNeighbourhood(randomPositions(seed), semantic);
    assertUnit(random, 'random neighbourhood');
    assert.ok(projected > random + 0.05, `projected ${projected} vs random ${random}`);
  }
});

test('neighbourhood: agrees with a brute-force full sort', () => {
  const semantic = semanticNeighbours(dataset, NEIGHBOUR_K);
  const positions = randomPositions(42);
  let total = 0;
  for (let i = 0; i < N; i++) {
    const order = [];
    for (let j = 0; j < N; j++) if (j !== i) order.push(j);
    const distance = (j) => Math.hypot(positions[2 * j] - positions[2 * i], positions[2 * j + 1] - positions[2 * i + 1]);
    order.sort((a, b) => distance(a) - distance(b) || a - b);
    const nearest = new Set(order.slice(0, NEIGHBOUR_K));
    total += Array.from(semantic[i]).filter((j) => nearest.has(j)).length / NEIGHBOUR_K;
  }
  assert.ok(Math.abs(scoreNeighbourhood(positions, semantic) - total / N) < EPSILON);
});

test('separation: separated clusters beat a shuffled assignment of the same positions', () => {
  const positions = clusteredPositions();
  const separated = scoreSeparation(positions, clusters);
  assert.ok(separated >= 0.9, `separated layout scored ${separated}`);

  const rng = createRng(11);
  const shuffled = Int32Array.from(clusters);
  for (let i = N - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  const mixed = scoreSeparation(positions, shuffled);
  assertUnit(mixed, 'mixed separation');
  assert.ok(mixed <= 0.55, `mixed layout scored ${mixed}`);
  assert.ok(separated > mixed + 0.3);
  assert.ok(scoreSeparation(randomPositions(3), clusters) <= 0.55);
});

test('separation: silhouette mapping and exact values on a hand-made layout', () => {
  // Two clusters of two points: a = 0.1 and b = 0.5 for every point.
  const positions = Float64Array.of(0, 0, 0.1, 0, 0, 0.5, 0.1, 0.5);
  const labels = [0, 0, 1, 1];
  const a = 0.1;
  const b = (0.5 + Math.hypot(0.1, 0.5)) / 2;
  const expected = (b - a) / b;
  assert.ok(Math.abs(meanSilhouette(positions, labels) - expected) < EPSILON);
  assert.ok(Math.abs(scoreSeparation(positions, labels, 1) - (0.5 + 0.5 * expected)) < EPSILON);
  assert.equal(scoreSeparation(positions, labels, 0.5), 1);
  // Swapping the labels of two points puts each cluster's members far apart.
  assert.ok(scoreSeparation(positions, [0, 1, 1, 0]) < 0.5);
});

test('separation: degenerate inputs stay finite', () => {
  const stacked = new Float64Array(2 * N).fill(0.5);
  assert.equal(meanSilhouette(stacked, clusters), 0);
  assert.equal(scoreSeparation(stacked, clusters), 0.5);
  // A single cluster, and a cluster with a single point.
  assert.equal(scoreSeparation(randomPositions(1, 5), [0, 0, 0, 0, 0]), 0.5);
  assertUnit(scoreSeparation(randomPositions(2, 5), [0, 0, 0, 0, 1]), 'singleton cluster');
  assert.equal(scoreSeparation(new Float64Array(0), []), 0.5);
});

test('overlap: stacked points score 0, a spaced grid scores exactly 1', () => {
  assert.equal(scoreOverlap(new Float64Array(2 * N).fill(0.5)), 0);
  assert.equal(scoreOverlap(gridPositions(N, 2 * POINT_RADIUS + 0.001)), 1);
  assert.equal(scoreOverlap(clusteredPositions()), 1);
  assert.equal(scoreOverlap(Float64Array.of(0.5, 0.5)), 1);
});

test('overlap: penetration is smooth and uses the given radius', () => {
  // Two points at half a diameter: each penetrates its neighbour by 0.5.
  const pair = Float64Array.of(0.5, 0.5, 0.5 + POINT_RADIUS, 0.5);
  assert.ok(Math.abs(scoreOverlap(pair) - 0.5) < EPSILON);
  assert.equal(scoreOverlap(pair, POINT_RADIUS / 4), 1);
  let last = -1;
  for (const spacing of [0, 0.004, 0.008, 0.012, 0.016, 0.02, 0.025]) {
    const score = scoreOverlap(gridPositions(64, spacing));
    assert.ok(score > last || score === 1, `overlap did not improve at spacing ${spacing}`);
    last = score;
  }
  assert.equal(last, 1);
});

test('stability: 1 without or with an identical previous layout, 0 at the documented bound', () => {
  const base = Float64Array.from({ length: 2 * N }, (_, i) => (i % 2 === 0 ? 0.25 : 0.5));
  const shifted = (dx) => base.map((v, i) => (i % 2 === 0 ? v + dx : v));
  assert.equal(scoreStability(base, null), 1);
  assert.equal(scoreStability(base, undefined), 1);
  assert.equal(scoreStability(base, Float64Array.from(base)), 1);

  let last = 1;
  for (const dx of [0.03125, 0.0625, 0.125, 0.1875]) {
    const score = scoreStability(shifted(dx), base);
    assert.ok(Math.abs(score - (1 - dx / DISCRIMINATOR_DEFAULTS.stabilityScale)) < EPSILON);
    assert.ok(score < last, `stability did not decrease at displacement ${dx}`);
    last = score;
  }
  assert.equal(scoreStability(shifted(0.25), base), 0);
  assert.equal(scoreStability(shifted(0.5), base), 0);
  assert.equal(scoreStability(shifted(0.25), base, 0.5), 0.5);
});

test('stability: a malformed previous layout counts as no previous layout', () => {
  const discriminator = createDiscriminator({ dataset });
  const candidate = candidateOf(randomPositions(4));
  const malformed = [
    candidateOf(new Float64Array(10)),
    candidateOf(Array.from(randomPositions(5))),
    candidateOf(new Float64Array(2 * N).fill(NaN)),
    { id: 'no-positions' },
    'not a layout',
  ];
  for (const previous of malformed) {
    const evaluation = discriminator.evaluate({ candidate, previous });
    assertWellFormed(evaluation);
    assert.equal(evaluation.valid, true);
    assert.equal(evaluation.criteria.stability, 1);
  }
  assert.equal(scoreStability(candidate.positions, new Float64Array(3)), 1);
});

test('overall is the weighted sum of the criteria', () => {
  const discriminator = createDiscriminator({ dataset });
  const previous = candidateOf(randomPositions(8), '1:1');
  for (const positions of [randomPositions(9), clusteredPositions(), projectedPositions(2, 3)]) {
    const evaluation = discriminator.evaluate({ candidate: candidateOf(positions, '1:2'), previous });
    assert.deepEqual(evaluation.weights, { ...DEFAULT_WEIGHTS });
    const expected = CRITERIA.reduce((sum, key) => sum + evaluation.weights[key] * evaluation.criteria[key], 0);
    assert.ok(Math.abs(evaluation.overall - expected) < EPSILON);
    assert.ok(evaluation.criteria.stability < 1);
  }
});

test('criteria reported by evaluate match the standalone functions', () => {
  const positions = randomPositions(21);
  const previous = candidateOf(randomPositions(22), '1:1');
  const { criteria } = createDiscriminator({ dataset }).evaluate({ candidate: candidateOf(positions), previous });
  assert.equal(criteria.neighbourhood, scoreNeighbourhood(positions, semanticNeighbours(dataset, NEIGHBOUR_K)));
  assert.equal(criteria.separation, scoreSeparation(positions, clusters));
  assert.equal(criteria.overlap, scoreOverlap(positions));
  assert.equal(criteria.stability, scoreStability(positions, previous.positions));
});

test('custom weights are normalised and options are honoured', () => {
  const weights = { neighbourhood: 2, separation: 2, overlap: 4, stability: 0 };
  const discriminator = createDiscriminator({ dataset, options: { weights, k: 4, pointRadius: 0.03, stabilityScale: 0.5 } });
  const positions = randomPositions(13);
  const evaluation = discriminator.evaluate({ candidate: candidateOf(positions), previous: null });
  assertWellFormed(evaluation);
  assert.deepEqual(evaluation.weights, { neighbourhood: 0.25, separation: 0.25, overlap: 0.5, stability: 0 });
  assert.deepEqual(weights, { neighbourhood: 2, separation: 2, overlap: 4, stability: 0 });
  assert.equal(evaluation.criteria.neighbourhood, scoreNeighbourhood(positions, semanticNeighbours(dataset, 4)));
  assert.equal(evaluation.criteria.overlap, scoreOverlap(positions, 0.03));
  const { criteria } = evaluation;
  const expected = 0.25 * criteria.neighbourhood + 0.25 * criteria.separation + 0.5 * criteria.overlap;
  assert.ok(Math.abs(evaluation.overall - expected) < EPSILON);

  const onlyOverlap = createDiscriminator({
    dataset,
    options: { weights: { neighbourhood: 0, separation: 0, overlap: 3, stability: 0 } },
  });
  assert.equal(onlyOverlap.evaluate({ candidate: candidateOf(clusteredPositions()), previous: null }).overall, 1);
});

test('createDiscriminator rejects an invalid dataset or options with a TypeError', () => {
  const bad = [
    undefined,
    {},
    { dataset: null },
    { dataset: { points: [] } },
    { dataset: { points: [{ id: 'a', cluster: 0 }] } },
    { dataset: { points: [{ id: 'a', vector: [1], cluster: -1 }] } },
    { dataset, options: null },
    { dataset, options: { weights: null } },
    { dataset, options: { weights: { neighbourhood: 1, separation: 1, overlap: 1 } } },
    { dataset, options: { weights: { ...DEFAULT_WEIGHTS, stability: -0.1 } } },
    { dataset, options: { weights: { ...DEFAULT_WEIGHTS, overlap: NaN } } },
    { dataset, options: { weights: { ...DEFAULT_WEIGHTS, beauty: 1 } } },
    { dataset, options: { weights: { neighbourhood: 0, separation: 0, overlap: 0, stability: 0 } } },
    { dataset, options: { k: 0 } },
    { dataset, options: { k: 2.5 } },
    { dataset, options: { pointRadius: 0 } },
    { dataset, options: { silhouetteFull: -1 } },
    { dataset, options: { stabilityScale: Infinity } },
  ];
  for (const config of bad) {
    assert.throws(() => createDiscriminator(config), TypeError, JSON.stringify(config?.options ?? config));
  }
});

test('malformed candidates give valid=false, overall 0 and never throw', () => {
  const discriminator = createDiscriminator({ dataset });
  const withValue = (index, value) => {
    const positions = randomPositions(6);
    positions[index] = value;
    return candidateOf(positions, 'bad:1');
  };
  const malformed = [
    null,
    undefined,
    42,
    {},
    candidateOf(new Float64Array(2 * N - 2), 'bad:1'),
    withValue(7, NaN),
    withValue(8, 1.5),
    withValue(9, -0.01),
    withValue(10, Infinity),
    candidateOf(Array.from(randomPositions(6)), 'bad:1'),
    candidateOf(new Float32Array(2 * N), 'bad:1'),
    { ...candidateOf(randomPositions(6)), id: '' },
    { ...candidateOf(randomPositions(6)), iteration: 0 },
  ];
  for (const candidate of malformed) {
    const evaluation = discriminator.evaluate({ candidate, previous: null });
    assertWellFormed(evaluation);
    assert.equal(evaluation.valid, false);
    assert.equal(evaluation.overall, 0);
    assert.ok(evaluation.issues.length > 0);
    assert.ok(evaluation.issues.every((issue) => typeof issue === 'string'));
    for (const key of CRITERIA) assert.equal(evaluation.criteria[key], 0);
    assert.deepEqual(evaluation.weights, { ...DEFAULT_WEIGHTS });
    assert.equal(typeof evaluation.candidateId, 'string');
    assert.ok(evaluation.candidateId.length > 0);
  }
  assert.equal(discriminator.evaluate({ candidate: withValue(7, NaN), previous: null }).candidateId, 'bad:1');
  assert.equal(discriminator.evaluate({ candidate: null, previous: null }).candidateId, 'invalid');
  for (const input of [undefined, null, {}]) {
    const evaluation = discriminator.evaluate(input);
    assertWellFormed(evaluation);
    assert.equal(evaluation.valid, false);
  }
});

test('degenerate layouts and tiny datasets never produce NaN', () => {
  const stacked = createDiscriminator({ dataset }).evaluate({
    candidate: candidateOf(new Float64Array(2 * N).fill(0.5)),
    previous: null,
  });
  assertWellFormed(stacked);
  assert.equal(stacked.criteria.overlap, 0);

  // Fewer points than k + 1, and a single point.
  const tiny = tinyDataset();
  const small = createDiscriminator({ dataset: tiny }).evaluate({
    candidate: candidateOf(Float64Array.of(0.1, 0.1, 0.15, 0.1, 0.9, 0.9, 0.85, 0.9)),
    previous: null,
  });
  assertWellFormed(small);
  assert.equal(small.valid, true);
  assert.equal(small.criteria.neighbourhood, 1);

  const single = createDiscriminator({ dataset: { ...tiny, points: tiny.points.slice(0, 1) } }).evaluate({
    candidate: candidateOf(Float64Array.of(0.5, 0.5)),
    previous: null,
  });
  assertWellFormed(single);
  assert.deepEqual(single.criteria, { neighbourhood: 1, separation: 0.5, overlap: 1, stability: 1 });
});

test('evaluate is deterministic and does not mutate its inputs', () => {
  const candidate = candidateOf(randomPositions(30), '1:2');
  const previous = candidateOf(randomPositions(31), '1:1');
  Object.freeze(candidate.meta);
  Object.freeze(previous.meta);
  Object.freeze(candidate);
  Object.freeze(previous);
  const input = Object.freeze({ candidate, previous });
  const candidateCopy = Float64Array.from(candidate.positions);
  const previousCopy = Float64Array.from(previous.positions);

  const first = createDiscriminator({ dataset }).evaluate(input);
  const second = createDiscriminator({ dataset }).evaluate(input);
  const third = createDiscriminator({ dataset: createDataset() }).evaluate(input);
  assert.deepEqual(second, first);
  assert.deepEqual(third, first);
  assert.deepEqual(candidate.positions, candidateCopy);
  assert.deepEqual(previous.positions, previousCopy);

  // Results are detached: editing one does not leak into the next.
  first.weights.neighbourhood = 99;
  first.criteria.overlap = 99;
  assert.deepEqual(createDiscriminator({ dataset }).evaluate(input), second);
});

test('structured layouts outscore a random one, each on the criterion it serves', () => {
  const discriminator = createDiscriminator({ dataset });
  const evaluate = (positions) => discriminator.evaluate({ candidate: candidateOf(positions), previous: null });
  const random = evaluate(randomPositions(17));
  const projected = evaluate(projectedPositions(2, 3));
  const clustered = evaluate(clusteredPositions());
  assert.ok(projected.overall > random.overall, `projected ${projected.overall} vs random ${random.overall}`);
  assert.ok(clustered.overall > random.overall, `clustered ${clustered.overall} vs random ${random.overall}`);
  // A grid per topic separates the topics best but ignores the structure inside each one; a
  // projection keeps more true neighbours. Neither dominates the other overall.
  assert.ok(clustered.criteria.separation > projected.criteria.separation);
  assert.ok(projected.criteria.neighbourhood > random.criteria.neighbourhood);
});

test('one evaluation of the full dataset is fast', () => {
  const discriminator = createDiscriminator({ dataset });
  const layouts = Array.from({ length: 10 }, (_, i) => candidateOf(randomPositions(100 + i), `1:${i + 1}`));
  const run = (rounds) => {
    for (let i = 0; i < rounds; i++) {
      discriminator.evaluate({ candidate: layouts[i % 10], previous: layouts[(i + 1) % 10] });
    }
  };
  run(100);
  const rounds = 300;
  const started = performance.now();
  run(rounds);
  const perCall = (performance.now() - started) / rounds;
  assert.ok(perCall < 5, `evaluate took ${perCall.toFixed(3)} ms on average`);
});
