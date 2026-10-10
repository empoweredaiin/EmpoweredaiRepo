// End-to-end tests: the real generator, heuristic evaluator and loop controller wired together
// exactly as src/main.js does, with a fake clock so no test depends on wall time.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { CRITERIA, LIMITS, STATUS, TERMINATION, candidateIssues, evaluationIssues } from '../src/contracts.js';
import { cosine, semanticNeighbours } from '../src/core/knn.js';
import { createDataset } from '../src/data/dataset.js';
import { createGenerator } from '../src/generator/generator.js';
import { createDiscriminator } from '../src/discriminator/discriminator.js';
import { createLoopController } from '../src/loop/controller.js';
import { describeTermination } from '../src/ui/hud.js';

const dataset = createDataset();

function run(seed, { msPerIteration = 16, limits } = {}) {
  let clock = 0;
  const controller = createLoopController({
    generator: createGenerator({ dataset, seed }),
    discriminator: createDiscriminator({ dataset }),
    limits,
    now: () => clock,
    yieldToHost: () => {
      clock += msPerIteration;
      return Promise.resolve();
    },
  });
  return controller.start().then((state) => ({ state, controller }));
}

test('a full run terminates with an explicit reason inside every hard limit', async () => {
  const { state } = await run(7);
  assert.equal(state.status, STATUS.TERMINATED);
  assert.ok(Object.values(TERMINATION).includes(state.terminationReason), `reason: ${state.terminationReason}`);
  assert.notEqual(state.terminationReason, TERMINATION.CANCELLED);
  assert.ok(state.iteration >= 1 && state.iteration <= LIMITS.maxIterations);
  assert.equal(state.history.length, state.iteration);
  assert.equal(state.acceptedCount + state.rejectedCount, state.iteration);
  assert.ok(describeTermination(state.terminationReason, LIMITS).detail.length > 0);
});

test('the best score never decreases and rejected candidates never replace the best', async () => {
  const { state } = await run(7);
  let best = 0;
  for (const record of state.history) {
    if (record.accepted) {
      assert.ok(record.score > best, `iteration ${record.iteration} accepted without improving`);
      best = record.score;
    }
    assert.equal(record.bestScore, best, `iteration ${record.iteration}`);
  }
  assert.equal(state.bestScore, best);
  assert.equal(state.best.evaluation.overall, best);
});

test('the loop improves on the initial layout and the result is well-formed', async () => {
  const { state } = await run(7);
  assert.ok(state.bestScore > state.history[0].score, 'no improvement over the first candidate');
  assert.deepEqual(candidateIssues(state.best.candidate, dataset.points.length), []);
  assert.deepEqual(evaluationIssues(state.best.evaluation), []);
  assert.equal(state.best.evaluation.kind, 'heuristic');
  for (const key of CRITERIA) assert.ok(key in state.best.evaluation.criteria);
});

test('the same seed reproduces the run exactly; a different seed does not', async () => {
  const [a, b, c] = await Promise.all([run(11), run(11), run(12)]);
  assert.deepEqual(a.state.history, b.state.history);
  assert.equal(a.state.terminationReason, b.state.terminationReason);
  assert.deepEqual(a.state.best.candidate.positions, b.state.best.candidate.positions);
  assert.notDeepEqual(a.state.best.candidate.positions, c.state.best.candidate.positions);
});

test('determinism does not depend on timing: a slower clock gives the same iterations', async () => {
  const [fast, slow] = await Promise.all([run(5, { msPerIteration: 1 }), run(5, { msPerIteration: 40 })]);
  const shared = Math.min(fast.state.history.length, slow.state.history.length);
  assert.deepEqual(fast.state.history.slice(0, shared), slow.state.history.slice(0, shared));
});

test('reset then start on one controller replays the same run', async () => {
  const { state: first, controller } = await run(3);
  controller.reset();
  assert.equal(controller.getState().status, STATUS.IDLE);
  assert.equal(controller.getState().best, null);
  const second = await controller.start();
  assert.deepEqual(second.history, first.history);
});

test('the time limit ends a slow run with an explicit reason', async () => {
  const { state } = await run(7, {
    msPerIteration: 2500,
    limits: { ...LIMITS, plateauIterations: 1000, convergenceScore: 2 },
  });
  assert.equal(state.terminationReason, TERMINATION.TIME_LIMIT);
  // The clock advances 2.5 s in each yield, so 10 s have elapsed after the yield that follows
  // iteration 4, and iteration 5 must not start.
  assert.equal(state.iteration, 4);
  assert.ok(state.activeMs >= LIMITS.timeLimitMs);
});

test('the iteration limit ends a run that neither converges nor plateaus', async () => {
  const { state } = await run(7, { limits: { ...LIMITS, plateauIterations: 1000, convergenceScore: 2 } });
  assert.equal(state.terminationReason, TERMINATION.MAX_ITERATIONS);
  assert.equal(state.iteration, LIMITS.maxIterations);
});

test('running the loop leaves the source vectors untouched', async () => {
  const before = JSON.stringify(dataset.points.map((p) => p.vector));
  await run(7);
  assert.equal(JSON.stringify(dataset.points.map((p) => p.vector)), before);
  assert.ok(Object.isFrozen(dataset.points[0].vector));
});

test('every termination reason has an explicit description', () => {
  for (const reason of Object.values(TERMINATION)) {
    const { title, detail } = describeTermination(reason, LIMITS);
    assert.ok(title && detail, reason);
  }
});

test('the neighbours a selection reveals are the true five nearest by cosine similarity', () => {
  const lists = semanticNeighbours(dataset, 8);
  const points = dataset.points;
  for (let i = 0; i < points.length; i++) {
    const ranked = points
      .map((point, j) => ({ j, similarity: j === i ? -Infinity : cosine(points[i].vector, point.vector) }))
      .sort((a, b) => b.similarity - a.similarity || a.j - b.j)
      .slice(0, 5)
      .map((entry) => entry.j);
    assert.deepEqual(Array.from(lists[i]).slice(0, 5), ranked, `point ${i}`);
    assert.ok(!ranked.includes(i), 'a point is never its own neighbour');
  }
});

test('only a convergence is described as converged; a plateau is named as one', () => {
  for (const reason of Object.values(TERMINATION)) {
    assert.equal(describeTermination(reason, LIMITS).converged, reason === TERMINATION.CONVERGED, reason);
  }
  const plateau = describeTermination(TERMINATION.PLATEAU, LIMITS);
  assert.match(plateau.title, /plateau/i);
  assert.match(plateau.title, /not converged/i);
  assert.doesNotMatch(plateau.detail, /converged/i);
});

test('scoring weights and stopping thresholds are unchanged', () => {
  assert.deepEqual({ ...LIMITS }, {
    convergenceScore: 0.9,
    convergenceStreak: 3,
    plateauEpsilon: 0.005,
    plateauIterations: 10,
    maxIterations: 100,
    timeLimitMs: 10_000,
  });
  const evaluation = createDiscriminator({ dataset }).evaluate({
    candidate: createGenerator({ dataset, seed: 1 }).generate({ iteration: 1, parent: null, feedback: null }),
    previous: null,
  });
  assert.deepEqual(evaluation.weights, { neighbourhood: 0.4, separation: 0.3, overlap: 0.2, stability: 0.1 });
});

test('pacing the loop at the page speed does not change how seeds 0 to 19 end', async () => {
  for (let seed = 0; seed < 20; seed++) {
    const [instant, paced] = await Promise.all([run(seed, { msPerIteration: 0 }), run(seed, { msPerIteration: 200 })]);
    assert.equal(paced.state.terminationReason, instant.state.terminationReason, `seed ${seed}`);
    assert.deepEqual(paced.state.history, instant.state.history, `seed ${seed}`);
  }
});
