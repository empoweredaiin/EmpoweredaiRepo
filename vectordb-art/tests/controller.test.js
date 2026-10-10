import { describe, test } from 'node:test';
import assert from 'node:assert/strict';

import { DEFAULT_WEIGHTS, EVENTS, LIMITS, STATUS, TERMINATION } from '../src/contracts.js';
import { createLoopController } from '../src/loop/controller.js';

/* ------------------------------------------------------------------ */
/* Fakes                                                              */
/* ------------------------------------------------------------------ */

/** Fake generator that records every input. `onGenerate` runs before each candidate is built. */
function fakeGenerator(onGenerate = () => {}) {
  const calls = [];
  return {
    seed: 1,
    calls,
    generate(input) {
      calls.push(input);
      onGenerate(input);
      const { iteration, parent } = input;
      return {
        id: `s:${iteration}`,
        seed: 1,
        iteration,
        parentId: parent?.id ?? null,
        positions: new Float64Array(2),
        meta: { strategy: 'fake', stepSize: 0 },
      };
    },
  };
}

function makeEvaluation(candidateId, overall, valid = true) {
  return {
    candidateId,
    kind: 'heuristic',
    criteria: { neighbourhood: overall, separation: overall, overlap: overall, stability: overall },
    weights: DEFAULT_WEIGHTS,
    overall,
    valid,
    issues: valid ? [] : ['malformed'],
  };
}

/**
 * Fake discriminator driven by a script: an array indexed by iteration - 1 or a
 * function of the iteration. An entry is an overall score or `{ overall, valid }`.
 * Running past the end of an array script throws, which fails the run loudly.
 */
function fakeDiscriminator(script) {
  const calls = [];
  return {
    calls,
    evaluate(input) {
      calls.push(input);
      const { iteration, id } = input.candidate;
      const entry = typeof script === 'function' ? script(iteration) : script[iteration - 1];
      if (entry === undefined) throw new Error(`no scripted score for iteration ${iteration}`);
      return typeof entry === 'object' ? makeEvaluation(id, entry.overall, entry.valid) : makeEvaluation(id, entry);
    },
  };
}

const flush = () => new Promise((resolve) => setImmediate(resolve));

/** A yieldToHost whose promises stay pending until release() is called. */
function createGate() {
  let pending = [];
  return {
    yieldToHost: () => new Promise((resolve) => pending.push(resolve)),
    /** Resolves every pending yield and lets the loop run until it blocks again. */
    async release() {
      const resolvers = pending;
      pending = [];
      for (const resolve of resolvers) resolve();
      await flush();
    },
  };
}

/** Keeps releasing the gate until the run promise settles. */
async function drain(gate, promise) {
  let settled = false;
  const mark = () => {
    settled = true;
  };
  promise.then(mark, mark);
  while (!settled) await gate.release();
  return promise;
}

/**
 * Builds a controller around fakes. The clock is manual (`clock.t`), and by
 * default yieldToHost resolves immediately.
 */
function setup({ scores, limits, accept, gated = false, onGenerate, onYield } = {}) {
  const clock = { t: 0 };
  const generator = fakeGenerator((input) => onGenerate?.(input, clock));
  const discriminator = fakeDiscriminator(scores);
  const gate = gated ? createGate() : null;
  const yieldToHost = gate
    ? gate.yieldToHost
    : () => {
        onYield?.(clock);
        return Promise.resolve();
      };
  const controller = createLoopController({
    generator,
    discriminator,
    limits,
    accept,
    now: () => clock.t,
    yieldToHost,
  });
  const events = [];
  controller.subscribe((event) => events.push(event));
  const types = () => events.map((event) => event.type);
  const iterationStates = () => events.filter((e) => e.type === EVENTS.ITERATION).map((e) => e.state);
  return { controller, generator, discriminator, clock, gate, events, types, iterationStates };
}

const acceptAll = () => true;
const iterationsOf = (state) => state.history.map((record) => record.iteration);

/* ------------------------------------------------------------------ */
/* Stopping conditions                                                */
/* ------------------------------------------------------------------ */

describe('stopping conditions', () => {
  test('default LIMITS equal the spec values', () => {
    assert.deepEqual(
      { ...LIMITS },
      {
        convergenceScore: 0.9,
        convergenceStreak: 3,
        plateauEpsilon: 0.005,
        plateauIterations: 10,
        maxIterations: 100,
        timeLimitMs: 10_000,
      },
    );
  });

  test('convergence: terminates exactly when the third consecutive accepted iteration >= 0.90 completes', async () => {
    const { controller, generator, types } = setup({ scores: [0.5, 0.9, 0.92, 0.93] });
    const final = await controller.start();
    assert.equal(final.status, STATUS.TERMINATED);
    assert.equal(final.terminationReason, TERMINATION.CONVERGED);
    assert.equal(final.iteration, 4);
    assert.equal(final.convergenceStreak, 3);
    assert.equal(final.acceptedCount, 4);
    assert.equal(generator.calls.length, 4);
    assert.deepEqual(types(), ['start', 'iteration', 'iteration', 'iteration', 'iteration', 'terminate']);
  });

  test('convergence: a rejected iteration in between does not reset the streak', async () => {
    const { controller, iterationStates } = setup({ scores: [0.91, 0.5, 0.92, 0.93] });
    const final = await controller.start();
    assert.equal(final.terminationReason, TERMINATION.CONVERGED);
    assert.equal(final.iteration, 4);
    assert.equal(final.rejectedCount, 1);
    assert.deepEqual(
      iterationStates().map((s) => s.convergenceStreak),
      [1, 1, 2, 3],
    );
  });

  test('convergence: an accepted iteration below 0.90 resets the streak', async () => {
    const { controller, iterationStates } = setup({
      scores: [0.91, 0.92, 0.5, 0.91, 0.92, 0.93],
      accept: acceptAll,
    });
    const final = await controller.start();
    assert.equal(final.terminationReason, TERMINATION.CONVERGED);
    assert.equal(final.iteration, 6);
    assert.equal(final.acceptedCount, 6);
    assert.deepEqual(
      iterationStates().map((s) => s.convergenceStreak),
      [1, 2, 0, 1, 2, 3],
    );
  });

  test('convergence: two accepted iterations >= 0.90 are not enough', async () => {
    const { controller } = setup({ scores: [0.91, 0.92, 0.1, 0.1, 0.1], limits: { maxIterations: 5 } });
    const final = await controller.start();
    assert.equal(final.terminationReason, TERMINATION.MAX_ITERATIONS);
    assert.equal(final.convergenceStreak, 2);
    assert.equal(final.iteration, 5);
  });

  test('plateau: terminates after exactly ten consecutive iterations with improvement < 0.005', async () => {
    const { controller, generator, iterationStates } = setup({ scores: () => 0.5 });
    const final = await controller.start();
    assert.equal(final.terminationReason, TERMINATION.PLATEAU);
    assert.equal(final.iteration, 11);
    assert.equal(final.plateauCount, 10);
    assert.equal(generator.calls.length, 11);
    assert.deepEqual(
      iterationStates().map((s) => s.plateauCount),
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    );
  });

  test('plateau: an improvement >= 0.005 resets the counter', async () => {
    const scores = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6, ...Array(10).fill(0.1)];
    const { controller, iterationStates } = setup({ scores });
    const final = await controller.start();
    assert.equal(final.terminationReason, TERMINATION.PLATEAU);
    assert.equal(final.iteration, 17);
    assert.deepEqual(
      iterationStates().map((s) => s.plateauCount),
      [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    );
  });

  test('plateau: accepted but tiny improvements (0.004 each) still count', async () => {
    const { controller } = setup({ scores: (i) => 0.5 + 0.004 * (i - 1) });
    const final = await controller.start();
    assert.equal(final.terminationReason, TERMINATION.PLATEAU);
    assert.equal(final.iteration, 11);
    assert.equal(final.acceptedCount, 11);
    assert.equal(final.rejectedCount, 0);
    assert.equal(final.plateauCount, 10);
  });

  test('plateau boundary: an improvement of exactly epsilon resets, just below it counts', async () => {
    // Binary-exact values, so the improvement is exactly 0.125 and then 0.0625.
    const exact = setup({
      scores: [0.25, 0.375, 0.4375],
      limits: { plateauEpsilon: 0.125, maxIterations: 3 },
    });
    await exact.controller.start();
    assert.deepEqual(
      exact.iterationStates().map((s) => s.plateauCount),
      [0, 0, 1],
    );

    // Default epsilon: 0.5 -> 0.505 is an improvement of (at least) 0.005 in floating point.
    const decimal = setup({ scores: [0.5, 0.505, 0.509], limits: { maxIterations: 3 } });
    await decimal.controller.start();
    assert.deepEqual(
      decimal.iterationStates().map((s) => s.plateauCount),
      [0, 0, 1],
    );
  });

  test('max iterations: stops at exactly 100 attempted iterations with the default limits', async () => {
    const { controller, generator } = setup({ scores: (i) => i * 0.008 });
    const final = await controller.start();
    assert.equal(final.terminationReason, TERMINATION.MAX_ITERATIONS);
    assert.equal(final.iteration, 100);
    assert.equal(final.history.length, 100);
    assert.equal(final.acceptedCount, 100);
    assert.equal(generator.calls.length, 100);
    assert.ok(final.bestScore < LIMITS.convergenceScore);
  });

  test('time limit: stops when activeMs reaches 10000 with the default limits', async () => {
    const { controller, generator } = setup({
      scores: (i) => i * 0.01,
      onGenerate: (_input, clock) => {
        clock.t += 1000;
      },
    });
    const final = await controller.start();
    assert.equal(final.terminationReason, TERMINATION.TIME_LIMIT);
    assert.equal(final.iteration, 10);
    assert.equal(final.activeMs, 10_000);
    assert.equal(generator.calls.length, 10);
  });

  test('time limit: no iteration starts once the limit was reached while yielding to the host', async () => {
    const { controller, generator } = setup({
      scores: (i) => i * 0.01,
      onYield: (clock) => {
        clock.t += 1000;
      },
    });
    const final = await controller.start();
    assert.equal(final.terminationReason, TERMINATION.TIME_LIMIT);
    assert.equal(final.iteration, 10);
    assert.equal(final.activeMs, 10_000);
    assert.equal(generator.calls.length, 10);
  });

  test('time limit: paused time is excluded from activeMs and does not trip the limit', async () => {
    const { controller, generator, clock, gate } = setup({ scores: (i) => i * 0.01, gated: true });
    const promise = controller.start();
    clock.t = 1000;
    assert.equal(controller.pause(), true);
    clock.t += 60_000;
    assert.equal(controller.getState().activeMs, 1000);
    assert.equal(controller.resume(), true);
    await gate.release();

    const state = controller.getState();
    assert.equal(state.status, STATUS.RUNNING);
    assert.equal(state.iteration, 2);
    assert.equal(state.activeMs, 1000);
    assert.equal(generator.calls.length, 2);

    clock.t += 500;
    controller.stop();
    assert.equal((await promise).activeMs, 1500);
  });

  test('precedence: converged > plateau > max-iterations > time-limit on the same iteration', async () => {
    const converged = setup({ scores: [0.91, 0.92, 0.93], limits: { maxIterations: 3 } });
    assert.equal((await converged.controller.start()).terminationReason, TERMINATION.CONVERGED);

    const plateau = setup({ scores: [0.5, 0.5, 0.5], limits: { plateauIterations: 2, maxIterations: 3 } });
    const plateauFinal = await plateau.controller.start();
    assert.equal(plateauFinal.iteration, 3);
    assert.equal(plateauFinal.terminationReason, TERMINATION.PLATEAU);

    const maxIterations = setup({
      scores: [0.1, 0.2],
      limits: { maxIterations: 2, timeLimitMs: 2000 },
      onGenerate: (_input, clock) => {
        clock.t += 1000;
      },
    });
    const maxFinal = await maxIterations.controller.start();
    assert.equal(maxFinal.activeMs, 2000);
    assert.equal(maxFinal.terminationReason, TERMINATION.MAX_ITERATIONS);
  });
});

/* ------------------------------------------------------------------ */
/* Cancellation                                                       */
/* ------------------------------------------------------------------ */

describe('cancellation', () => {
  test('stop() mid-run terminates synchronously with cancelled, keeps state, and nothing runs afterwards', async () => {
    const { controller, generator, gate, events, types } = setup({ scores: (i) => i * 0.01, gated: true });
    const promise = controller.start();
    await gate.release();
    assert.equal(controller.getState().iteration, 2);

    assert.equal(controller.stop(), true);
    const stopped = controller.getState();
    assert.equal(stopped.status, STATUS.TERMINATED);
    assert.equal(stopped.terminationReason, TERMINATION.CANCELLED);
    assert.equal(stopped.best.candidate.id, 's:2');
    assert.equal(stopped.history.length, 2);

    const final = await promise;
    assert.deepEqual(final, stopped);

    const eventCount = events.length;
    await gate.release();
    await gate.release();
    assert.equal(generator.calls.length, 2);
    assert.equal(events.length, eventCount);
    assert.deepEqual(types(), ['start', 'iteration', 'iteration', 'terminate']);
    assert.deepEqual(controller.getState(), stopped);
  });

  test('stop() while paused works', async () => {
    const { controller, generator, gate, types } = setup({ scores: (i) => i * 0.01, gated: true });
    const promise = controller.start();
    controller.pause();
    await gate.release();
    assert.equal(controller.stop(), true);
    const final = await promise;
    assert.equal(final.status, STATUS.TERMINATED);
    assert.equal(final.terminationReason, TERMINATION.CANCELLED);
    assert.equal(final.iteration, 1);
    await gate.release();
    assert.equal(generator.calls.length, 1);
    assert.deepEqual(types(), ['start', 'iteration', 'pause', 'terminate']);
  });

  test('stop() when idle or already terminated returns false', async () => {
    const { controller, events } = setup({ scores: [0.5], limits: { maxIterations: 1 } });
    assert.equal(controller.stop(), false);
    assert.equal(events.length, 0);
    const final = await controller.start();
    assert.equal(controller.stop(), false);
    assert.equal(controller.getState().terminationReason, final.terminationReason);
  });

  test('stop() called from an iteration listener prevents any further iteration', async () => {
    const { controller, generator } = setup({ scores: (i) => i * 0.01 });
    controller.subscribe((event) => {
      if (event.type === EVENTS.ITERATION && event.state.iteration === 3) controller.stop();
    });
    const final = await controller.start();
    await flush();
    assert.equal(final.terminationReason, TERMINATION.CANCELLED);
    assert.equal(final.iteration, 3);
    assert.equal(generator.calls.length, 3);
  });

  test('reset() mid-run cancels, clears state, silences the stale loop, and a new run starts cleanly', async () => {
    const { controller, generator, gate, events, types } = setup({
      scores: (i) => i * 0.01,
      limits: { maxIterations: 3 },
      gated: true,
    });
    const first = controller.start();
    await gate.release();
    assert.equal(controller.getState().iteration, 2);

    controller.reset();
    assert.deepEqual(types(), ['start', 'iteration', 'iteration', 'terminate', 'reset']);
    assert.equal(events.at(-2).state.terminationReason, TERMINATION.CANCELLED);
    assert.equal(events.at(-2).state.status, STATUS.TERMINATED);

    const cleared = controller.getState();
    assert.deepEqual(cleared, {
      status: STATUS.IDLE,
      runId: 2,
      iteration: 0,
      acceptedCount: 0,
      rejectedCount: 0,
      best: null,
      bestScore: 0,
      last: null,
      history: [],
      convergenceStreak: 0,
      plateauCount: 0,
      activeMs: 0,
      terminationReason: null,
    });

    const cancelled = await first;
    assert.equal(cancelled.status, STATUS.TERMINATED);
    assert.equal(cancelled.terminationReason, TERMINATION.CANCELLED);
    assert.equal(cancelled.iteration, 2);
    assert.equal(cancelled.runId, 1);

    // The stale loop's pending yield resolves: it must stay silent.
    const eventCount = events.length;
    await gate.release();
    await gate.release();
    assert.equal(events.length, eventCount);
    assert.equal(generator.calls.length, 2);
    assert.deepEqual(controller.getState(), cleared);

    const second = await drain(gate, controller.start());
    assert.equal(second.runId, 3);
    assert.equal(second.terminationReason, TERMINATION.MAX_ITERATIONS);
    assert.deepEqual(iterationsOf(second), [1, 2, 3]);
    assert.equal(generator.calls[2].iteration, 1);
    assert.equal(generator.calls[2].parent, null);
    assert.equal(generator.calls.length, 5);
  });

  test('reset() while the stale yield is still pending does not disturb the next run', async () => {
    const { controller, generator, gate } = setup({
      scores: (i) => i * 0.01,
      limits: { maxIterations: 3 },
      gated: true,
    });
    controller.start();
    controller.reset();
    // The old loop is still parked on its yield when the new run begins.
    const final = await drain(gate, controller.start());
    assert.deepEqual(iterationsOf(final), [1, 2, 3]);
    assert.equal(generator.calls.length, 4);
  });

  test('reset() when idle or terminated emits just reset', async () => {
    const { controller, types } = setup({ scores: [0.5], limits: { maxIterations: 1 } });
    controller.reset();
    assert.deepEqual(types(), ['reset']);
    await controller.start();
    controller.reset();
    assert.deepEqual(types(), ['reset', 'start', 'iteration', 'terminate', 'reset']);
    const state = controller.getState();
    assert.equal(state.status, STATUS.IDLE);
    assert.equal(state.history.length, 0);
    assert.equal(state.terminationReason, null);
  });
});

/* ------------------------------------------------------------------ */
/* Rejected candidates                                                */
/* ------------------------------------------------------------------ */

describe('rejected candidates', () => {
  test('a lower-scoring candidate is rejected and never replaces best; best is fed to the next iteration', async () => {
    const { controller, generator, discriminator, iterationStates } = setup({
      scores: [0.6, 0.4, 0.7],
      limits: { maxIterations: 3 },
    });
    const final = await controller.start();
    const [afterFirst, afterRejected] = iterationStates();

    assert.equal(afterRejected.best, afterFirst.best);
    assert.equal(afterRejected.best.candidate.id, 's:1');
    assert.equal(afterRejected.bestScore, 0.6);
    assert.equal(afterRejected.rejectedCount, 1);
    assert.equal(afterRejected.acceptedCount, 1);
    assert.equal(afterRejected.last.candidate.id, 's:2');
    assert.equal(afterRejected.last.accepted, false);
    assert.deepEqual(
      { ...final.history[1], criteria: undefined },
      { iteration: 2, candidateId: 's:2', score: 0.4, accepted: false, bestScore: 0.6, criteria: undefined },
    );

    assert.equal(generator.calls[0].parent, null);
    assert.equal(generator.calls[0].feedback, null);
    assert.equal(discriminator.calls[0].previous, null);
    assert.equal(generator.calls[2].parent, afterFirst.best.candidate);
    assert.equal(generator.calls[2].feedback, afterFirst.best.evaluation);
    assert.equal(discriminator.calls[2].previous, afterFirst.best.candidate);

    assert.equal(final.best.candidate.id, 's:3');
    assert.equal(final.best.candidate.parentId, 's:1');
  });

  test('an evaluation with valid=false or a NaN overall is rejected even if "higher"', async () => {
    const scores = [0.5, { overall: 0.99, valid: false }, NaN, Infinity];
    for (const accept of [undefined, acceptAll]) {
      const { controller } = setup({ scores, accept, limits: { maxIterations: 4 } });
      const final = await controller.start();
      assert.equal(final.best.candidate.id, 's:1');
      assert.equal(final.bestScore, 0.5);
      assert.equal(final.rejectedCount, 3);
      assert.equal(final.convergenceStreak, 0);
      assert.deepEqual(
        final.history.map((r) => [r.accepted, r.score, r.bestScore]),
        [
          [true, 0.5, 0.5],
          [false, 0, 0.5],
          [false, 0, 0.5],
          [false, 0, 0.5],
        ],
      );
    }
  });

  test('an equal score is rejected (strict improvement)', async () => {
    const { controller } = setup({ scores: [0.5, 0.5], limits: { maxIterations: 2 } });
    const final = await controller.start();
    assert.equal(final.acceptedCount, 1);
    assert.equal(final.rejectedCount, 1);
    assert.equal(final.best.candidate.id, 's:1');
    assert.equal(final.history[1].accepted, false);
  });
});

/* ------------------------------------------------------------------ */
/* Determinism                                                        */
/* ------------------------------------------------------------------ */

describe('determinism', () => {
  const scores = (i) => ((i * 37) % 23) / 40 + i * 0.002;

  test('two controllers with identical fakes produce identical histories and termination', async () => {
    const a = await setup({ scores }).controller.start();
    const b = await setup({ scores }).controller.start();
    assert.ok(a.history.length > 5);
    assert.deepEqual(a.history, b.history);
    assert.equal(a.terminationReason, b.terminationReason);
    assert.equal(a.bestScore, b.bestScore);
  });

  test('run, reset(), run again on one controller reproduces the same history', async () => {
    const { controller } = setup({ scores });
    const first = await controller.start();
    controller.reset();
    const second = await controller.start();
    assert.deepEqual(second.history, first.history);
    assert.equal(second.terminationReason, first.terminationReason);
    assert.equal(second.best.candidate.id, first.best.candidate.id);
  });
});

/* ------------------------------------------------------------------ */
/* Concurrency                                                        */
/* ------------------------------------------------------------------ */

describe('concurrency', () => {
  test('start() twice while running returns the same promise and only one loop runs', async () => {
    const { controller, generator, gate, types } = setup({
      scores: (i) => i * 0.01,
      limits: { maxIterations: 6 },
      gated: true,
    });
    const first = controller.start();
    const second = controller.start();
    assert.equal(second, first);
    await gate.release();
    assert.equal(controller.start(), first);

    const final = await drain(gate, first);
    assert.equal(final.runId, 1);
    assert.equal(generator.calls.length, 6);
    assert.deepEqual(
      generator.calls.map((call) => call.iteration),
      [1, 2, 3, 4, 5, 6],
    );
    assert.deepEqual(iterationsOf(final), [1, 2, 3, 4, 5, 6]);
    assert.equal(types().filter((type) => type === EVENTS.START).length, 1);
  });

  test('start() while paused neither resumes nor starts a second loop', async () => {
    const { controller, generator, gate } = setup({ scores: (i) => i * 0.01, gated: true });
    const first = controller.start();
    controller.pause();
    assert.equal(controller.start(), first);
    await gate.release();
    await gate.release();

    const state = controller.getState();
    assert.equal(state.status, STATUS.PAUSED);
    assert.equal(state.runId, 1);
    assert.equal(state.iteration, 1);
    assert.equal(generator.calls.length, 1);

    controller.stop();
    await first;
  });

  test('start() after termination begins a fresh run', async () => {
    const { controller, events } = setup({ scores: [0.5, 0.6, 0.7], limits: { maxIterations: 3 } });
    const firstPromise = controller.start();
    const first = await firstPromise;
    assert.equal(first.runId, 1);

    const secondPromise = controller.start();
    assert.notEqual(secondPromise, firstPromise);
    const startState = events.filter((e) => e.type === EVENTS.START)[1].state;
    assert.equal(startState.status, STATUS.RUNNING);
    assert.equal(startState.runId, 2);
    assert.equal(startState.iteration, 0);
    assert.equal(startState.best, null);
    assert.equal(startState.history.length, 0);
    assert.equal(startState.terminationReason, null);
    assert.equal(startState.activeMs, 0);

    const second = await secondPromise;
    assert.equal(second.runId, 2);
    assert.deepEqual(second.history, first.history);
    assert.equal(first.history.length, 3);
  });
});

/* ------------------------------------------------------------------ */
/* Pause / resume                                                     */
/* ------------------------------------------------------------------ */

describe('pause and resume', () => {
  test('pause() stops iterations and resume() continues from the same iteration with best preserved', async () => {
    const { controller, generator, gate, types } = setup({ scores: (i) => i * 0.01, gated: true });
    assert.equal(controller.pause(), false);
    assert.equal(controller.resume(), false);

    const promise = controller.start();
    assert.equal(controller.resume(), false);
    assert.equal(controller.pause(), true);
    assert.equal(controller.pause(), false);
    assert.equal(controller.getState().status, STATUS.PAUSED);

    await gate.release();
    await gate.release();
    assert.equal(generator.calls.length, 1);
    const paused = controller.getState();
    assert.equal(paused.iteration, 1);

    assert.equal(controller.resume(), true);
    assert.equal(controller.resume(), false);
    await flush();
    const resumed = controller.getState();
    assert.equal(resumed.status, STATUS.RUNNING);
    assert.equal(resumed.iteration, 2);
    assert.equal(generator.calls.length, 2);
    assert.equal(generator.calls[1].iteration, 2);
    assert.equal(generator.calls[1].parent, paused.best.candidate);
    assert.deepEqual(types(), ['start', 'iteration', 'pause', 'resume', 'iteration']);

    controller.stop();
    const final = await promise;
    assert.equal(controller.pause(), false);
    assert.equal(controller.resume(), false);
    assert.equal(final.iteration, 2);
  });

  test('pause() and resume() within a single host yield do not skip or duplicate iterations', async () => {
    const { controller, gate } = setup({ scores: (i) => i * 0.01, limits: { maxIterations: 4 }, gated: true });
    const promise = controller.start();
    controller.pause();
    controller.resume();
    controller.pause();
    controller.resume();
    const final = await drain(gate, promise);
    assert.deepEqual(iterationsOf(final), [1, 2, 3, 4]);
  });
});

/* ------------------------------------------------------------------ */
/* Misc                                                               */
/* ------------------------------------------------------------------ */

describe('misc', () => {
  test('snapshots are detached from the controller', async () => {
    const { controller, gate, events } = setup({ scores: (i) => i * 0.01, limits: { maxIterations: 3 }, gated: true });
    const promise = controller.start();
    const early = controller.getState();
    early.history.push({ bogus: true });
    early.iteration = 99;
    events[1].state.history.length = 0;

    const final = await drain(gate, promise);
    assert.equal(early.history.length, 2);
    assert.equal(early.status, STATUS.RUNNING);
    assert.deepEqual(iterationsOf(final), [1, 2, 3]);
    assert.deepEqual(iterationsOf(controller.getState()), [1, 2, 3]);
    assert.notEqual(controller.getState().history, controller.getState().history);
  });

  test('a throwing listener does not break the run or other listeners', async (t) => {
    const logged = t.mock.method(console, 'error', () => {});
    const { controller, events } = setup({ scores: [0.5, 0.6], limits: { maxIterations: 2 } });
    controller.subscribe(() => {
      throw new Error('listener failure');
    });
    const late = [];
    controller.subscribe((event) => late.push(event.type));

    const final = await controller.start();
    assert.equal(final.terminationReason, TERMINATION.MAX_ITERATIONS);
    assert.equal(events.length, 4);
    assert.deepEqual(late, ['start', 'iteration', 'iteration', 'terminate']);
    assert.equal(logged.mock.callCount(), 4);
  });

  test('unsubscribe stops delivery', async () => {
    const { controller } = setup({ scores: [0.5, 0.6], limits: { maxIterations: 2 } });
    const seen = [];
    const unsubscribe = controller.subscribe((event) => {
      seen.push(event.type);
      if (event.type === EVENTS.ITERATION) unsubscribe();
    });
    await controller.start();
    assert.deepEqual(seen, ['start', 'iteration']);
  });

  test('iteration events carry the record that was appended to history', async () => {
    const { controller, events } = setup({ scores: [0.5, 0.4], limits: { maxIterations: 2 } });
    const final = await controller.start();
    const iterationEvents = events.filter((e) => e.type === EVENTS.ITERATION);
    assert.deepEqual(
      iterationEvents.map((e) => e.record),
      final.history,
    );
    assert.equal(iterationEvents[0].state.history.length, 1);
    assert.equal(events[0].record, undefined);
  });

  test('invalid options throw TypeError', () => {
    const generator = fakeGenerator();
    const discriminator = fakeDiscriminator([0.5]);
    assert.throws(() => createLoopController(), TypeError);
    assert.throws(() => createLoopController({ discriminator }), TypeError);
    assert.throws(() => createLoopController({ generator }), TypeError);
    assert.throws(() => createLoopController({ generator: { generate: 1 }, discriminator }), TypeError);
    assert.throws(() => createLoopController({ generator, discriminator: { evaluate: 'x' } }), TypeError);
    assert.doesNotThrow(() => createLoopController({ generator, discriminator }));
  });

  test('a throwing generator ends the run: terminated, reason null, promise rejects', async () => {
    const failure = new Error('generator exploded');
    const { controller, types } = setup({
      scores: (i) => i * 0.01,
      onGenerate: ({ iteration }) => {
        if (iteration === 3) throw failure;
      },
    });
    await assert.rejects(controller.start(), failure);
    const state = controller.getState();
    assert.equal(state.status, STATUS.TERMINATED);
    assert.equal(state.terminationReason, null);
    assert.equal(state.iteration, 2);
    assert.equal(state.best.candidate.id, 's:2');
    assert.deepEqual(types(), ['start', 'iteration', 'iteration', 'terminate']);

    // The controller stays usable.
    controller.reset();
    assert.equal(controller.getState().status, STATUS.IDLE);
  });

  test('default clock and yieldToHost work without injection', async () => {
    const controller = createLoopController({
      generator: fakeGenerator(),
      discriminator: fakeDiscriminator([0.5, 0.6, 0.7]),
      limits: { maxIterations: 3 },
    });
    const final = await controller.start();
    assert.equal(final.terminationReason, TERMINATION.MAX_ITERATIONS);
    assert.equal(final.iteration, 3);
    assert.ok(final.activeMs >= 0 && final.activeMs < LIMITS.timeLimitMs);
  });
});
