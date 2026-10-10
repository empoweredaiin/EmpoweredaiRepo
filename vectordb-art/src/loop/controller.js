/**
 * Loop controller: drives generate -> evaluate -> accept/reject iterations and
 * enforces the stopping rules. See the "Behaviour (normative)" block on the
 * LoopController typedef in src/contracts.js; this file implements it.
 *
 * The generator and the discriminator are injected, never imported.
 */

import { EVENTS, LIMITS, STATUS, TERMINATION } from '../contracts.js';

/** @typedef {import('../contracts.js').LoopController} LoopController */
/** @typedef {import('../contracts.js').LoopControllerOptions} LoopControllerOptions */
/** @typedef {import('../contracts.js').LoopState} LoopState */
/** @typedef {import('../contracts.js').LoopEvent} LoopEvent */
/** @typedef {import('../contracts.js').IterationRecord} IterationRecord */
/** @typedef {import('../contracts.js').Evaluation} Evaluation */

/**
 * Default acceptance rule: a valid candidate that strictly improves on the best.
 * @param {Evaluation} evaluation
 * @param {LoopState['best']} best
 * @returns {boolean}
 */
function acceptIfBetter(evaluation, best) {
  return evaluation.valid && (best === null || evaluation.overall > best.evaluation.overall);
}

/** Per-run progress. Replaced wholesale by start() and reset(). */
function freshProgress() {
  return {
    iteration: 0,
    acceptedCount: 0,
    rejectedCount: 0,
    /** @type {LoopState['best']} */
    best: null,
    /** @type {LoopState['last']} */
    last: null,
    /** @type {IterationRecord[]} */
    history: [],
    convergenceStreak: 0,
    plateauCount: 0,
    /** @type {LoopState['terminationReason']} */
    terminationReason: null,
  };
}

/**
 * Creates a loop controller.
 *
 * Notes beyond the normative block in src/contracts.js:
 * - The first iteration runs synchronously inside start(); the loop first yields
 *   to the host after it.
 * - Hard limits: the stopping conditions are also checked immediately before an
 *   iteration starts, so time that elapsed while yielding to the host can never
 *   let an iteration begin at or beyond the time limit.
 * - A candidate whose evaluation is not valid, or whose overall score is not a
 *   finite number, is always rejected (whatever `accept` says) and is recorded
 *   with score 0.
 * - If the generator, the discriminator, `accept` or `yieldToHost` throws, the
 *   run ends: status becomes 'terminated' with terminationReason null, a
 *   'terminate' event is emitted and the run promise rejects with the error.
 *
 * @param {LoopControllerOptions} options
 * @returns {LoopController}
 */
export function createLoopController(options) {
  const { generator, discriminator } = options ?? {};
  if (typeof generator?.generate !== 'function') {
    throw new TypeError('createLoopController: generator.generate must be a function');
  }
  if (typeof discriminator?.evaluate !== 'function') {
    throw new TypeError('createLoopController: discriminator.evaluate must be a function');
  }

  const limits = { ...LIMITS, ...options.limits };
  const now = options.now ?? (() => performance.now());
  const yieldToHost = options.yieldToHost ?? (() => new Promise((resolve) => setTimeout(resolve, 0)));
  const accept = options.accept ?? acceptIfBetter;

  /** @type {Set<(event: LoopEvent) => void>} */
  const listeners = new Set();

  /** @type {LoopState['status']} */
  let status = STATUS.IDLE;
  let runId = 0;
  let progress = freshProgress();

  /** Active time banked by completed 'running' stretches of this run. */
  let bankedMs = 0;
  /** Clock reading at which the current 'running' stretch began. */
  let runningSince = 0;

  /** Settlement handles of the run in flight (or of the last run). */
  /** @type {{ promise: Promise<LoopState>, resolve: (state: LoopState) => void, reject: (error: unknown) => void }|null} */
  let run = null;
  /** Wakes the loop while it waits in 'paused'. */
  /** @type {(() => void)|null} */
  let wake = null;

  const activeMs = () => bankedMs + (status === STATUS.RUNNING ? now() - runningSince : 0);
  const bestScore = () => (progress.best === null ? 0 : progress.best.evaluation.overall);
  const inFlight = () => status === STATUS.RUNNING || status === STATUS.PAUSED;
  /** True while the loop holding `token` is the one allowed to act. */
  const isCurrent = (token) => token === runId && inFlight();

  /** @returns {LoopState} */
  function getState() {
    return {
      status,
      runId,
      iteration: progress.iteration,
      acceptedCount: progress.acceptedCount,
      rejectedCount: progress.rejectedCount,
      best: progress.best,
      bestScore: bestScore(),
      last: progress.last,
      history: progress.history.slice(),
      convergenceStreak: progress.convergenceStreak,
      plateauCount: progress.plateauCount,
      activeMs: activeMs(),
      terminationReason: progress.terminationReason,
    };
  }

  /**
   * @param {LoopEvent['type']} type
   * @param {{ state?: LoopState, record?: IterationRecord }} [extra]
   */
  function emit(type, extra = {}) {
    const event = { type, state: extra.state ?? getState() };
    if (extra.record) event.record = extra.record;
    for (const listener of [...listeners]) {
      try {
        listener(event);
      } catch (error) {
        console.error('LoopController listener failed:', error);
      }
    }
  }

  function wakeLoop() {
    const resolve = wake;
    wake = null;
    resolve?.();
  }

  /**
   * Ends the run in flight. With `error` the run promise rejects and
   * terminationReason stays null; otherwise it resolves with the final snapshot.
   * @param {LoopState['terminationReason']} reason
   * @param {unknown} [error]
   */
  function terminate(reason, error) {
    const settling = run;
    bankedMs = activeMs();
    status = STATUS.TERMINATED;
    progress.terminationReason = reason;
    wakeLoop();
    const state = getState();
    emit(EVENTS.TERMINATE, { state });
    if (reason === null) settling.reject(error);
    else settling.resolve(state);
  }

  /** @returns {LoopState['terminationReason']} First matching stopping condition, or null. */
  function stoppingReason() {
    if (progress.convergenceStreak >= limits.convergenceStreak) return TERMINATION.CONVERGED;
    if (progress.plateauCount >= limits.plateauIterations) return TERMINATION.PLATEAU;
    if (progress.iteration >= limits.maxIterations) return TERMINATION.MAX_ITERATIONS;
    if (activeMs() >= limits.timeLimitMs) return TERMINATION.TIME_LIMIT;
    return null;
  }

  /** @returns {boolean} True when a stopping condition matched and the run was terminated. */
  function terminateIfDone() {
    const reason = stoppingReason();
    if (reason === null) return false;
    terminate(reason);
    return true;
  }

  /** Steps 1-6 of one iteration. */
  function iterate() {
    const { best } = progress;
    const iteration = progress.iteration + 1;
    const candidate = generator.generate({
      iteration,
      parent: best?.candidate ?? null,
      feedback: best?.evaluation ?? null,
    });
    const evaluation = discriminator.evaluate({ candidate, previous: best?.candidate ?? null });
    if (evaluation === null || typeof evaluation !== 'object') {
      throw new TypeError('LoopController: discriminator.evaluate must return an Evaluation object');
    }

    const scorable = evaluation.valid === true && Number.isFinite(evaluation.overall);
    const bestBefore = bestScore();
    const accepted = scorable && Boolean(accept(evaluation, best));

    progress.iteration = iteration;
    if (accepted) {
      progress.best = Object.freeze({ candidate, evaluation });
      progress.acceptedCount += 1;
      progress.convergenceStreak =
        evaluation.overall >= limits.convergenceScore ? progress.convergenceStreak + 1 : 0;
    } else {
      progress.rejectedCount += 1;
    }
    progress.last = Object.freeze({ candidate, evaluation, accepted });

    const improvement = bestScore() - bestBefore;
    progress.plateauCount = improvement < limits.plateauEpsilon ? progress.plateauCount + 1 : 0;

    /** @type {IterationRecord} */
    const record = Object.freeze({
      iteration,
      candidateId: String(candidate?.id ?? ''),
      score: scorable ? evaluation.overall : 0,
      accepted,
      bestScore: bestScore(),
      criteria: evaluation.criteria,
    });
    progress.history.push(record);
    emit(EVENTS.ITERATION, { record });
  }

  /**
   * The one async loop of a run. `token` is the runId it belongs to: after every
   * await (and after every emit, since listeners may call back in) it stops as
   * soon as it is no longer the current run.
   * @param {number} token
   */
  async function loop(token) {
    try {
      while (isCurrent(token)) {
        if (status === STATUS.PAUSED) {
          await new Promise((resolve) => {
            wake = resolve;
          });
          continue;
        }
        if (terminateIfDone()) return;
        iterate();
        if (!isCurrent(token) || terminateIfDone()) return;
        await yieldToHost();
      }
    } catch (error) {
      if (isCurrent(token)) terminate(null, error);
    }
  }

  function start() {
    if (inFlight()) return run.promise;

    let resolve;
    let reject;
    const promise = new Promise((res, rej) => {
      resolve = res;
      reject = rej;
    });
    run = { promise, resolve, reject };

    runId += 1;
    progress = freshProgress();
    bankedMs = 0;
    runningSince = now();
    status = STATUS.RUNNING;
    emit(EVENTS.START);
    loop(runId);
    return promise;
  }

  function pause() {
    if (status !== STATUS.RUNNING) return false;
    bankedMs = activeMs();
    status = STATUS.PAUSED;
    emit(EVENTS.PAUSE);
    return true;
  }

  function resume() {
    if (status !== STATUS.PAUSED) return false;
    runningSince = now();
    status = STATUS.RUNNING;
    emit(EVENTS.RESUME);
    wakeLoop();
    return true;
  }

  function stop() {
    if (!inFlight()) return false;
    terminate(TERMINATION.CANCELLED);
    return true;
  }

  function reset() {
    if (inFlight()) terminate(TERMINATION.CANCELLED);
    runId += 1;
    status = STATUS.IDLE;
    progress = freshProgress();
    bankedMs = 0;
    emit(EVENTS.RESET);
  }

  function subscribe(listener) {
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  }

  return { start, pause, resume, stop, reset, getState, subscribe };
}
