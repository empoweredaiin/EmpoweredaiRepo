/**
 * Shared contracts for the VectorDB art experiment.
 *
 * This file is the single source of truth for the data structures and function
 * signatures that cross module boundaries. It is owned by the orchestrator:
 * modules import from it, they do not redefine or extend these shapes locally.
 *
 * Module map
 *   src/data/dataset.js              createDataset({ seed })            -> Dataset
 *   src/generator/generator.js       createGenerator({ dataset, seed }) -> Generator
 *   src/discriminator/discriminator.js createDiscriminator({ dataset }) -> Discriminator
 *   src/loop/controller.js           createLoopController({ ... })      -> LoopController
 *   src/render/renderer.js           createRenderer({ canvas, dataset }) -> Renderer
 *
 * Dependency rule: generator, discriminator, loop and render never import each
 * other. They may import src/contracts.js and src/core/*. Only src/main.js wires
 * them together.
 */

/* ------------------------------------------------------------------ */
/* Constants                                                          */
/* ------------------------------------------------------------------ */

/** Layout space is the unit square: every coordinate is in [0, 1]. */
export const LAYOUT_MIN = 0;
export const LAYOUT_MAX = 1;

/**
 * Radius of one point in layout units. Two points overlap when the distance
 * between their centres is below 2 * POINT_RADIUS.
 */
export const POINT_RADIUS = 0.012;

/** Size of the semantic neighbourhood (k nearest neighbours by cosine similarity). */
export const NEIGHBOUR_K = 8;

/** Criterion keys, in display order. Every score is in [0, 1], higher is better. */
export const CRITERIA = Object.freeze(['neighbourhood', 'separation', 'overlap', 'stability']);

/** Weights of the overall score. They sum to 1. */
export const DEFAULT_WEIGHTS = Object.freeze({
  neighbourhood: 0.4,
  separation: 0.3,
  overlap: 0.2,
  stability: 0.1,
});

/** Stopping rules. See LoopController for the exact accounting. */
export const LIMITS = Object.freeze({
  convergenceScore: 0.9,
  convergenceStreak: 3,
  plateauEpsilon: 0.005,
  plateauIterations: 10,
  maxIterations: 100,
  timeLimitMs: 10_000,
});

export const STATUS = Object.freeze({
  IDLE: 'idle',
  RUNNING: 'running',
  PAUSED: 'paused',
  TERMINATED: 'terminated',
});

export const TERMINATION = Object.freeze({
  CONVERGED: 'converged',
  PLATEAU: 'plateau',
  MAX_ITERATIONS: 'max-iterations',
  TIME_LIMIT: 'time-limit',
  CANCELLED: 'cancelled',
});

export const EVENTS = Object.freeze({
  START: 'start',
  ITERATION: 'iteration',
  PAUSE: 'pause',
  RESUME: 'resume',
  TERMINATE: 'terminate',
  RESET: 'reset',
});

/* ------------------------------------------------------------------ */
/* Dataset                                                            */
/* ------------------------------------------------------------------ */

/**
 * One stored embedding. Frozen; never mutated by any module.
 * @typedef {Object} SourcePoint
 * @property {string} id
 * @property {ReadonlyArray<number>} vector  Unit-length embedding of length Dataset.dim.
 * @property {number} cluster                Index into Dataset.clusterNames.
 * @property {string} label
 */

/**
 * @typedef {Object} Dataset
 * @property {number} seed
 * @property {number} dim
 * @property {ReadonlyArray<SourcePoint>} points
 * @property {ReadonlyArray<string>} clusterNames
 */

/* ------------------------------------------------------------------ */
/* Candidate layout                                                   */
/* ------------------------------------------------------------------ */

/**
 * A 2D arrangement of every dataset point. Immutable once returned: a consumer
 * that needs to change positions copies them first.
 *
 * @typedef {Object} CandidateLayout
 * @property {string} id                 `${seed}:${iteration}`.
 * @property {number} seed
 * @property {number} iteration          1-based attempted iteration that produced it.
 * @property {string|null} parentId      Id of the layout it was derived from.
 * @property {Float64Array} positions    Length 2 * N, interleaved [x0, y0, x1, y1, ...].
 *                                       Point i is dataset.points[i]. All values finite, in [0, 1].
 * @property {{ strategy: string, stepSize: number }} meta  Diagnostic only.
 */

/* ------------------------------------------------------------------ */
/* Generator                                                          */
/* ------------------------------------------------------------------ */

/**
 * @typedef {Object} GenerateInput
 * @property {number} iteration                1-based attempted iteration.
 * @property {CandidateLayout|null} parent     Best accepted layout so far, null on the first iteration.
 * @property {Evaluation|null} feedback        Evaluation of `parent`, null on the first iteration.
 */

/**
 * `generate` is a pure function of (dataset, seed, input): the same arguments
 * always give the same positions, regardless of how often or in what order it
 * was called before. It must not mutate the dataset, the parent or the feedback.
 *
 * @typedef {Object} Generator
 * @property {number} seed
 * @property {(input: GenerateInput) => CandidateLayout} generate
 */

/* ------------------------------------------------------------------ */
/* Discriminator (heuristic evaluator)                                */
/* ------------------------------------------------------------------ */

/**
 * @typedef {Object} EvaluateInput
 * @property {CandidateLayout} candidate
 * @property {CandidateLayout|null} previous  Best accepted layout the candidate is compared
 *                                            against for stability. Null -> stability is 1.
 */

/**
 * Result of scoring one candidate. `kind` is always 'heuristic': these are
 * hand-written geometric measures, not the output of a trained model.
 *
 * @typedef {Object} Evaluation
 * @property {string} candidateId
 * @property {'heuristic'} kind
 * @property {{ neighbourhood: number, separation: number, overlap: number, stability: number }} criteria
 *           Each in [0, 1], higher is better. `overlap` is 1 when no two points overlap.
 * @property {{ neighbourhood: number, separation: number, overlap: number, stability: number }} weights
 * @property {number} overall   Weighted mean of the criteria, in [0, 1].
 * @property {boolean} valid    False when the candidate was malformed; then every score is 0.
 * @property {string[]} issues  Human-readable validation problems, empty when valid.
 */

/**
 * `evaluate` is pure and never throws on a malformed candidate: it returns an
 * Evaluation with valid=false instead.
 *
 * @typedef {Object} Discriminator
 * @property {(input: EvaluateInput) => Evaluation} evaluate
 */

/* ------------------------------------------------------------------ */
/* Loop controller                                                    */
/* ------------------------------------------------------------------ */

/**
 * @typedef {Object} IterationRecord
 * @property {number} iteration     1-based attempted iteration.
 * @property {string} candidateId
 * @property {number} score         Overall score of this candidate.
 * @property {boolean} accepted
 * @property {number} bestScore     Best score after this iteration.
 * @property {Evaluation['criteria']} criteria
 */

/**
 * Snapshot returned by getState() and carried on every event. A snapshot is
 * detached from the controller: later iterations do not change it.
 *
 * @typedef {Object} LoopState
 * @property {'idle'|'running'|'paused'|'terminated'} status
 * @property {number} runId                 Incremented by every start() that begins a run and by every reset().
 * @property {number} iteration             Attempted iterations completed in this run.
 * @property {number} acceptedCount
 * @property {number} rejectedCount
 * @property {{ candidate: CandidateLayout, evaluation: Evaluation }|null} best
 * @property {number} bestScore             0 when there is no best yet.
 * @property {{ candidate: CandidateLayout, evaluation: Evaluation, accepted: boolean }|null} last
 * @property {IterationRecord[]} history
 * @property {number} convergenceStreak
 * @property {number} plateauCount
 * @property {number} activeMs              Time spent in 'running' during this run. Paused time excluded.
 * @property {null|'converged'|'plateau'|'max-iterations'|'time-limit'|'cancelled'} terminationReason
 */

/**
 * @typedef {Object} LoopEvent
 * @property {'start'|'iteration'|'pause'|'resume'|'terminate'|'reset'} type
 * @property {LoopState} state
 * @property {IterationRecord} [record]  Present on 'iteration' events.
 */

/**
 * @typedef {Object} LoopControllerOptions
 * @property {Generator} generator
 * @property {Discriminator} discriminator
 * @property {typeof LIMITS} [limits]                Defaults to LIMITS.
 * @property {() => number} [now]                    Millisecond clock. Defaults to performance.now.
 * @property {() => Promise<void>} [yieldToHost]     Awaited between iterations so the UI can paint.
 *                                                   Defaults to a 0 ms timeout.
 * @property {(evaluation: Evaluation, best: LoopState['best']) => boolean} [accept]
 *           Defaults to: evaluation.valid && (best === null || evaluation.overall > best.evaluation.overall).
 */

/**
 * Behaviour (normative):
 *
 * start()   From 'idle' or 'terminated': clears state, begins a new run and returns a promise that
 *           resolves with the final LoopState when that run terminates. From 'running' or 'paused':
 *           starts nothing and returns the promise of the run already in flight. There is never
 *           more than one loop executing.
 * pause()   'running' -> 'paused'. Active-time accounting stops at the call. Returns false otherwise.
 * resume()  'paused' -> 'running'. Active-time accounting restarts at the call. Returns false otherwise.
 * stop()    'running'|'paused' -> 'terminated' with reason 'cancelled', synchronously. State (best,
 *           history) is kept. No iteration completes after the call. Returns false otherwise.
 * reset()   Any status -> 'idle' with all state cleared, synchronously. A run in flight is cancelled
 *           first ('terminate' event, reason 'cancelled', then 'reset') and its promise resolves
 *           with that cancelled snapshot. A stale loop can never write into the new state.
 *
 * One iteration:
 *   1. candidate  = generator.generate({ iteration, parent: best?.candidate ?? null,
 *                                        feedback: best?.evaluation ?? null })
 *   2. evaluation = discriminator.evaluate({ candidate, previous: best?.candidate ?? null })
 *   3. bestBefore = bestScore. If accept(evaluation, best): best = { candidate, evaluation }.
 *      A rejected candidate never replaces best.
 *   4. Convergence streak: counted over accepted iterations only. Accepted with overall >=
 *      convergenceScore -> streak + 1; accepted below it -> streak = 0; rejected -> unchanged.
 *   5. Plateau: improvement = bestScore after - bestBefore. If improvement < plateauEpsilon ->
 *      plateauCount + 1, else plateauCount = 0.
 *   6. Emit 'iteration'.
 *   7. Check stopping conditions, first match wins:
 *        converged       convergenceStreak >= limits.convergenceStreak
 *        plateau         plateauCount      >= limits.plateauIterations
 *        max-iterations  iteration         >= limits.maxIterations
 *        time-limit      activeMs          >= limits.timeLimitMs
 *      If one matches: status 'terminated', emit 'terminate', resolve the run promise.
 *   8. Otherwise await yieldToHost(), then continue only if this run is still current and
 *      'running'. While 'paused' the loop waits without accruing active time.
 *   9. The same ordered check runs again immediately before the next iteration, because active
 *      time also accrues during the yield. The hard limits therefore hold strictly: no iteration
 *      starts once iteration >= maxIterations, once activeMs >= timeLimitMs, or after stop()/reset().
 *
 * Details:
 *   - Iteration 1 runs synchronously inside start(), so subscribe before calling start().
 *   - An evaluation with valid !== true or a non-finite overall is always rejected, whatever
 *     `accept` says, and is recorded with score 0.
 *   - If generate, evaluate, accept or yieldToHost throws, the run ends with status 'terminated'
 *     and terminationReason null, a 'terminate' event is emitted and the run promise rejects with
 *     the error. Callers must handle that rejection. Completed iterations are kept.
 *   - Do not call start() from a 'terminate' listener fired by reset().
 *
 * @typedef {Object} LoopController
 * @property {() => Promise<LoopState>} start
 * @property {() => boolean} pause
 * @property {() => boolean} resume
 * @property {() => boolean} stop
 * @property {() => void} reset
 * @property {() => LoopState} getState
 * @property {(listener: (event: LoopEvent) => void) => () => void} subscribe  Returns an unsubscribe function.
 */

/* ------------------------------------------------------------------ */
/* Renderer                                                           */
/* ------------------------------------------------------------------ */

/**
 * The renderer owns the canvas, its own requestAnimationFrame loop and pointer
 * handling, including zoom (wheel, pinch, double-click), pan (drag while zoomed
 * in), the floating tooltip on the hovered point and the word labels shown when
 * zoomed in. It knows nothing about the generator, the discriminator or the loop:
 * main.js pushes layouts into it.
 *
 * @typedef {Object} RendererOptions
 * @property {HTMLCanvasElement} canvas
 * @property {Dataset} dataset
 * @property {ReadonlyArray<ArrayLike<number>>} [neighbours]  Semantic neighbours per point
 *           (see core/knn.js), used to draw links around the hovered point.
 * @property {(index: number|null) => void} [onHover]         Called when the hovered point changes.
 * @property {(zoom: number) => void} [onViewChange]          Called when the zoom level changes (1 = whole layout).
 * @property {() => { top: number, right: number, bottom: number, left: number }} [getInsets]
 *           CSS-pixel margins the layout must stay clear of (the control panel). Read on every resize.
 */

/**
 * @typedef {Object} Renderer
 * @property {(positions: Float64Array|null, options?: { immediate?: boolean }) => void} setLayout
 *           Target layout the particles ease towards. null scatters them to their idle drift.
 *           The array is copied, never retained or mutated.
 * @property {(kind: 'accepted'|'rejected'|'terminated') => void} pulse  Brief visual accent.
 * @property {(factor: number) => void} zoomBy   Multiply the zoom about the centre of the layout area.
 * @property {() => void} resetView              Return to the unzoomed view.
 * @property {() => void} start     Begin the animation loop (idempotent).
 * @property {() => void} stop      Halt the animation loop (idempotent).
 * @property {() => void} resize    Re-measure the canvas; also called automatically on resize.
 * @property {() => void} destroy   Stop and remove every listener.
 */

/* ------------------------------------------------------------------ */
/* Validation helpers                                                 */
/* ------------------------------------------------------------------ */

/** @param {number} value */
export function clamp01(value) {
  if (!Number.isFinite(value)) return 0;
  return value < 0 ? 0 : value > 1 ? 1 : value;
}

/**
 * Lists everything wrong with a candidate layout. Empty array means valid.
 * @param {unknown} candidate
 * @param {number} pointCount
 * @returns {string[]}
 */
export function candidateIssues(candidate, pointCount) {
  const issues = [];
  if (candidate === null || typeof candidate !== 'object') return ['candidate is not an object'];
  const c = /** @type {CandidateLayout} */ (candidate);
  if (typeof c.id !== 'string' || c.id === '') issues.push('id must be a non-empty string');
  if (!Number.isInteger(c.iteration) || c.iteration < 1) issues.push('iteration must be an integer >= 1');
  if (!(c.positions instanceof Float64Array)) {
    issues.push('positions must be a Float64Array');
    return issues;
  }
  if (c.positions.length !== pointCount * 2) {
    issues.push(`positions must have length ${pointCount * 2}, got ${c.positions.length}`);
    return issues;
  }
  for (let i = 0; i < c.positions.length; i++) {
    const v = c.positions[i];
    if (!Number.isFinite(v) || v < LAYOUT_MIN || v > LAYOUT_MAX) {
      issues.push(`positions[${i}] is outside [0, 1]: ${v}`);
      break;
    }
  }
  return issues;
}

/**
 * Lists everything wrong with an evaluation. Empty array means well-formed.
 * @param {unknown} evaluation
 * @returns {string[]}
 */
export function evaluationIssues(evaluation) {
  if (evaluation === null || typeof evaluation !== 'object') return ['evaluation is not an object'];
  const e = /** @type {Evaluation} */ (evaluation);
  const issues = [];
  if (e.kind !== 'heuristic') issues.push("kind must be 'heuristic'");
  if (typeof e.valid !== 'boolean') issues.push('valid must be a boolean');
  if (!Array.isArray(e.issues)) issues.push('issues must be an array');
  const inUnit = (v) => typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 1;
  if (!inUnit(e.overall)) issues.push('overall must be a finite number in [0, 1]');
  for (const key of CRITERIA) {
    if (!e.criteria || !inUnit(e.criteria[key])) issues.push(`criteria.${key} must be a finite number in [0, 1]`);
    if (!e.weights || !inUnit(e.weights[key])) issues.push(`weights.${key} must be a finite number in [0, 1]`);
  }
  return issues;
}
