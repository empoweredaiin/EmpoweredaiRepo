/**
 * Pure state of the renderer's visual accents. Each kind has a single slot, so
 * triggering a pulse many times per second saturates instead of stacking up.
 */

/** Seconds the 'terminated' ring takes to play. */
export const TERMINATED_DURATION = 2.2;

/** Decay time constants in seconds. */
const ACCEPTED_TAU = 0.35;
const REJECTED_TAU = 0.16;
const SILENT_BELOW = 0.002;

/**
 * @typedef {Object} PulseState
 * @property {number} accepted       Energy in [0, 1], decaying to 0.
 * @property {number} rejected       Energy in [0, 1], decaying to 0.
 * @property {number} terminatedAge  Seconds since the ring started, or -1 when inactive.
 */

/** @returns {PulseState} */
export function createPulseState() {
  return { accepted: 0, rejected: 0, terminatedAge: -1 };
}

/**
 * Starts (or refreshes) an accent. 'accepted' and 'rejected' are reset to full
 * energy rather than added to; a 'terminated' ring already playing is left to
 * finish.
 * @param {PulseState} state
 * @param {'accepted'|'rejected'|'terminated'} kind
 * @returns {boolean} False when `kind` is not a known pulse.
 */
export function triggerPulse(state, kind) {
  if (kind === 'accepted') state.accepted = 1;
  else if (kind === 'rejected') state.rejected = 1;
  else if (kind === 'terminated') {
    if (state.terminatedAge < 0) state.terminatedAge = 0;
  } else return false;
  return true;
}

/**
 * Advances every accent by `dt` seconds.
 * @param {PulseState} state
 * @param {number} dt
 */
export function stepPulses(state, dt) {
  if (state.accepted > 0) {
    state.accepted *= Math.exp(-dt / ACCEPTED_TAU);
    if (state.accepted < SILENT_BELOW) state.accepted = 0;
  }
  if (state.rejected > 0) {
    state.rejected *= Math.exp(-dt / REJECTED_TAU);
    if (state.rejected < SILENT_BELOW) state.rejected = 0;
  }
  if (state.terminatedAge >= 0) {
    state.terminatedAge += dt;
    if (state.terminatedAge >= TERMINATED_DURATION) state.terminatedAge = -1;
  }
}

/**
 * Silences every accent at once.
 * @param {PulseState} state
 */
export function clearPulses(state) {
  state.accepted = 0;
  state.rejected = 0;
  state.terminatedAge = -1;
}
