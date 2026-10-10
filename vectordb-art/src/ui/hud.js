/**
 * The control panel. It renders LoopState snapshots and reports button presses;
 * it holds no application state of its own.
 *
 * Hierarchy, most to least prominent: the best score, the one action that makes
 * sense right now, why the run ended, then the score history and breakdown.
 * Only the buttons that apply to the current status are shown.
 */
import { CRITERIA, STATUS, TERMINATION } from '../contracts.js';

const SVG_NS = 'http://www.w3.org/2000/svg';
const CHART = { width: 268, height: 76, left: 0, right: 30, top: 8, bottom: 14 };
/** Scores below this are drawn on the chart's floor; real runs start around 0.6. */
const CHART_FLOOR = 0.4;

const STATUS_LABELS = {
  [STATUS.IDLE]: 'Idle',
  [STATUS.RUNNING]: 'Running',
  [STATUS.PAUSED]: 'Paused',
  [STATUS.TERMINATED]: 'Terminated',
};

/** Which buttons are shown in each status. The first one listed is the primary action. */
const VISIBLE = {
  [STATUS.IDLE]: ['start'],
  [STATUS.RUNNING]: ['pause', 'stop', 'reset'],
  [STATUS.PAUSED]: ['resume', 'stop', 'reset'],
  [STATUS.TERMINATED]: ['start', 'reset'],
};

/**
 * Explicit, human-readable termination reasons.
 * @param {string} reason
 * @param {typeof import('../contracts.js').LIMITS} limits
 * @returns {{ title: string, detail: string }}
 */
export function describeTermination(reason, limits) {
  switch (reason) {
    case TERMINATION.CONVERGED:
      return {
        title: 'Converged',
        detail: `Score stayed at or above ${limits.convergenceScore.toFixed(2)} for ${limits.convergenceStreak} consecutive accepted iterations.`,
      };
    case TERMINATION.PLATEAU:
      return {
        title: 'Plateau',
        detail: `Best score improved by less than ${limits.plateauEpsilon} for ${limits.plateauIterations} consecutive iterations.`,
      };
    case TERMINATION.MAX_ITERATIONS:
      return { title: 'Iteration limit', detail: `Reached the maximum of ${limits.maxIterations} attempted iterations.` };
    case TERMINATION.TIME_LIMIT:
      return { title: 'Time limit', detail: `Reached ${limits.timeLimitMs / 1000} seconds of active execution.` };
    case TERMINATION.CANCELLED:
      return { title: 'Stopped', detail: 'Cancelled by the user.' };
    default:
      return { title: 'Terminated', detail: 'The run ended because of an error.' };
  }
}

const score = (value) => value.toFixed(3);

function svg(name, attributes = {}) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  return node;
}

/**
 * @param {Object} options
 * @param {Document} options.document
 * @param {import('../contracts.js').Dataset} options.dataset
 * @param {typeof import('../contracts.js').LIMITS} options.limits
 * @param {Record<string, number>} options.weights
 * @param {Record<string, string>} options.criterionDescriptions
 * @param {string[]} options.clusterColors  CSS colour per cluster index.
 * @param {number} options.maxZoom          Largest zoom factor the canvas allows.
 * @param {{ start: () => void, pause: () => void, resume: () => void, stop: () => void, reset: () => void,
 *           zoomIn: () => void, zoomOut: () => void, zoomReset: () => void }} options.handlers
 */
export function createHud({ document: doc, dataset, limits, weights, criterionDescriptions, clusterColors, maxZoom, handlers }) {
  const $ = (id) => doc.getElementById(id);
  const buttons = {
    start: $('btn-start'),
    pause: $('btn-pause'),
    resume: $('btn-resume'),
    stop: $('btn-stop'),
    reset: $('btn-reset'),
  };
  const seedInput = $('seed');
  const chartBox = $('chart');
  const chartSvg = $('chart-svg');
  const tooltip = $('chart-tooltip');
  let history = [];

  for (const [name, button] of Object.entries(buttons)) button.addEventListener('click', () => handlers[name]());
  $('zoom-in').addEventListener('click', () => handlers.zoomIn());
  $('zoom-out').addEventListener('click', () => handlers.zoomOut());
  $('zoom-reset').addEventListener('click', () => handlers.zoomReset());
  $('iteration-max').textContent = String(limits.maxIterations);
  $('target-score').textContent = limits.convergenceScore.toFixed(2);

  /* ----- score breakdown ----- */
  const criterionRows = {};
  for (const key of CRITERIA) {
    const item = doc.createElement('li');
    item.title = `${criterionDescriptions[key] ?? ''} Weight ${weights[key]}.`.trim();
    item.innerHTML = `
      <div class="criterion__head"><span class="criterion__name">${key}</span><span class="criterion__value">—</span></div>
      <div class="criterion__track"><div class="criterion__fill"></div></div>`;
    $('criteria').append(item);
    criterionRows[key] = { value: item.querySelector('.criterion__value'), fill: item.querySelector('.criterion__fill') };
  }

  /* ----- topic legend ----- */
  dataset.clusterNames.forEach((name, index) => {
    const item = doc.createElement('li');
    const swatch = doc.createElement('span');
    swatch.className = 'swatch';
    swatch.style.background = clusterColors[index];
    item.append(swatch, name);
    $('topics').append(item);
  });

  /* ----- score history ----- */
  const plotRight = CHART.width - CHART.right;
  const plotWidth = plotRight - CHART.left;
  const plotBottom = CHART.height - CHART.bottom;
  const plotHeight = plotBottom - CHART.top;
  // The x axis grows in steps of 25 iterations so a short run is not squeezed into a corner.
  let xMax = Math.min(25, limits.maxIterations);
  const x = (iteration) => CHART.left + ((iteration - 1) / Math.max(1, xMax - 1)) * plotWidth;
  const y = (value) => CHART.top + (1 - (Math.max(CHART_FLOOR, Math.min(1, value)) - CHART_FLOOR) / (1 - CHART_FLOOR)) * plotHeight;

  chartSvg.setAttribute('viewBox', `0 0 ${CHART.width} ${CHART.height}`);
  const targetY = y(limits.convergenceScore);
  const targetLabel = svg('text', { class: 'axis', x: CHART.width, y: targetY + 3, 'text-anchor': 'end' });
  targetLabel.textContent = limits.convergenceScore.toFixed(2);
  const firstLabel = svg('text', { class: 'axis', x: CHART.left, y: CHART.height - 2, 'text-anchor': 'start' });
  firstLabel.textContent = '1';
  const lastLabel = svg('text', { class: 'axis', x: plotRight, y: CHART.height - 2, 'text-anchor': 'end' });
  const cursor = svg('line', { class: 'cursor', y1: CHART.top, y2: plotBottom, visibility: 'hidden' });
  const dots = svg('g');
  const bestPath = svg('path', { class: 'best' });
  chartSvg.append(
    svg('line', { class: 'baseline', x1: CHART.left, x2: plotRight, y1: plotBottom, y2: plotBottom }),
    svg('line', { class: 'target', x1: CHART.left, x2: plotRight, y1: targetY, y2: targetY }),
    targetLabel,
    firstLabel,
    lastLabel,
    cursor,
    dots,
    bestPath,
  );

  function drawChart() {
    xMax = Math.min(limits.maxIterations, Math.max(25, Math.ceil(history.length / 25) * 25));
    lastLabel.textContent = `${xMax} iterations`;
    dots.replaceChildren(
      ...history.map((record) =>
        svg('circle', {
          class: record.accepted ? 'dot--accepted' : 'dot--rejected',
          cx: x(record.iteration),
          cy: y(record.score),
          r: record.accepted ? 1.8 : 1.6,
        }),
      ),
    );
    // Step line: the best score holds until an iteration improves it.
    let d = '';
    history.forEach((record, index) => {
      const px = x(record.iteration).toFixed(2);
      const py = y(record.bestScore).toFixed(2);
      d += index === 0 ? `M${px} ${py}` : `H${px}V${py}`;
    });
    bestPath.setAttribute('d', d);
  }

  function hideTooltip() {
    tooltip.hidden = true;
    cursor.setAttribute('visibility', 'hidden');
  }

  chartBox.addEventListener('pointermove', (event) => {
    if (history.length === 0) return hideTooltip();
    const box = chartSvg.getBoundingClientRect();
    const scale = Math.min(box.width / CHART.width, box.height / CHART.height);
    const offsetX = (box.width - CHART.width * scale) / 2;
    const chartX = (event.clientX - box.left - offsetX) / scale;
    const iteration = 1 + ((chartX - CHART.left) / plotWidth) * Math.max(1, xMax - 1);
    const record = history[Math.max(0, Math.min(history.length - 1, Math.round(iteration) - 1))];
    cursor.setAttribute('x1', x(record.iteration));
    cursor.setAttribute('x2', x(record.iteration));
    cursor.setAttribute('visibility', 'visible');
    tooltip.textContent = `#${record.iteration} · ${score(record.score)} ${record.accepted ? 'kept' : 'rejected'} · best ${score(record.bestScore)}`;
    tooltip.hidden = false;
    const left = offsetX + x(record.iteration) * scale;
    tooltip.style.left = `${Math.max(0, Math.min(box.width - tooltip.offsetWidth, left - tooltip.offsetWidth / 2))}px`;
  });
  chartBox.addEventListener('pointerleave', hideTooltip);

  return {
    /** @returns {number|null} The seed in the input, or null when it is not a usable integer. */
    getSeed() {
      const value = Number(seedInput.value);
      return seedInput.value.trim() !== '' && Number.isInteger(value) && value >= 0 ? value : null;
    },

    /** @param {string|null} message */
    setError(message) {
      $('error').hidden = !message;
      $('error').textContent = message ?? '';
    },

    /**
     * Announces the hovered particle to assistive technology; sighted users get the canvas tooltip.
     * @param {import('../contracts.js').SourcePoint|null} point
     */
    setHover(point) {
      $('hovered').textContent = point ? `${point.label}, topic ${dataset.clusterNames[point.cluster]}` : '';
    },

    /** @param {number} zoom Current zoom factor, 1 = the whole layout. */
    setZoom(zoom) {
      $('zoom-reset').textContent = `${Math.round(zoom * 100)}%`;
      $('zoom-reset').disabled = zoom <= 1;
      $('zoom-out').disabled = zoom <= 1;
      $('zoom-in').disabled = zoom >= maxZoom;
    },

    /** @param {import('../contracts.js').LoopState} state */
    update(state) {
      const pill = $('status-pill');
      pill.dataset.status = state.status;
      pill.textContent = STATUS_LABELS[state.status];
      for (const [name, button] of Object.entries(buttons)) button.hidden = !VISIBLE[state.status].includes(name);
      seedInput.disabled = state.status === STATUS.RUNNING || state.status === STATUS.PAUSED;

      const started = state.status !== STATUS.IDLE;
      $('intro').hidden = started;
      $('progress').hidden = !started;
      $('best-score').textContent = state.best ? score(state.bestScore) : '—';
      // The score and the detail sections appear once there is something to show in them.
      for (const node of doc.querySelectorAll('.hero__label, .hero__value, .block')) node.hidden = !started;
      $('iteration').textContent = String(state.iteration);
      $('accepted').textContent = String(state.acceptedCount);
      $('rejected').textContent = String(state.rejectedCount);
      $('active-time').textContent = (state.activeMs / 1000).toFixed(1);

      const reason = $('reason');
      reason.hidden = state.status !== STATUS.TERMINATED;
      if (state.status === STATUS.TERMINATED) {
        const { title, detail } = describeTermination(state.terminationReason, limits);
        const strong = doc.createElement('strong');
        strong.textContent = `Ended: ${title}`;
        reason.replaceChildren(strong, detail);
      }

      for (const key of CRITERIA) {
        const best = state.best?.evaluation.criteria[key];
        criterionRows[key].value.textContent = best === undefined ? '—' : best.toFixed(2);
        criterionRows[key].fill.style.width = `${(best ?? 0) * 100}%`;
      }

      history = state.history;
      drawChart();
      if (history.length === 0) hideTooltip();
    },
  };
}
