/**
 * The interface chrome: top bar, readout rail, inspector and view controls. It
 * renders LoopState snapshots and reports button presses. The only state it
 * keeps is presentational: whether the inspector is open.
 *
 * The rail carries what matters at a glance (the one applicable action, best
 * score, iteration, score history, how the run ended). Everything else lives
 * in the inspector, which opens on demand, when a run ends, or when a mark is
 * selected.
 */
import { CRITERIA, STATUS, TERMINATION } from '../contracts.js';

const SVG_NS = 'http://www.w3.org/2000/svg';
const CHART = { width: 232, height: 48, left: 0, right: 30, top: 6, bottom: 4 };
/** Scores below this are drawn on the chart's floor; real runs start around 0.6. */
const CHART_FLOOR = 0.4;
const NARROW_QUERY = '(max-width: 860px)';

const STATUS_LABELS = {
  [STATUS.IDLE]: 'Idle',
  [STATUS.RUNNING]: 'Running',
  [STATUS.PAUSED]: 'Paused',
  [STATUS.TERMINATED]: 'Terminated',
};

/** Short display names for the criteria; the full definition is in each item's tooltip. */
const CRITERION_NAMES = { neighbourhood: 'Neighbours', separation: 'Separation', overlap: 'Overlap', stability: 'Stability' };

/** Which buttons are shown in each status. The first one listed is the primary action. */
const VISIBLE = {
  [STATUS.IDLE]: ['start'],
  [STATUS.RUNNING]: ['pause', 'stop', 'reset'],
  [STATUS.PAUSED]: ['resume', 'stop', 'reset'],
  [STATUS.TERMINATED]: ['start', 'reset'],
};

/** One 12×12 glyph per way a run can end, drawn with the current text colour. */
const REASON_GLYPHS = {
  [TERMINATION.CONVERGED]: '<path d="M1.5 6.5 L4.8 9.8 L10.5 2.5"/>',
  [TERMINATION.PLATEAU]: '<path d="M1 10 L4.5 4 H11"/>',
  [TERMINATION.MAX_ITERATIONS]: '<path d="M2 10.5V1.5 M5 10.5V1.5 M8 10.5V1.5 M10.5 1.5 L0.5 10.5"/>',
  [TERMINATION.TIME_LIMIT]: '<circle cx="6" cy="6" r="4.6"/><path d="M6 3.2V6 L8 7.4"/>',
  [TERMINATION.CANCELLED]: '<path d="M2.5 2.5h7v7h-7z"/>',
};

/**
 * Explicit, human-readable termination reasons. Only TERMINATION.CONVERGED is a
 * convergence: a plateau, a limit or a cancellation is reported as what it is.
 * @param {string} reason
 * @param {typeof import('../contracts.js').LIMITS} limits
 * @returns {{ title: string, detail: string, converged: boolean }}
 */
export function describeTermination(reason, limits) {
  return { ...terminationText(reason, limits), converged: reason === TERMINATION.CONVERGED };
}

function terminationText(reason, limits) {
  switch (reason) {
    case TERMINATION.CONVERGED:
      return {
        title: 'Converged',
        detail: `Score stayed at or above ${limits.convergenceScore.toFixed(2)} for ${limits.convergenceStreak} consecutive accepted iterations.`,
      };
    case TERMINATION.PLATEAU:
      return {
        title: 'Plateau, not converged',
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
 * @param {string[]} options.clusterMarks   SVG markup of each cluster's shape, for legends and lists.
 * @param {number} options.maxZoom          Largest zoom factor the canvas allows.
 * @param {{ start: () => void, pause: () => void, resume: () => void, stop: () => void, reset: () => void,
 *           zoomIn: () => void, zoomOut: () => void, zoomReset: () => void }} options.handlers
 */
export function createHud({ document: doc, dataset, limits, weights, criterionDescriptions, clusterMarks, maxZoom, handlers }) {
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
  const inspector = $('inspector');
  const inspectorToggle = $('inspector-toggle');
  let history = [];
  let lastStatus = STATUS.IDLE;
  /** True once the user has closed the inspector by hand; it then stops opening by itself. */
  let dismissed = false;

  for (const [name, button] of Object.entries(buttons)) button.addEventListener('click', () => handlers[name]());
  $('zoom-in').addEventListener('click', () => handlers.zoomIn());
  $('zoom-out').addEventListener('click', () => handlers.zoomOut());
  $('zoom-reset').addEventListener('click', () => handlers.zoomReset());
  $('iteration-max').textContent = String(limits.maxIterations);
  $('target-score').textContent = limits.convergenceScore.toFixed(2);
  $('dataset-info').textContent =
    `${dataset.points.length} vectors · ${dataset.dim}-d · ${dataset.clusterNames.length} topics · synthetic`;

  /* ----- inspector ----- */
  function setInspector(open) {
    inspector.hidden = !open;
    inspectorToggle.setAttribute('aria-expanded', String(open));
  }
  /**
   * Opens the inspector because something relevant appeared, unless the user has dismissed it.
   * On narrow screens it never opens by itself: it would take the canvas's space.
   */
  function revealInspector() {
    if (!dismissed && !doc.defaultView.matchMedia(NARROW_QUERY).matches) setInspector(true);
  }
  inspectorToggle.addEventListener('click', () => {
    const open = inspector.hidden;
    dismissed = !open;
    setInspector(open);
  });

  /* ----- score breakdown ----- */
  const criterionRows = {};
  for (const key of CRITERIA) {
    const item = doc.createElement('li');
    item.title = criterionDescriptions[key] ?? '';
    item.innerHTML = `
      <div class="criterion__head">
        <span><span class="criterion__name">${CRITERION_NAMES[key] ?? key}</span><span class="criterion__weight">×${weights[key]}</span></span>
        <span class="criterion__value">—</span>
      </div>
      <div class="criterion__track"><div class="criterion__fill"></div><div class="criterion__tick" hidden></div></div>`;
    $('criteria').append(item);
    criterionRows[key] = {
      value: item.querySelector('.criterion__value'),
      fill: item.querySelector('.criterion__fill'),
      tick: item.querySelector('.criterion__tick'),
    };
  }

  /* ----- topic legend ----- */
  dataset.clusterNames.forEach((name, index) => {
    const item = doc.createElement('li');
    item.innerHTML = clusterMarks[index] ?? '';
    item.append(name);
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
  const cursor = svg('line', { class: 'cursor', y1: CHART.top, y2: plotBottom, visibility: 'hidden' });
  const dots = svg('g');
  const bestPath = svg('path', { class: 'best' });
  chartSvg.append(
    svg('line', { class: 'baseline', x1: CHART.left, x2: plotRight, y1: plotBottom, y2: plotBottom }),
    svg('line', { class: 'target', x1: CHART.left, x2: plotRight, y1: targetY, y2: targetY }),
    targetLabel,
    cursor,
    dots,
    bestPath,
  );

  function drawChart() {
    xMax = Math.min(limits.maxIterations, Math.max(25, Math.ceil(history.length / 25) * 25));
    dots.replaceChildren(
      ...history.map((record) =>
        svg('circle', {
          class: record.accepted ? 'dot--accepted' : 'dot--rejected',
          cx: x(record.iteration),
          cy: y(record.score),
          r: record.accepted ? 1.5 : 1.4,
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

  /** Fills the block that says how the run ended, with its own glyph and treatment per reason. */
  function renderReason(state) {
    const box = $('reason');
    box.hidden = state.status !== STATUS.TERMINATED;
    if (box.hidden) return;
    const { title, detail, converged } = describeTermination(state.terminationReason, limits);
    const target = limits.convergenceScore.toFixed(2);
    const outcome = converged
      ? ` Target ${target} reached.`
      : state.best
        ? ` Target ${target} not reached; best ${score(state.bestScore)}.`
        : ` Target ${target} not reached.`;
    box.dataset.reason = state.terminationReason ?? 'error';
    box.dataset.outcome = converged ? 'converged' : 'not-converged';
    box.innerHTML =
      `<svg class="reason__glyph" viewBox="0 0 12 12" width="12" height="12" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.4">${REASON_GLYPHS[state.terminationReason] ?? ''}</svg>` +
      '<strong class="reason__title"></strong><span class="reason__detail"></span>';
    box.querySelector('.reason__title').textContent = `Ended: ${title}`;
    box.querySelector('.reason__detail').textContent = detail + outcome;
  }

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

    /**
     * The step the loop is on: a candidate proposed and being evaluated, then kept or rejected.
     * @param {null | { kind: 'evaluating', iteration: number }
     *              | { kind: 'accepted'|'rejected', record: import('../contracts.js').IterationRecord }} phase
     */
    setPhase(phase) {
      const line = $('phase');
      line.hidden = !phase;
      if (!phase) return;
      line.dataset.kind = phase.kind;
      if (phase.kind === 'evaluating') line.textContent = `Candidate ${phase.iteration} · evaluating`;
      else if (phase.kind === 'accepted') line.textContent = `Candidate ${phase.record.iteration} · ${score(phase.record.score)} · kept`;
      else line.textContent = `Candidate ${phase.record.iteration} · ${score(phase.record.score)} · rejected, best kept`;
    },

    /**
     * Describes the selected mark and its revealed neighbours: in the inspector, in the hint line
     * and for assistive technology.
     * @param {import('../contracts.js').SourcePoint|null} point
     * @param {{ point: import('../contracts.js').SourcePoint, similarity: number }[]} [nearest]
     */
    setSelection(point, nearest = []) {
      $('selection-section').hidden = !point;
      if (!point) {
        $('view-hint').textContent = 'Click a mark for its nearest neighbours · scroll to zoom · drag to pan';
        $('selected').textContent = '';
        return;
      }
      const topic = dataset.clusterNames[point.cluster];
      $('view-hint').textContent = `${point.label} · ${nearest.length} nearest by cosine similarity · synthetic vectors · Esc to clear`;
      $('selected').textContent =
        `${point.label} selected, topic ${topic}. Nearest by cosine similarity: ` +
        nearest.map((entry) => `${entry.point.label} ${entry.similarity.toFixed(3)}`).join(', ') +
        '. The vectors are synthetic.';

      const title = $('selection-title');
      title.innerHTML = clusterMarks[point.cluster] ?? '';
      const small = doc.createElement('small');
      small.textContent = topic;
      title.append(point.label, small);
      $('selection-list').replaceChildren(
        ...nearest.map((entry) => {
          const row = doc.createElement('li');
          row.innerHTML = clusterMarks[entry.point.cluster] ?? '<span></span>';
          const word = doc.createElement('span');
          word.textContent = entry.point.label;
          const value = doc.createElement('span');
          value.className = 'num';
          value.textContent = entry.similarity.toFixed(3);
          row.append(word, value);
          return row;
        }),
      );
      revealInspector();
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
      pill.textContent =
        state.status === STATUS.TERMINATED
          ? `Ended · ${describeTermination(state.terminationReason, limits).title.split(',')[0]}`
          : STATUS_LABELS[state.status];
      for (const [name, button] of Object.entries(buttons)) button.hidden = !VISIBLE[state.status].includes(name);
      seedInput.disabled = state.status === STATUS.RUNNING || state.status === STATUS.PAUSED;

      // The readout appears once there is something to read.
      const started = state.status !== STATUS.IDLE;
      $('intro').hidden = started;
      $('readout').hidden = !started;
      $('history').hidden = !started;
      $('best-score').textContent = state.best ? score(state.bestScore) : '—';
      $('iteration').textContent = String(state.iteration);
      $('accepted').textContent = String(state.acceptedCount);
      $('rejected').textContent = String(state.rejectedCount);
      $('active-time').textContent = (state.activeMs / 1000).toFixed(1);

      renderReason(state);
      if (state.status === STATUS.TERMINATED && lastStatus !== STATUS.TERMINATED) revealInspector();
      lastStatus = state.status;

      for (const key of CRITERIA) {
        const best = state.best?.evaluation.criteria[key];
        criterionRows[key].value.textContent = best === undefined ? '—' : best.toFixed(2);
        criterionRows[key].fill.style.width = `${(best ?? 0) * 100}%`;
        // The tick marks where the latest candidate scored; the bar is the best layout.
        const last = state.last?.evaluation.criteria[key];
        const tick = criterionRows[key].tick;
        tick.hidden = last === undefined;
        if (last !== undefined) {
          tick.style.left = `${last * 100}%`;
          tick.dataset.accepted = String(state.last.accepted);
        }
      }

      history = state.history;
      drawChart();
      if (history.length === 0) hideTooltip();
    },
  };
}
