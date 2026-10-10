/**
 * Canvas renderer: one flat geometric mark per dataset point over a drafting
 * grid. Background hairlines between near neighbours stay very quiet; clicking
 * a mark selects it and draws clear connections to its nearest neighbours. It owns the canvas, its requestAnimationFrame loop and pointer
 * handling; all simulation maths lives in particles.js and pulses.js.
 *
 * It knows nothing about generators, scores or iterations: layouts are pushed
 * in through setLayout().
 */
import {
  clampCamera,
  createCamera,
  panBy,
  resetCamera,
  snapCamera,
  stepCamera,
  toScreenX,
  toScreenY,
  toWorldX,
  toWorldY,
  zoomAt,
} from './camera.js';
import { POINT_RADIUS } from '../contracts.js';
import {
  BACKGROUND,
  DEFAULT_DT,
  clampDt,
  clusterColour,
  computeViewport,
  createFrame,
  createParticles,
  hitTest,
  setTargets,
  snapParticles,
  stepParticles,
} from './particles.js';
import { TERMINATED_DURATION, clearPulses, createPulseState, stepPulses, triggerPulse } from './pulses.js';
import { STROKE_RATIO, isStroked, shapeOf, traceShape } from './shapes.js';

const MAX_PIXEL_RATIO = 2;
/** Each point is joined to this many of its nearest semantic neighbours. */
const EDGE_NEIGHBOURS = 3;
/** Edge opacity by length, as a fraction of the layout size: short edges read, long ones recede. */
const EDGE_BANDS = Object.freeze([
  Object.freeze({ maxLength: 0.07, alpha: 0.13 }),
  Object.freeze({ maxLength: 0.22, alpha: 0.04 }),
  Object.freeze({ maxLength: Infinity, alpha: 0.012 }),
]);
/** How many nearest neighbours a selected mark reveals. */
const DEFAULT_SELECTION_SIZE = 5;
const SELECT_EASE_RATE = 9;
/** Seconds a rejected candidate's overlay takes to fade. */
const GHOST_FADE_TIME = 0.35;
/** Decorative sway, in screen pixels and radians per second. It never touches simulation or layout coordinates. */
const DRIFT_AMPLITUDE = 1.2;
const DRIFT_RATE = 0.32;
const GRID_DIVISIONS = 8;
const HIT_RADIUS_MOUSE = 22;
const HIT_RADIUS_TOUCH = 32;
const FOCUS_EASE_RATE = 12;
const REDUCED_MOTION_QUERY = '(prefers-reduced-motion: reduce)';
/** Word labels fade in between these zoom levels. */
const LABEL_ZOOM_START = 2.4;
const LABEL_ZOOM_FULL = 3.4;
const LABEL_FONT = '600 12px "Barlow Condensed", "Arial Narrow", sans-serif';
const TOOLTIP_FONT = '700 16px "Barlow Condensed", "Arial Narrow", sans-serif';
const TOOLTIP_NOTE_FONT = '400 12px "Barlow", system-ui, sans-serif';
const TOOLTIP_VALUE_FONT = '400 12px "IBM Plex Mono", ui-monospace, monospace';
const LABEL_TRACKING = '0.06em';
const INK = '#e8e6e1';
const INK_MUTED = '#9a9ba0';
const ACCENT = '#e8603d';
const SURFACE = '#1e1f23';
/** Above this zoom the pointer reads particles instead of pushing them. */
const PUSH_MAX_ZOOM = 1.15;
const WHEEL_ZOOM_RATE = 0.0016;
const PINCH_WHEEL_ZOOM_RATE = 0.012;
const DOUBLE_CLICK_ZOOM = 3;
/** A press that moves less than this many pixels is a tap, not a drag. */
const DRAG_THRESHOLD = 4;

/**
 * Creates the renderer for a dataset.
 *
 * Beyond the contract's RendererOptions, two optional fields are honoured:
 * `insets` (static) and `getInsets` (re-read on every resize) describe the
 * canvas edges covered by overlays, in CSS pixels; the layout is kept clear of
 * them. `getInsets` wins when both are given.
 *
 * The view can be zoomed (wheel, pinch, double-click, zoomBy) and panned (drag
 * while zoomed in). Hovering a mark brackets it and shows one tooltip with its
 * word and topic. Clicking or tapping a mark selects it: its nearest neighbours
 * (the first `selectionSize` entries of `neighbours[index]`, which the caller
 * supplies in order of real vector similarity) are joined to it and everything
 * else recedes. Hovering one of those neighbours adds `similarity(selected,
 * hovered)` to the tooltip. Zoomed in far enough, every mark is labelled.
 *
 * The renderer never computes similarity or chooses neighbours itself.
 *
 * @param {import('../contracts.js').RendererOptions & {
 *   insets?: import('./particles.js').Insets,
 *   getInsets?: () => import('./particles.js').Insets | null | undefined,
 * }} options
 * @returns {import('../contracts.js').Renderer}
 */
export function createRenderer(options) {
  const {
    canvas,
    dataset,
    neighbours = null,
    similarity = null,
    selectionSize = DEFAULT_SELECTION_SIZE,
    onHover = null,
    onSelect = null,
    onViewChange = null,
    insets = null,
    getInsets = null,
  } = options;
  const doc = canvas.ownerDocument;
  const win = doc.defaultView;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('createRenderer: 2D canvas context is not available');

  const count = dataset.points.length;
  const particles = createParticles({ count, seed: dataset.seed });
  const pulses = createPulseState();
  const frame = createFrame();
  const viewport = frame.viewport;

  /* Per-cluster drawing resources. */
  const clusterOf = new Uint8Array(count);
  let clusterCount = 1;
  for (let i = 0; i < count; i++) {
    clusterOf[i] = dataset.points[i].cluster;
    if (clusterOf[i] + 1 > clusterCount) clusterCount = clusterOf[i] + 1;
  }
  const strokeStyles = [];
  const shapes = [];
  for (let c = 0; c < clusterCount; c++) {
    const [r, g, b] = clusterColour(c);
    strokeStyles.push(`rgb(${r}, ${g}, ${b})`);
    shapes.push(shapeOf(c));
  }
  const labels = dataset.points.map((point) => point.label.toUpperCase());

  /* Each pair of near neighbours once, grouped by the cluster of its first point: [i0, j0, i1, j1, ...]. */
  const edges = Array.from({ length: clusterCount }, () => []);
  if (neighbours) {
    const seen = new Set();
    for (let i = 0; i < count; i++) {
      const list = neighbours[i];
      for (let k = 0; list && k < Math.min(EDGE_NEIGHBOURS, list.length); k++) {
        const j = list[k];
        if (!(j >= 0 && j < count) || j === i) continue;
        const key = i < j ? i * count + j : j * count + i;
        if (seen.has(key)) continue;
        seen.add(key);
        edges[clusterOf[i]].push(i, j);
      }
    }
  }

  /** Role of each point in the current selection: 0 unrelated, 1 neighbour, 2 the selected point. */
  const related = new Uint8Array(count);
  /** Where each mark is drawn this frame: simulation position plus decorative drift. */
  const drawn = { x: new Float32Array(count), y: new Float32Array(count) };
  /** Proposed candidate layout being shown, in layout (unit square) coordinates. */
  const ghost = new Float64Array(count * 2);
  /** Pointer position in screen (CSS pixel) coordinates. */
  const pointer = { x: 0, y: 0, radius: HIT_RADIUS_MOUSE, inside: false };
  const camera = createCamera();
  /** Pointers currently pressed, by pointerId, for dragging and pinching. */
  const pressed = new Map();
  let dragDistance = 0;
  let reportedZoom = 1;
  const motionQuery = typeof win.matchMedia === 'function' ? win.matchMedia(REDUCED_MOTION_QUERY) : null;

  let cssWidth = 0;
  let cssHeight = 0;
  let pixelRatio = 1;
  let coreRadius = 2;
  let needsSnap = true;
  let running = false;
  let destroyed = false;
  let rafId = 0;
  let lastTime = -1;
  /** @type {number|null} */
  let hovered = null;
  /** Last hovered point; kept while its highlight fades out. */
  let focus = -1;
  let focusMix = 0;
  let selected = -1;
  /** @type {number[]} */
  let selectedNeighbours = [];
  let selectMix = 0;
  /** 'none' | 'proposed' | 'rejected' */
  let ghostState = 'none';
  let ghostMix = 0;

  frame.reducedMotion = motionQuery ? motionQuery.matches : false;

  /* ---------------------------------------------------------------- */
  /* Hover                                                            */
  /* ---------------------------------------------------------------- */

  /** @param {number|null} index */
  function setHovered(index) {
    if (index === hovered) return;
    hovered = index;
    frame.exempt = index ?? -1;
    if (index !== null) focus = index;
    if (onHover) onHover(index);
  }

  /**
   * Selects a point, or clears the selection with null. Selecting the point
   * that is already selected clears it.
   * @param {number|null} index
   */
  function select(index) {
    if (destroyed) return;
    const next = index === null || index === selected || !(index >= 0 && index < count) ? -1 : index;
    if (next === selected) return;
    selected = next;
    related.fill(0);
    selectedNeighbours = [];
    if (next >= 0) {
      related[next] = 2;
      const list = neighbours ? neighbours[next] : null;
      for (let k = 0; list && k < list.length && selectedNeighbours.length < selectionSize; k++) {
        const j = list[k];
        if (!(j >= 0 && j < count) || j === next) continue;
        related[j] = 1;
        selectedNeighbours.push(j);
      }
      // Moving the selection eases from part-way, so focus visibly transfers instead of snapping.
      selectMix = Math.min(selectMix, 0.35);
    }
    if (onSelect) onSelect(next < 0 ? null : next, selectedNeighbours.slice());
    if (rafId === 0) draw();
  }

  /** @param {KeyboardEvent} event */
  function onKeyDown(event) {
    if (event.key === 'Escape' && selected >= 0) select(null);
  }

  function updateHover() {
    const dragging = pressed.size > 0 && dragDistance > DRAG_THRESHOLD;
    frame.pointerX = toWorldX(camera, pointer.x);
    frame.pointerY = toWorldY(camera, pointer.y);
    frame.pointerActive = pointer.inside && !dragging && camera.zoom < PUSH_MAX_ZOOM;
    if (dragging || !pointer.inside) return;
    setHovered(hitTest(particles, frame.pointerX, frame.pointerY, pointer.radius / camera.zoom));
  }

  /* ---------------------------------------------------------------- */
  /* View                                                             */
  /* ---------------------------------------------------------------- */

  /**
   * Applies a change made to the camera target and tells the host about the zoom level.
   * @param {boolean} immediate
   */
  function viewChanged(immediate) {
    if (immediate || frame.reducedMotion) snapCamera(camera);
    if (onViewChange && camera.targetZoom !== reportedZoom) {
      reportedZoom = camera.targetZoom;
      onViewChange(reportedZoom);
    }
    if (rafId === 0) draw();
  }

  /** @param {number} factor Multiplies the zoom, about the centre of the layout area. */
  function zoomBy(factor) {
    if (destroyed || !(factor > 0)) return;
    zoomAt(camera, viewport.x + viewport.size / 2, viewport.y + viewport.size / 2, factor, cssWidth, cssHeight);
    viewChanged(false);
  }

  function resetView() {
    if (destroyed) return;
    resetCamera(camera);
    viewChanged(false);
  }

  /** @param {WheelEvent} event */
  function onWheel(event) {
    event.preventDefault();
    // Trackpad pinches arrive as ctrl+wheel with small deltas; line-mode wheels report lines.
    const rate = event.ctrlKey ? PINCH_WHEEL_ZOOM_RATE : WHEEL_ZOOM_RATE;
    const delta = event.deltaMode === 1 ? event.deltaY * 32 : event.deltaY;
    zoomAt(camera, event.offsetX, event.offsetY, Math.exp(-delta * rate), cssWidth, cssHeight);
    viewChanged(false);
  }

  /** @param {MouseEvent} event */
  function onDoubleClick(event) {
    if (camera.targetZoom > 1.05) resetCamera(camera);
    else zoomAt(camera, event.offsetX, event.offsetY, DOUBLE_CLICK_ZOOM, cssWidth, cssHeight);
    viewChanged(false);
  }

  /** Distance and midpoint of the first two pressed pointers. */
  function pinchState() {
    const [p, q] = [...pressed.values()];
    return { distance: Math.hypot(p.x - q.x, p.y - q.y), x: (p.x + q.x) / 2, y: (p.y + q.y) / 2 };
  }

  /** @param {PointerEvent} event */
  function onPointerDown(event) {
    if (pressed.size === 0) dragDistance = 0;
    pressed.set(event.pointerId, { x: event.offsetX, y: event.offsetY });
    try {
      canvas.setPointerCapture?.(event.pointerId);
    } catch {
      // Capture is a convenience; a synthetic or already released pointer cannot be captured.
    }
    onPointerMove(event);
  }

  /**
   * A press released without dragging is a click: it selects the mark under it, or clears.
   * @param {PointerEvent} event
   */
  function onPointerUp(event) {
    const wasPressed = pressed.delete(event.pointerId);
    if (!wasPressed || pressed.size > 0 || dragDistance > DRAG_THRESHOLD) return;
    const radius = (event.pointerType === 'touch' ? HIT_RADIUS_TOUCH : HIT_RADIUS_MOUSE) / camera.zoom;
    const hit = hitTest(particles, toWorldX(camera, event.offsetX), toWorldY(camera, event.offsetY), radius);
    if (hit === null) select(null);
    else select(hit);
  }

  /** @param {PointerEvent} event */
  function onPointerMove(event) {
    const held = pressed.get(event.pointerId);
    if (held) {
      const dx = event.offsetX - held.x;
      const dy = event.offsetY - held.y;
      dragDistance += Math.hypot(dx, dy);
      if (pressed.size >= 2) {
        const before = pinchState();
        held.x = event.offsetX;
        held.y = event.offsetY;
        const after = pinchState();
        if (before.distance > 0) zoomAt(camera, after.x, after.y, after.distance / before.distance, cssWidth, cssHeight);
        panBy(camera, after.x - before.x, after.y - before.y, cssWidth, cssHeight);
        viewChanged(true);
      } else {
        held.x = event.offsetX;
        held.y = event.offsetY;
        if (camera.targetZoom > 1 && dragDistance > DRAG_THRESHOLD) {
          panBy(camera, dx, dy, cssWidth, cssHeight);
          viewChanged(true);
        }
      }
    }
    pointer.x = event.offsetX;
    pointer.y = event.offsetY;
    pointer.radius = event.pointerType === 'touch' ? HIT_RADIUS_TOUCH : HIT_RADIUS_MOUSE;
    pointer.inside = true;
    updateHover();
  }

  /**
   * The pointer left the canvas (or a touch ended). A mouse or pen clears the
   * hover; a touch keeps its last selection so a tap stays readable.
   * @param {PointerEvent} event
   */
  function onPointerLeave(event) {
    pressed.delete(event.pointerId);
    pointer.inside = false;
    frame.pointerActive = false;
    if (event.pointerType !== 'touch') setHovered(null);
  }

  /* ---------------------------------------------------------------- */
  /* Drawing                                                          */
  /* ---------------------------------------------------------------- */

  /** Radius of one mark in world units; marks grow with the square root of the zoom. */
  function markRadius() {
    return (coreRadius * 1.05) / Math.sqrt(camera.zoom);
  }

  /** The layout square as a drafting sheet: hairline grid, frame and corner marks. */
  function drawGrid() {
    const { x, y, size } = viewport;
    if (size <= 0) return;
    const hair = 1 / camera.zoom;
    ctx.strokeStyle = INK;
    ctx.lineWidth = hair;

    // The same module continues edge to edge, so the frame reads as part of a larger sheet.
    const cell = size / GRID_DIVISIONS;
    ctx.globalAlpha = 0.04;
    ctx.beginPath();
    for (let gx = x - Math.ceil(x / cell) * cell; gx <= cssWidth; gx += cell) {
      ctx.moveTo(gx, 0);
      ctx.lineTo(gx, cssHeight);
    }
    for (let gy = y - Math.ceil(y / cell) * cell; gy <= cssHeight; gy += cell) {
      ctx.moveTo(0, gy);
      ctx.lineTo(cssWidth, gy);
    }
    ctx.stroke();

    ctx.globalAlpha = 0.14;
    ctx.strokeRect(x, y, size, size);

    // Corner marks, like the registration marks on a print sheet.
    const arm = size * 0.035;
    ctx.globalAlpha = 0.55;
    ctx.beginPath();
    for (const [cx, cy, dx, dy] of [
      [x, y, 1, 1],
      [x + size, y, -1, 1],
      [x, y + size, 1, -1],
      [x + size, y + size, -1, -1],
    ]) {
      ctx.moveTo(cx + dx * arm, cy);
      ctx.lineTo(cx, cy);
      ctx.lineTo(cx, cy + dy * arm);
    }
    ctx.stroke();
  }

  /** Refreshes the drawn positions: simulation position plus a slow, tiny sway. */
  function updateDrawn() {
    const { x, y, phase, time } = particles;
    const amplitude = frame.reducedMotion ? 0 : DRIFT_AMPLITUDE / camera.zoom;
    for (let i = 0; i < count; i++) {
      drawn.x[i] = x[i] + amplitude * Math.sin(time * DRIFT_RATE + phase[i]);
      drawn.y[i] = y[i] + amplitude * Math.cos(time * DRIFT_RATE * 0.83 + phase[i] * 1.7);
    }
  }

  /** Background hairlines between near neighbours: deliberately faint, fainter still under a selection. */
  function drawEdges() {
    const { x, y } = drawn;
    const dim = 1 - 0.7 * selectMix;
    ctx.lineWidth = 1 / camera.zoom;
    let from = 0;
    for (const band of EDGE_BANDS) {
      const to = band.maxLength * viewport.size;
      ctx.globalAlpha = band.alpha * dim;
      for (let c = 0; c < clusterCount; c++) {
        const list = edges[c];
        ctx.strokeStyle = strokeStyles[c];
        ctx.beginPath();
        for (let e = 0; e < list.length; e += 2) {
          const i = list[e];
          const j = list[e + 1];
          const length = Math.hypot(x[i] - x[j], y[i] - y[j]);
          if (length < from || length >= to) continue;
          ctx.moveTo(x[i], y[i]);
          ctx.lineTo(x[j], y[j]);
        }
        ctx.stroke();
      }
      from = to;
    }
  }

  /**
   * The proposed candidate: a small tick where each mark would go, joined to
   * where the mark is now. Off-white while proposed, rust while a rejection fades.
   */
  function drawGhost() {
    if (ghostMix < 0.01) return;
    const { x, y } = drawn;
    const view = viewport;
    const tick = 1.6 / camera.zoom;
    const colour = ghostState === 'rejected' ? ACCENT : INK;
    ctx.strokeStyle = colour;
    ctx.fillStyle = colour;
    ctx.lineWidth = 1 / camera.zoom;
    ctx.globalAlpha = 0.16 * ghostMix;
    ctx.beginPath();
    for (let i = 0; i < count; i++) {
      ctx.moveTo(x[i], y[i]);
      ctx.lineTo(view.x + ghost[2 * i] * view.size, view.y + ghost[2 * i + 1] * view.size);
    }
    ctx.stroke();
    ctx.globalAlpha = 0.5 * ghostMix;
    for (let i = 0; i < count; i++) {
      ctx.fillRect(view.x + ghost[2 * i] * view.size - tick / 2, view.y + ghost[2 * i + 1] * view.size - tick / 2, tick, tick);
    }
  }

  /** Clear connections from the selected mark to each of its nearest neighbours. */
  function drawSelection() {
    if (selectMix < 0.01 || selected < 0) return;
    const { x, y } = drawn;
    ctx.globalAlpha = 0.92 * selectMix;
    ctx.strokeStyle = INK;
    ctx.lineWidth = 1.5 / camera.zoom;
    ctx.beginPath();
    for (const j of selectedNeighbours) {
      ctx.moveTo(x[selected], y[selected]);
      ctx.lineTo(x[j], y[j]);
    }
    ctx.stroke();
  }

  function drawParticles() {
    const { x, y } = drawn;
    const { depth } = particles;
    const base = markRadius() * (1 + 0.1 * pulses.accepted);
    const fade = 1 - 0.12 * pulses.rejected;

    for (let i = 0; i < count; i++) {
      const c = clusterOf[i];
      // Marks are opaque so the hairlines beneath never show through; depth varies size only.
      let alpha = fade;
      let radius = base * (0.86 + 0.28 * depth[i]);
      if (selectMix > 0) {
        if (related[i] === 2) radius *= 1 + 0.3 * selectMix;
        else if (related[i] === 0) alpha *= 1 - 0.62 * selectMix;
      }
      if (i === focus && focusMix > 0) radius *= 1 + 0.18 * focusMix;
      ctx.globalAlpha = alpha;
      ctx.beginPath();
      traceShape(ctx, shapes[c], x[i], y[i], radius);
      if (isStroked(shapes[c])) {
        ctx.strokeStyle = strokeStyles[c];
        ctx.lineWidth = radius * STROKE_RATIO;
        ctx.stroke();
      } else {
        ctx.fillStyle = strokeStyles[c];
        ctx.fill();
      }
    }
  }

  /**
   * Corner brackets around one mark.
   * @param {number} index
   * @param {number} alpha
   */
  function drawBracket(index, alpha) {
    const half = focusRadius() / camera.zoom;
    const arm = half * 0.5;
    const cx = drawn.x[index];
    const cy = drawn.y[index];
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = INK;
    ctx.lineWidth = 1.25 / camera.zoom;
    ctx.beginPath();
    for (const [dx, dy] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
      ctx.moveTo(cx + dx * half, cy + dy * (half - arm));
      ctx.lineTo(cx + dx * half, cy + dy * half);
      ctx.lineTo(cx + dx * (half - arm), cy + dy * half);
    }
    ctx.stroke();
  }

  /** Full brackets on the selected mark, lighter ones on the hovered mark. */
  function drawBrackets() {
    if (selected >= 0 && selectMix >= 0.01) drawBracket(selected, selectMix);
    if (focus >= 0 && focus !== selected && focusMix >= 0.01) drawBracket(focus, 0.6 * focusMix);
  }

  /** Half the side of the bracket around a mark, in screen pixels. */
  function focusRadius() {
    return coreRadius * 2.2 * Math.sqrt(camera.zoom) + 4;
  }

  /**
   * Text with a background-coloured outline, so it stays readable over lines and marks.
   * @param {string} text
   * @param {number} sx
   * @param {number} sy
   */
  function haloText(text, sx, sy) {
    ctx.lineJoin = 'round';
    ctx.lineWidth = 3.5;
    ctx.strokeStyle = BACKGROUND;
    ctx.strokeText(text, sx, sy);
    ctx.fillText(text, sx, sy);
  }

  /** Words beside every visible mark, once zoomed in far enough to read them. */
  function drawLabels() {
    const strength = (camera.zoom - LABEL_ZOOM_START) / (LABEL_ZOOM_FULL - LABEL_ZOOM_START);
    const { x, y } = drawn;
    const offset = coreRadius * Math.sqrt(camera.zoom) + 5;
    ctx.font = LABEL_FONT;
    ctx.letterSpacing = LABEL_TRACKING;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    ctx.fillStyle = INK;
    const tooltipOn = focusMix > 0.5;
    if (strength > 0) {
      for (let i = 0; i < count; i++) {
        if (i === selected || (i === focus && tooltipOn)) continue;
        const sx = toScreenX(camera, x[i]);
        const sy = toScreenY(camera, y[i]);
        if (sx < -80 || sx > cssWidth || sy < -10 || sy > cssHeight + 10) continue;
        const dim = related[i] === 0 ? 1 - 0.7 * selectMix : 1;
        ctx.globalAlpha = Math.min(1, strength) * 0.85 * dim;
        haloText(labels[i], sx + offset, sy + 0.5);
      }
    }
    // The selected mark keeps its word at any zoom, unless the tooltip is already showing it.
    // It sits below the mark, clear of the tooltip, which opens above.
    if (selected >= 0 && selectMix >= 0.01 && !(focus === selected && tooltipOn)) {
      ctx.globalAlpha = selectMix;
      ctx.textAlign = 'center';
      haloText(labels[selected], toScreenX(camera, x[selected]), toScreenY(camera, y[selected]) + focusRadius() + 10);
      ctx.textAlign = 'left';
    }
    ctx.letterSpacing = '0px';
  }

  /**
   * The one floating tooltip, beside the hovered mark: its shape, word and
   * topic, plus its similarity to the selected mark when it is one of that
   * mark's revealed neighbours.
   */
  function drawTooltip() {
    if (focusMix < 0.01 || focus < 0) return;
    const point = dataset.points[focus];
    const word = labels[focus];
    const note = dataset.clusterNames[point.cluster] ?? '';
    let extra = '';
    if (selected >= 0 && focus === selected) extra = 'selected';
    else if (selected >= 0 && related[focus] === 1 && similarity) {
      const value = similarity(selected, focus);
      if (Number.isFinite(value)) extra = `cos ${value.toFixed(3)}`;
    }
    const widthOf = (text) => ctx.measureText(text)?.width ?? text.length * 7;
    ctx.font = TOOLTIP_FONT;
    ctx.letterSpacing = LABEL_TRACKING;
    const wordWidth = widthOf(word);
    ctx.letterSpacing = '0px';
    ctx.font = TOOLTIP_VALUE_FONT;
    const extraWidth = extra ? widthOf(extra) + 14 : 0;
    ctx.font = TOOLTIP_NOTE_FONT;
    const noteWidth = widthOf(note);
    const padding = 11;
    const glyph = 5;
    const textLeft = padding + glyph * 2 + 9;
    const width = textLeft + wordWidth + 10 + noteWidth + extraWidth + padding;
    const height = 32;
    const sx = toScreenX(camera, drawn.x[focus]);
    const sy = toScreenY(camera, drawn.y[focus]);
    const gap = focusRadius() + 8;
    const fieldRight = viewport.fieldX + viewport.fieldW;
    let left = sx + gap;
    if (left + width > fieldRight - 8) left = sx - gap - width;
    const top = Math.round(Math.min(cssHeight - height - 8, Math.max(viewport.fieldY + 8, sy - gap - height)));
    left = Math.round(left);
    const middle = top + height / 2;

    ctx.globalAlpha = focusMix;
    ctx.fillStyle = SURFACE;
    ctx.fillRect(left, top, width, height);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.16)';
    ctx.lineWidth = 1;
    ctx.lineJoin = 'miter';
    ctx.strokeRect(left + 0.5, top + 0.5, width - 1, height - 1);
    // A short accent rule along the top edge ties the tooltip to the rest of the interface.
    ctx.fillStyle = ACCENT;
    ctx.fillRect(left, top, 18, 2);

    const shape = shapes[point.cluster];
    ctx.beginPath();
    traceShape(ctx, shape, left + padding + glyph, middle, glyph);
    if (isStroked(shape)) {
      ctx.strokeStyle = strokeStyles[point.cluster];
      ctx.lineWidth = glyph * STROKE_RATIO;
      ctx.stroke();
    } else {
      ctx.fillStyle = strokeStyles[point.cluster];
      ctx.fill();
    }

    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    ctx.font = TOOLTIP_FONT;
    ctx.letterSpacing = LABEL_TRACKING;
    ctx.fillStyle = INK;
    ctx.fillText(word, left + textLeft, middle + 1);
    ctx.letterSpacing = '0px';
    if (extra) {
      ctx.font = TOOLTIP_VALUE_FONT;
      ctx.fillStyle = ACCENT;
      ctx.fillText(extra, left + textLeft + wordWidth + 10 + noteWidth + 14, middle + 1);
    }
    ctx.font = TOOLTIP_NOTE_FONT;
    ctx.fillStyle = INK_MUTED;
    ctx.fillText(note, left + textLeft + wordWidth + 10, middle + 1);
  }

  function drawTerminatedFrame() {
    if (pulses.terminatedAge < 0) return;
    const progress = pulses.terminatedAge / TERMINATED_DURATION;
    const eased = 1 - (1 - progress) * (1 - progress) * (1 - progress);
    const grow = viewport.size * 0.06 * eased;
    ctx.globalAlpha = 0.9 * (1 - progress) * (1 - progress);
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 1.5 / camera.zoom;
    ctx.strokeRect(viewport.x - grow, viewport.y - grow, viewport.size + grow * 2, viewport.size + grow * 2);
  }

  /** Repaints the canvas from the current simulation state. */
  function draw() {
    if (cssWidth === 0 || cssHeight === 0) return;
    updateDrawn();
    ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;
    ctx.fillStyle = BACKGROUND;
    ctx.fillRect(0, 0, cssWidth, cssHeight);

    // Everything that belongs to the layout is drawn through the camera.
    const scale = pixelRatio * camera.zoom;
    ctx.setTransform(scale, 0, 0, scale, pixelRatio * camera.x, pixelRatio * camera.y);
    ctx.lineCap = 'butt';
    ctx.lineJoin = 'miter';
    drawGrid();
    drawEdges();
    drawGhost();
    drawSelection();
    drawParticles();
    drawBrackets();
    drawTerminatedFrame();

    // Text is drawn in screen space so it stays crisp and the same size at any zoom.
    ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    drawLabels();
    drawTooltip();
  }

  /* ---------------------------------------------------------------- */
  /* Loop                                                             */
  /* ---------------------------------------------------------------- */

  /** @param {number} dt */
  function update(dt) {
    const still = frame.reducedMotion;
    if (!still) stepPulses(pulses, dt);
    if (still) snapCamera(camera);
    else stepCamera(camera, dt);
    updateHover();
    stepParticles(particles, dt, frame);
    const goal = hovered === null ? 0 : 1;
    focusMix = still ? goal : focusMix + (goal - focusMix) * (1 - Math.exp(-dt * FOCUS_EASE_RATE));
    const selectGoal = selected < 0 ? 0 : 1;
    selectMix = still ? selectGoal : selectMix + (selectGoal - selectMix) * (1 - Math.exp(-dt * SELECT_EASE_RATE));
    if (ghostState === 'rejected') {
      ghostMix = still ? 0 : Math.max(0, ghostMix - dt / GHOST_FADE_TIME);
      if (ghostMix === 0) ghostState = 'none';
    }
  }

  /** @param {number} now */
  function tick(now) {
    rafId = win.requestAnimationFrame(tick);
    const dt = lastTime < 0 ? DEFAULT_DT : clampDt((now - lastTime) / 1000);
    lastTime = now;
    update(dt);
    draw();
  }

  function schedule() {
    if (rafId !== 0 || !running || doc.hidden) return;
    lastTime = -1;
    rafId = win.requestAnimationFrame(tick);
  }

  function unschedule() {
    if (rafId === 0) return;
    win.cancelAnimationFrame(rafId);
    rafId = 0;
  }

  function onVisibilityChange() {
    if (doc.hidden) unschedule();
    else schedule();
  }

  function onMotionPreferenceChange() {
    frame.reducedMotion = motionQuery.matches;
    if (frame.reducedMotion) {
      clearPulses(pulses);
      snapParticles(particles, frame);
    }
    if (rafId === 0) draw();
  }

  /* ---------------------------------------------------------------- */
  /* Public API                                                       */
  /* ---------------------------------------------------------------- */

  function resize() {
    if (destroyed) return;
    cssWidth = canvas.clientWidth;
    cssHeight = canvas.clientHeight;
    pixelRatio = Math.min(Math.max(win.devicePixelRatio || 1, 1), MAX_PIXEL_RATIO);
    computeViewport(cssWidth, cssHeight, getInsets ? getInsets() : insets, viewport);
    clampCamera(camera, cssWidth, cssHeight);
    snapCamera(camera);
    coreRadius = Math.min(Math.max(POINT_RADIUS * viewport.size * 0.55, 1.6), 5.5);

    const pixelWidth = Math.round(cssWidth * pixelRatio);
    const pixelHeight = Math.round(cssHeight * pixelRatio);
    const resized = canvas.width !== pixelWidth || canvas.height !== pixelHeight;
    if (resized) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }
    if (cssWidth === 0 || cssHeight === 0) return;

    // A resized window is followed at once: particles jump to the new fit instead of drifting to it.
    if (needsSnap || resized) {
      snapParticles(particles, frame);
      needsSnap = false;
    }
    if (rafId === 0) draw();
  }

  /**
   * @param {Float64Array|null} positions
   * @param {{ immediate?: boolean }} [layoutOptions]
   */
  function setLayout(positions, layoutOptions) {
    if (destroyed) return;
    setTargets(particles, positions);
    if ((layoutOptions && layoutOptions.immediate) || frame.reducedMotion) {
      if (!needsSnap) {
        snapParticles(particles, frame);
        if (rafId === 0) draw();
      }
    }
  }

  /**
   * Shows a proposed candidate layout over the current one, or hides it with null.
   * The array is copied. It stays until the next pulse('accepted' | 'rejected').
   * @param {Float64Array|null} positions
   */
  function showCandidate(positions) {
    if (destroyed) return;
    if (!positions || positions.length !== ghost.length) {
      ghostState = 'none';
      ghostMix = 0;
    } else {
      ghost.set(positions);
      ghostState = 'proposed';
      ghostMix = 1;
    }
    if (rafId === 0) draw();
  }

  /**
   * Resolves the candidate overlay and plays a brief accent: an accepted
   * candidate's overlay disappears as the marks move to it, a rejected one
   * turns rust and fades while the marks stay where they are.
   * @param {'accepted'|'rejected'|'terminated'} kind
   */
  function pulse(kind) {
    if (destroyed) return;
    if (kind === 'accepted' || (kind === 'rejected' && frame.reducedMotion)) {
      ghostState = 'none';
      ghostMix = 0;
    } else if (kind === 'rejected' && ghostState === 'proposed') {
      ghostState = 'rejected';
    }
    if (frame.reducedMotion) return;
    triggerPulse(pulses, kind);
  }

  function start() {
    if (destroyed || running) return;
    running = true;
    schedule();
  }

  function stop() {
    running = false;
    unschedule();
  }

  const resizeObserver = typeof win.ResizeObserver === 'function' ? new win.ResizeObserver(resize) : null;
  const previousTouchAction = canvas.style.touchAction;

  function destroy() {
    if (destroyed) return;
    stop();
    destroyed = true;
    if (resizeObserver) resizeObserver.disconnect();
    win.removeEventListener('resize', resize);
    win.removeEventListener('keydown', onKeyDown);
    doc.removeEventListener('visibilitychange', onVisibilityChange);
    if (motionQuery) motionQuery.removeEventListener('change', onMotionPreferenceChange);
    canvas.removeEventListener('pointermove', onPointerMove);
    canvas.removeEventListener('pointerdown', onPointerDown);
    canvas.removeEventListener('pointerup', onPointerUp);
    canvas.removeEventListener('wheel', onWheel);
    canvas.removeEventListener('dblclick', onDoubleClick);
    canvas.removeEventListener('pointerleave', onPointerLeave);
    canvas.removeEventListener('pointercancel', onPointerLeave);
    canvas.style.touchAction = previousTouchAction;
  }

  if (resizeObserver) resizeObserver.observe(canvas);
  win.addEventListener('resize', resize);
  win.addEventListener('keydown', onKeyDown);
  doc.addEventListener('visibilitychange', onVisibilityChange);
  if (motionQuery) motionQuery.addEventListener('change', onMotionPreferenceChange);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('wheel', onWheel, { passive: false });
  canvas.addEventListener('dblclick', onDoubleClick);
  canvas.addEventListener('pointerleave', onPointerLeave);
  canvas.addEventListener('pointercancel', onPointerLeave);
  // Dragging a finger across the canvas should stir the particles, not pan the page.
  canvas.style.touchAction = 'none';

  resize();

  return { setLayout, showCandidate, pulse, select, zoomBy, resetView, start, stop, resize, destroy };
}
