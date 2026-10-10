/**
 * Canvas renderer: one glowing particle per dataset point over a quiet dust
 * field. It owns the canvas, its requestAnimationFrame loop and pointer
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
  createDust,
  createFrame,
  createParticles,
  hitTest,
  setTargets,
  snapParticles,
  stepDust,
  stepParticles,
} from './particles.js';
import { TERMINATED_DURATION, clearPulses, createPulseState, stepPulses, triggerPulse } from './pulses.js';

const MAX_PIXEL_RATIO = 2;
const DUST_COUNT = 150;
const DUST_COLOUR = Object.freeze([150, 165, 210]);
const SPRITE_SIZE = 64;
/** The sprite's bright core ends at about this fraction of its radius. */
const SPRITE_CORE_RATIO = 0.2;
const HIT_RADIUS_MOUSE = 22;
const HIT_RADIUS_TOUCH = 32;
const TRAIL_FADE_RATE = 5;
const TRAIL_ALPHA = 0.55;
const TRAIL_MIN_SPEED = 20;
/** Seconds without any moving particle after which the trail layer is wiped clean. */
const TRAIL_SETTLE_TIME = 1.2;
const FOCUS_EASE_RATE = 12;
const REDUCED_MOTION_QUERY = '(prefers-reduced-motion: reduce)';
/** Word labels fade in between these zoom levels. */
const LABEL_ZOOM_START = 2.4;
const LABEL_ZOOM_FULL = 3.4;
const LABEL_FONT = '500 11px ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
const TOOLTIP_FONT = '600 13px ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
const TOOLTIP_NOTE_FONT = '400 12px ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
const INK = '#eef0f6';
const INK_MUTED = '#a9aebd';
/** Above this zoom the pointer reads particles instead of pushing them. */
const PUSH_MAX_ZOOM = 1.15;
const WHEEL_ZOOM_RATE = 0.0016;
const PINCH_WHEEL_ZOOM_RATE = 0.012;
const DOUBLE_CLICK_ZOOM = 3;
/** A press that moves less than this many pixels is a tap, not a drag. */
const DRAG_THRESHOLD = 4;

/**
 * Pre-renders a soft radial glow so each particle costs one drawImage.
 * @param {Document} doc
 * @param {ReadonlyArray<number>} colour [r, g, b]
 * @returns {HTMLCanvasElement}
 */
function createSprite(doc, [r, g, b]) {
  const sprite = doc.createElement('canvas');
  sprite.width = SPRITE_SIZE;
  sprite.height = SPRITE_SIZE;
  const half = SPRITE_SIZE / 2;
  const context = sprite.getContext('2d');
  const gradient = context.createRadialGradient(half, half, 0, half, half, half);
  const lift = (channel) => Math.round(channel + (255 - channel) * 0.6);
  gradient.addColorStop(0, `rgba(${lift(r)}, ${lift(g)}, ${lift(b)}, 1)`);
  gradient.addColorStop(0.1, `rgba(${r}, ${g}, ${b}, 0.95)`);
  gradient.addColorStop(0.22, `rgba(${r}, ${g}, ${b}, 0.36)`);
  gradient.addColorStop(0.5, `rgba(${r}, ${g}, ${b}, 0.08)`);
  gradient.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);
  context.fillStyle = gradient;
  context.fillRect(0, 0, SPRITE_SIZE, SPRITE_SIZE);
  return sprite;
}

/**
 * Creates the renderer for a dataset.
 *
 * Beyond the contract's RendererOptions, two optional fields are honoured:
 * `insets` (static) and `getInsets` (re-read on every resize) describe the
 * canvas edges covered by overlays, in CSS pixels; the layout is kept clear of
 * them. `getInsets` wins when both are given.
 *
 * The view can be zoomed (wheel, pinch, double-click, zoomBy) and panned (drag
 * while zoomed in). Hovering a particle shows a floating tooltip with its word;
 * zoomed in far enough, every particle is labelled.
 *
 * @param {import('../contracts.js').RendererOptions & {
 *   insets?: import('./particles.js').Insets,
 *   getInsets?: () => import('./particles.js').Insets | null | undefined,
 * }} options
 * @returns {import('../contracts.js').Renderer}
 */
export function createRenderer(options) {
  const { canvas, dataset, neighbours = null, onHover = null, onViewChange = null, insets = null, getInsets = null } = options;
  const doc = canvas.ownerDocument;
  const win = doc.defaultView;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('createRenderer: 2D canvas context is not available');

  const count = dataset.points.length;
  const particles = createParticles({ count, seed: dataset.seed });
  const dust = createDust({ count: DUST_COUNT, seed: dataset.seed });
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
  const sprites = [];
  const strokeStyles = [];
  const members = [];
  for (let c = 0; c < clusterCount; c++) {
    const [r, g, b] = clusterColour(c);
    sprites.push(createSprite(doc, [r, g, b]));
    strokeStyles.push(`rgb(${r}, ${g}, ${b})`);
    const list = [];
    for (let i = 0; i < count; i++) if (clusterOf[i] === c) list.push(i);
    members.push(Uint32Array.from(list));
  }
  const dustSprite = createSprite(doc, DUST_COLOUR);
  const rejectSprite = createSprite(doc, [214, 72, 86]);

  /* Motion trails live on their own layer so static glows never accumulate. */
  const trail = doc.createElement('canvas');
  const trailCtx = trail.getContext('2d');
  const prevX = new Float32Array(count);
  const prevY = new Float32Array(count);

  const isNeighbour = new Uint8Array(count);
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
  let ambient = null;
  let needsSnap = true;
  let running = false;
  let destroyed = false;
  let rafId = 0;
  let lastTime = -1;
  /** @type {number|null} */
  let hovered = null;
  let focus = -1;
  let focusMix = 0;

  frame.reducedMotion = motionQuery ? motionQuery.matches : false;

  /* ---------------------------------------------------------------- */
  /* Hover                                                            */
  /* ---------------------------------------------------------------- */

  /** @param {number|null} index */
  function setHovered(index) {
    if (index === hovered) return;
    hovered = index;
    frame.exempt = index ?? -1;
    if (index !== null) {
      focus = index;
      isNeighbour.fill(0);
      const list = neighbours ? neighbours[index] : null;
      if (list) {
        for (let k = 0; k < list.length; k++) {
          const j = list[k];
          if (j >= 0 && j < count) isNeighbour[j] = 1;
        }
      }
    }
    if (onHover) onHover(index);
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
    if (rafId === 0) draw(0);
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

  /** @param {PointerEvent} event */
  function onPointerUp(event) {
    pressed.delete(event.pointerId);
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

  let trailQuietTime = 0;
  let trailDirty = false;

  function resetTrails() {
    trailDirty = false;
    prevX.set(particles.x);
    prevY.set(particles.y);
    trailCtx.setTransform(1, 0, 0, 1, 0, 0);
    trailCtx.clearRect(0, 0, trail.width, trail.height);
    trailCtx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  }

  /** @param {number} dt */
  function updateTrails(dt) {
    trailCtx.globalCompositeOperation = 'destination-out';
    trailCtx.globalAlpha = 1 - Math.exp(-dt * TRAIL_FADE_RATE);
    trailCtx.fillStyle = '#000';
    trailCtx.fillRect(0, 0, cssWidth, cssHeight);

    trailCtx.globalCompositeOperation = 'lighter';
    trailCtx.globalAlpha = 0.5;
    trailCtx.lineCap = 'round';
    trailCtx.lineWidth = coreRadius * 0.9;
    const { x, y, vx, vy } = particles;
    const minSpeed2 = TRAIL_MIN_SPEED * TRAIL_MIN_SPEED;
    let moving = false;
    for (let c = 0; c < clusterCount; c++) {
      const list = members[c];
      trailCtx.strokeStyle = strokeStyles[c];
      trailCtx.beginPath();
      for (let k = 0; k < list.length; k++) {
        const i = list[k];
        if (vx[i] * vx[i] + vy[i] * vy[i] < minSpeed2) continue;
        moving = true;
        trailCtx.moveTo(prevX[i], prevY[i]);
        trailCtx.lineTo(x[i], y[i]);
      }
      trailCtx.stroke();
    }
    prevX.set(x);
    prevY.set(y);

    // The alpha fade above cannot take 8-bit pixels all the way to zero, so faint streaks would
    // linger forever. Once nothing has moved for a while, clear the layer outright.
    if (moving) {
      trailQuietTime = 0;
      trailDirty = true;
    } else if (trailDirty && (trailQuietTime += dt) >= TRAIL_SETTLE_TIME) {
      resetTrails();
    }
  }

  function drawDust() {
    const still = frame.reducedMotion;
    const time = particles.time;
    for (let i = 0; i < dust.count; i++) {
      const depth = dust.depth[i];
      const twinkle = still ? 1 : 0.7 + 0.3 * Math.sin(time * 0.6 + dust.phase[i]);
      const radius = 2.5 + 6 * depth;
      ctx.globalAlpha = (0.05 + 0.16 * depth) * twinkle;
      ctx.drawImage(dustSprite, dust.x[i] * cssWidth - radius, dust.y[i] * cssHeight - radius, radius * 2, radius * 2);
    }
  }

  function drawLinks() {
    if (focusMix < 0.01 || focus < 0 || !neighbours) return;
    const list = neighbours[focus];
    if (!list) return;
    const { x, y } = particles;
    ctx.globalAlpha = 0.42 * focusMix;
    ctx.strokeStyle = strokeStyles[clusterOf[focus]];
    ctx.lineWidth = 1 / camera.zoom;
    ctx.beginPath();
    for (let k = 0; k < list.length; k++) {
      const j = list[k];
      if (j < 0 || j >= count) continue;
      ctx.moveTo(x[focus], y[focus]);
      ctx.lineTo(x[j], y[j]);
    }
    ctx.stroke();
  }

  function drawParticles() {
    const { x, y, depth, phase, time } = particles;
    const still = frame.reducedMotion;
    const accepted = pulses.accepted;
    const rejected = pulses.rejected;
    const bloom = pulses.terminatedAge < 0 ? 0 : 1 - pulses.terminatedAge / TERMINATED_DURATION;
    const brightness = (1 + 0.3 * accepted + 0.3 * bloom * bloom) * (1 - 0.18 * rejected);
    const swell = 1 + 0.14 * accepted;
    // Glows grow with the square root of the zoom, so zooming in opens up space between them.
    const spriteRadius = coreRadius / SPRITE_CORE_RATIO / Math.sqrt(camera.zoom);

    for (let i = 0; i < count; i++) {
      let alpha = (0.6 + 0.4 * depth[i]) * brightness;
      let radius = spriteRadius * (0.72 + 0.56 * depth[i]) * swell;
      if (!still) alpha *= 0.88 + 0.12 * Math.sin(time * 1.3 + phase[i]);
      if (focusMix > 0) {
        if (i === focus) {
          alpha *= 1 + 0.6 * focusMix;
          radius *= 1 + 0.35 * focusMix;
        } else if (isNeighbour[i]) alpha *= 1 + 0.25 * focusMix;
        else alpha *= 1 - 0.5 * focusMix;
      }
      let px = x[i];
      let py = y[i];
      if (rejected > 0) {
        px += Math.sin(i * 12.9898 + time * 83) * 1.5 * rejected;
        py += Math.cos(i * 78.233 + time * 71) * 1.5 * rejected;
      }
      ctx.globalAlpha = alpha > 1 ? 1 : alpha;
      ctx.drawImage(sprites[clusterOf[i]], px - radius, py - radius, radius * 2, radius * 2);
      if (rejected > 0.02) {
        ctx.globalAlpha = 0.22 * rejected;
        ctx.drawImage(rejectSprite, px - radius, py - radius, radius * 2, radius * 2);
      }
    }
  }

  function drawFocusRing() {
    if (focusMix < 0.01 || focus < 0) return;
    ctx.globalAlpha = 0.75 * focusMix;
    ctx.strokeStyle = '#eef0ff';
    ctx.lineWidth = 1.25 / camera.zoom;
    ctx.beginPath();
    ctx.arc(particles.x[focus], particles.y[focus], focusRadius() / camera.zoom, 0, Math.PI * 2);
    ctx.stroke();
  }

  /** Radius of the ring around the focused particle, in screen pixels. */
  function focusRadius() {
    return coreRadius * 2.2 * Math.sqrt(camera.zoom) + 3;
  }

  /** Words beside every visible particle, once zoomed in far enough to read them. */
  function drawLabels() {
    const strength = (camera.zoom - LABEL_ZOOM_START) / (LABEL_ZOOM_FULL - LABEL_ZOOM_START);
    if (strength <= 0) return;
    const { x, y } = particles;
    const offset = coreRadius * Math.sqrt(camera.zoom) + 5;
    ctx.font = LABEL_FONT;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    ctx.fillStyle = INK;
    for (let i = 0; i < count; i++) {
      const sx = toScreenX(camera, x[i]);
      const sy = toScreenY(camera, y[i]);
      if (sx < -80 || sx > cssWidth || sy < -10 || sy > cssHeight + 10) continue;
      if (i === focus && focusMix > 0.5) continue;
      const dim = focusMix > 0 && !isNeighbour[i] ? 1 - 0.65 * focusMix : 1;
      ctx.globalAlpha = Math.min(1, strength) * 0.82 * dim;
      ctx.fillText(dataset.points[i].label, sx + offset, sy);
    }
  }

  /** Floating tooltip beside the hovered particle: its word and topic. */
  function drawTooltip() {
    if (focusMix < 0.01 || focus < 0) return;
    const point = dataset.points[focus];
    const note = dataset.clusterNames[point.cluster] ?? '';
    const widthOf = (text) => ctx.measureText(text)?.width ?? text.length * 7;
    ctx.font = TOOLTIP_FONT;
    const wordWidth = widthOf(point.label);
    ctx.font = TOOLTIP_NOTE_FONT;
    const noteWidth = widthOf(note);
    const padding = 10;
    const swatch = 8;
    const textLeft = padding + swatch + 7;
    const width = textLeft + wordWidth + 8 + noteWidth + padding;
    const height = 30;
    const sx = toScreenX(camera, particles.x[focus]);
    const sy = toScreenY(camera, particles.y[focus]);
    const gap = focusRadius() + 8;
    let left = sx + gap;
    if (left + width > cssWidth - 8) left = sx - gap - width;
    const top = Math.min(cssHeight - height - 8, Math.max(8, sy - gap - height));
    const middle = top + height / 2;

    ctx.globalAlpha = focusMix;
    ctx.fillStyle = 'rgba(21, 24, 35, 0.94)';
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    if (typeof ctx.roundRect === 'function') ctx.roundRect(left, top, width, height, 8);
    else ctx.rect(left, top, width, height);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = strokeStyles[point.cluster];
    ctx.beginPath();
    ctx.arc(left + padding + swatch / 2, middle, swatch / 2, 0, Math.PI * 2);
    ctx.fill();

    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    ctx.font = TOOLTIP_FONT;
    ctx.fillStyle = INK;
    ctx.fillText(point.label, left + textLeft, middle + 0.5);
    ctx.font = TOOLTIP_NOTE_FONT;
    ctx.fillStyle = INK_MUTED;
    ctx.fillText(note, left + textLeft + wordWidth + 8, middle + 0.5);
  }

  function drawTerminatedRing() {
    if (pulses.terminatedAge < 0) return;
    const progress = pulses.terminatedAge / TERMINATED_DURATION;
    const eased = 1 - (1 - progress) * (1 - progress) * (1 - progress);
    const fade = (1 - progress) * (1 - progress);
    const cx = viewport.x + viewport.size / 2;
    const cy = viewport.y + viewport.size / 2;
    const radius = viewport.size * (0.08 + 0.72 * eased);
    ctx.strokeStyle = '#dcd8ff';
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.globalAlpha = 0.06 * fade;
    ctx.lineWidth = 14;
    ctx.stroke();
    ctx.globalAlpha = 0.4 * fade;
    ctx.lineWidth = 1.25 / camera.zoom;
    ctx.stroke();
  }

  /** @param {number} dt Seconds since the previous frame; 0 repaints without advancing trails. */
  function draw(dt) {
    if (cssWidth === 0 || cssHeight === 0) return;
    ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;
    ctx.fillStyle = BACKGROUND;
    ctx.fillRect(0, 0, cssWidth, cssHeight);
    if (ambient) {
      ctx.fillStyle = ambient;
      ctx.fillRect(0, 0, cssWidth, cssHeight);
    }

    ctx.globalCompositeOperation = 'lighter';
    drawDust();
    // Everything that belongs to the layout is drawn through the camera.
    const scale = pixelRatio * camera.zoom;
    ctx.setTransform(scale, 0, 0, scale, pixelRatio * camera.x, pixelRatio * camera.y);
    if (!frame.reducedMotion) {
      if (dt > 0) updateTrails(dt);
      ctx.globalAlpha = TRAIL_ALPHA;
      ctx.drawImage(trail, 0, 0, cssWidth, cssHeight);
    }
    drawLinks();
    drawParticles();
    drawFocusRing();
    drawTerminatedRing();

    // Text is drawn in screen space so it stays crisp and the same size at any zoom.
    ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    ctx.globalCompositeOperation = 'source-over';
    drawLabels();
    drawTooltip();
  }

  /* ---------------------------------------------------------------- */
  /* Loop                                                             */
  /* ---------------------------------------------------------------- */

  /** @param {number} dt */
  function update(dt) {
    const still = frame.reducedMotion;
    if (!still) {
      stepPulses(pulses, dt);
      stepDust(dust, dt, particles.time);
    }
    if (still) snapCamera(camera);
    else stepCamera(camera, dt);
    updateHover();
    stepParticles(particles, dt, frame);
    const goal = hovered === null ? 0 : 1;
    focusMix = still ? goal : focusMix + (goal - focusMix) * (1 - Math.exp(-dt * FOCUS_EASE_RATE));
  }

  /** @param {number} now */
  function tick(now) {
    rafId = win.requestAnimationFrame(tick);
    const dt = lastTime < 0 ? DEFAULT_DT : clampDt((now - lastTime) / 1000);
    lastTime = now;
    update(dt);
    draw(dt);
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
    resetTrails();
    if (rafId === 0) draw(0);
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
      trail.width = pixelWidth;
      trail.height = pixelHeight;
    }
    if (cssWidth === 0 || cssHeight === 0) return;

    const cx = viewport.x + viewport.size / 2;
    const cy = viewport.y + viewport.size / 2;
    ambient = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(cssWidth, cssHeight) * 0.7);
    ambient.addColorStop(0, 'rgba(46, 52, 104, 0.2)');
    ambient.addColorStop(0.5, 'rgba(24, 28, 60, 0.09)');
    ambient.addColorStop(1, 'rgba(7, 8, 13, 0)');

    // A resized window is followed at once: particles jump to the new fit instead of drifting to it.
    if (needsSnap || resized) {
      snapParticles(particles, frame);
      needsSnap = false;
    }
    if (resized) resetTrails();
    if (rafId === 0) draw(0);
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
        resetTrails();
        if (rafId === 0) draw(0);
      }
    }
  }

  /** @param {'accepted'|'rejected'|'terminated'} kind */
  function pulse(kind) {
    if (destroyed || frame.reducedMotion) return;
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

  return { setLayout, pulse, zoomBy, resetView, start, stop, resize, destroy };
}
