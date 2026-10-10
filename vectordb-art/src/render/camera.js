/**
 * Pure, DOM-free zoom and pan state for the canvas.
 *
 * The simulation always runs in "world" coordinates (CSS pixels of the unzoomed
 * canvas). The camera maps them to the screen: screen = world * zoom + offset.
 * Interactions move the target; stepCamera eases the current view towards it.
 */

export const MIN_ZOOM = 1;
export const MAX_ZOOM = 8;
/** Exponential ease rate in 1/s; about 0.25 s to settle. */
export const CAMERA_EASE_RATE = 14;

/**
 * @typedef {Object} Camera
 * @property {number} zoom
 * @property {number} x        Screen offset of the world origin, CSS pixels.
 * @property {number} y
 * @property {number} targetZoom
 * @property {number} targetX
 * @property {number} targetY
 */

/** @returns {Camera} */
export function createCamera() {
  return { zoom: 1, x: 0, y: 0, targetZoom: 1, targetX: 0, targetY: 0 };
}

/**
 * Keeps the target view inside the canvas, so zooming or panning never
 * reveals anything beyond the unzoomed frame.
 * @param {Camera} camera
 * @param {number} width   Canvas width, CSS pixels.
 * @param {number} height
 */
export function clampCamera(camera, width, height) {
  camera.targetZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, camera.targetZoom));
  camera.targetX = Math.min(0, Math.max(width * (1 - camera.targetZoom), camera.targetX));
  camera.targetY = Math.min(0, Math.max(height * (1 - camera.targetZoom), camera.targetY));
}

/**
 * Multiplies the target zoom by `factor`, keeping the world point under the
 * screen position (sx, sy) where it is.
 * @param {Camera} camera
 * @param {number} sx
 * @param {number} sy
 * @param {number} factor
 * @param {number} width
 * @param {number} height
 */
export function zoomAt(camera, sx, sy, factor, width, height) {
  const worldX = (sx - camera.targetX) / camera.targetZoom;
  const worldY = (sy - camera.targetY) / camera.targetZoom;
  camera.targetZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, camera.targetZoom * factor));
  camera.targetX = sx - worldX * camera.targetZoom;
  camera.targetY = sy - worldY * camera.targetZoom;
  clampCamera(camera, width, height);
}

/**
 * Moves the target view by a screen-space delta.
 * @param {Camera} camera
 * @param {number} dx
 * @param {number} dy
 * @param {number} width
 * @param {number} height
 */
export function panBy(camera, dx, dy, width, height) {
  camera.targetX += dx;
  camera.targetY += dy;
  clampCamera(camera, width, height);
}

/** Targets the unzoomed view. @param {Camera} camera */
export function resetCamera(camera) {
  camera.targetZoom = 1;
  camera.targetX = 0;
  camera.targetY = 0;
}

/** Jumps to the target without easing. @param {Camera} camera */
export function snapCamera(camera) {
  camera.zoom = camera.targetZoom;
  camera.x = camera.targetX;
  camera.y = camera.targetY;
}

/**
 * Eases the view towards its target. Zoom and offset share one rate, so the
 * point being zoomed about stays fixed on screen throughout.
 * @param {Camera} camera
 * @param {number} dt Seconds.
 * @returns {boolean} True while the view is still moving.
 */
export function stepCamera(camera, dt) {
  const dz = camera.targetZoom - camera.zoom;
  const dx = camera.targetX - camera.x;
  const dy = camera.targetY - camera.y;
  if (Math.abs(dz) < 1e-4 && Math.abs(dx) < 0.05 && Math.abs(dy) < 0.05) {
    snapCamera(camera);
    return false;
  }
  const t = 1 - Math.exp(-dt * CAMERA_EASE_RATE);
  camera.zoom += dz * t;
  camera.x += dx * t;
  camera.y += dy * t;
  return true;
}

/** @param {Camera} camera @param {number} sx */
export function toWorldX(camera, sx) {
  return (sx - camera.x) / camera.zoom;
}

/** @param {Camera} camera @param {number} sy */
export function toWorldY(camera, sy) {
  return (sy - camera.y) / camera.zoom;
}

/** @param {Camera} camera @param {number} wx */
export function toScreenX(camera, wx) {
  return wx * camera.zoom + camera.x;
}

/** @param {Camera} camera @param {number} wy */
export function toScreenY(camera, wy) {
  return wy * camera.zoom + camera.y;
}
