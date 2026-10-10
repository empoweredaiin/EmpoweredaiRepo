/**
 * One geometric mark per topic. Shape carries identity alongside colour, so a
 * topic can be told apart without relying on hue.
 */

export const SHAPES = Object.freeze(['circle', 'square', 'triangle', 'diamond', 'cross', 'ring']);

/** Shapes drawn as an outline instead of a filled area. */
const STROKED = new Set(['cross', 'ring']);

/** Outline thickness of a stroked shape, as a fraction of its radius. */
export const STROKE_RATIO = 0.5;

/**
 * @param {number} cluster
 * @returns {string} The shape name for a cluster index; wraps past the last shape.
 */
export function shapeOf(cluster) {
  const n = SHAPES.length;
  return SHAPES[(((cluster | 0) % n) + n) % n];
}

/** @param {string} shape */
export function isStroked(shape) {
  return STROKED.has(shape);
}

/**
 * Adds one shape to the current canvas path. The caller begins the path and
 * fills it, or strokes it when isStroked(shape). Sizes are balanced so every
 * shape reads as about the same visual weight at radius `r`.
 * @param {CanvasRenderingContext2D} ctx
 * @param {string} shape
 * @param {number} x
 * @param {number} y
 * @param {number} r
 */
export function traceShape(ctx, shape, x, y, r) {
  switch (shape) {
    case 'square': {
      const h = r * 0.88;
      ctx.rect(x - h, y - h, h * 2, h * 2);
      break;
    }
    case 'triangle': {
      const h = r * 1.25;
      ctx.moveTo(x, y - h);
      ctx.lineTo(x + h * 0.9, y + h * 0.62);
      ctx.lineTo(x - h * 0.9, y + h * 0.62);
      ctx.closePath();
      break;
    }
    case 'diamond': {
      const h = r * 1.3;
      ctx.moveTo(x, y - h);
      ctx.lineTo(x + h, y);
      ctx.lineTo(x, y + h);
      ctx.lineTo(x - h, y);
      ctx.closePath();
      break;
    }
    case 'cross': {
      const h = r * 1.1;
      ctx.moveTo(x - h, y);
      ctx.lineTo(x + h, y);
      ctx.moveTo(x, y - h);
      ctx.lineTo(x, y + h);
      break;
    }
    case 'ring':
      ctx.moveTo(x + r * 0.9, y);
      ctx.arc(x, y, r * 0.9, 0, Math.PI * 2);
      break;
    default:
      ctx.moveTo(x + r, y);
      ctx.arc(x, y, r, 0, Math.PI * 2);
  }
}

/**
 * The same shape as SVG markup, for legends.
 * @param {string} shape
 * @param {string} colour CSS colour.
 * @param {number} [size] Width and height in CSS pixels.
 * @returns {string}
 */
export function shapeSvg(shape, colour, size = 10) {
  const paint = isStroked(shape)
    ? `fill="none" stroke="${colour}" stroke-width="2" stroke-linecap="butt"`
    : `fill="${colour}"`;
  const body = {
    circle: `<circle cx="6" cy="6" r="4.2" ${paint}/>`,
    square: `<rect x="2.2" y="2.2" width="7.6" height="7.6" ${paint}/>`,
    triangle: `<path d="M6 1.2 L10.8 9.6 L1.2 9.6 Z" ${paint}/>`,
    diamond: `<path d="M6 0.8 L11.2 6 L6 11.2 L0.8 6 Z" ${paint}/>`,
    cross: `<path d="M1.4 6 H10.6 M6 1.4 V10.6" ${paint}/>`,
    ring: `<circle cx="6" cy="6" r="3.8" ${paint}/>`,
  }[shape];
  return `<svg class="mark" viewBox="0 0 12 12" width="${size}" height="${size}" aria-hidden="true">${body ?? ''}</svg>`;
}
