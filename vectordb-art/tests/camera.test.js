import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  MAX_ZOOM,
  MIN_ZOOM,
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
} from '../src/render/camera.js';
import { createDataset } from '../src/data/dataset.js';

const W = 800;
const H = 600;
const near = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) <= eps, `${a} vs ${b}`);

test('a new camera is the identity view', () => {
  const camera = createCamera();
  assert.equal(toScreenX(camera, 123), 123);
  assert.equal(toWorldY(camera, 45), 45);
});

test('zooming keeps the world point under the cursor fixed', () => {
  const camera = createCamera();
  zoomAt(camera, 500, 300, 2, W, H);
  snapCamera(camera);
  const wx = toWorldX(camera, 500);
  const wy = toWorldY(camera, 300);
  zoomAt(camera, 500, 300, 1.7, W, H);
  snapCamera(camera);
  near(toScreenX(camera, wx), 500);
  near(toScreenY(camera, wy), 300);
  near(camera.zoom, 3.4);
});

test('the cursor point also stays fixed while the zoom is easing', () => {
  const camera = createCamera();
  zoomAt(camera, 500, 300, 4, W, H);
  for (let i = 0; i < 5; i++) {
    assert.equal(stepCamera(camera, 1 / 60), true);
    near(toScreenX(camera, 500), 500, 1e-6);
    near(toScreenY(camera, 300), 300, 1e-6);
  }
  for (let i = 0; i < 600 && stepCamera(camera, 1 / 60); i++);
  assert.equal(camera.zoom, 4);
  assert.equal(stepCamera(camera, 1 / 60), false);
});

test('zoom is clamped to its limits', () => {
  const camera = createCamera();
  zoomAt(camera, 400, 300, 1000, W, H);
  assert.equal(camera.targetZoom, MAX_ZOOM);
  zoomAt(camera, 400, 300, 1e-6, W, H);
  assert.equal(camera.targetZoom, MIN_ZOOM);
  assert.equal(camera.targetX, 0);
  assert.equal(camera.targetY, 0);
});

test('panning cannot move the view beyond the unzoomed frame', () => {
  const camera = createCamera();
  panBy(camera, 300, -200, W, H);
  assert.deepEqual([camera.targetX, camera.targetY], [0, 0]);

  zoomAt(camera, 400, 300, 2, W, H);
  panBy(camera, 5000, 5000, W, H);
  assert.deepEqual([camera.targetX, camera.targetY], [0, 0]);
  panBy(camera, -5000, -5000, W, H);
  assert.deepEqual([camera.targetX, camera.targetY], [-W, -H]);
});

test('reset returns to the identity view', () => {
  const camera = createCamera();
  zoomAt(camera, 100, 100, 3, W, H);
  snapCamera(camera);
  resetCamera(camera);
  snapCamera(camera);
  assert.deepEqual(camera, createCamera());
});

test('world and screen conversions are inverses', () => {
  const camera = createCamera();
  zoomAt(camera, 620, 140, 2.5, W, H);
  snapCamera(camera);
  near(toWorldX(camera, toScreenX(camera, 333)), 333);
  near(toWorldY(camera, toScreenY(camera, 77)), 77);
});

test('every point carries a unique word from its own topic', () => {
  const dataset = createDataset();
  const labels = dataset.points.map((point) => point.label);
  assert.equal(new Set(labels).size, labels.length);
  for (const label of labels) assert.match(label, /^[a-z][a-z -]+$/);
  assert.deepEqual(
    createDataset().points.map((point) => point.label),
    labels,
  );
});
