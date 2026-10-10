# Vector Garden — a VectorDB art experiment

A small vector database (216 embeddings in six topics) drawn as a dark particle canvas. A seeded
**generator** proposes 2D layouts of the embeddings, a **heuristic evaluator** scores each one, and a
**loop controller** keeps only the proposals that improve on the best so far. You watch the particles
settle from a drifting cloud into a map of the collection.

Vanilla HTML, CSS and JavaScript modules. No framework, no backend, no dependencies, no build step.

## Run it

Requires Node.js 18 or newer (only for the local file server and the tests).

```sh
cd vectordb-art
npm start            # serves http://localhost:5173/
```

Then open <http://localhost:5173/>. To use another port: `node serve.mjs 8080`.

The page loads ES modules, which browsers refuse to load from `file://`, so opening `index.html`
directly does not work. Any static file server will do instead of `serve.mjs`, for example
`python -m http.server` or `npx serve`.

## Test it

```sh
npm test             # node --test, 125 tests, a couple of seconds
```

| File | Covers |
|---|---|
| `tests/controller.test.js` | Convergence, plateau, iteration limit, time limit, cancellation, rejected candidates, determinism, concurrent-loop prevention, pause/resume |
| `tests/generator.test.js` | Determinism, seeds, input immutability, bounds, semantic structure |
| `tests/discriminator.test.js` | Each criterion, validation of malformed layouts, weights |
| `tests/render.test.js` | Layout-to-screen mapping, particle easing, pointer force, hit testing, zoom and hover |
| `tests/camera.test.js` | Zoom and pan maths, word labels |
| `tests/integration.test.js` | The real modules wired together, with a fake clock |

## Using it

- **Start** begins a run with the seed in the box. **Pause** and **Resume** suspend and continue it.
  **Stop** ends it and keeps the result. **Reset** clears everything. Only the buttons that apply
  right now are shown: Start when idle, Pause and Stop while running, Resume and Stop while paused.
- The panel leads with the best score. Under it are the iteration count, how many proposals were
  kept or rejected, and, when the run ends, the reason. Below that are the score history and the
  four criterion scores; hover a criterion for its definition, or the chart for any iteration.
- The panel never scrolls. On a short or narrow window it drops the least important detail first,
  and the particle layout refits to whatever space is left.
- Move the pointer over the canvas to push particles around.
- **Hover a particle** to see a floating tooltip with its word and topic, and lines to its nearest
  neighbours in embedding space. On a touch screen, tap it.
- **Zoom in** to read the words: scroll the mouse wheel, pinch, double-click, or use the
  − / + buttons in the corner. From about 250% every particle is labelled. **Drag** to pan while
  zoomed in. Click the percentage, or double-click again, to return to the full view.
- The same seed always produces the same run. Change the seed for a different one.

## The algorithm

### Data

`src/data/dataset.js` builds the collection from a fixed seed: six topic centres in 16 dimensions,
arranged in three related pairs, with 36 unit-length vectors around each. Inside a topic the vectors
vary mostly along two directions, plus a little noise in every dimension. The dataset is deeply
frozen, so no module can change a source vector.

Each vector carries a word from its topic ("tide", "resistor", "ledger"). The words are display
labels only. The vectors are synthetic, not embeddings computed from those words, so which words
end up as neighbours inside a topic is arbitrary; only the grouping by topic is meaningful.

### Generator (`src/generator/`)

`createGenerator({ dataset, seed }).generate({ iteration, parent, feedback })` returns a candidate
layout: one `[x, y]` in the unit square per embedding.

- It works from cosine similarity between the vectors. It never reads the topic labels.
- **First iteration:** the vectors are projected onto a seeded random plane and fitted into the
  unit square. This is meaningful but crowded.
- **Later iterations:** it copies the best layout so far and applies two force-directed steps. Each
  point is pulled towards its eight nearest semantic neighbours, similar points gather, dissimilar
  points push apart, and colliding points are separated. Seeded jitter is added.
- The step size shrinks as iterations pass (half-life of 80 iterations).
- The weakest criterion in the previous evaluation gets the strongest force.
- A seeded roll picks one of three moves per iteration: `refine` (70%), `relax` (20%, more global)
  or `shake` (10%, larger and noisier). So after a rejection the same parent gets a different move.
- Each call rebuilds its random stream from `hash(seed, iteration)`. Nothing is carried between
  calls, which is what makes runs reproducible across pause, reset and restart.

### Heuristic evaluator (`src/discriminator/`)

The "discriminator" is a set of hand-written geometric measures. **It is not a trained model**, and
every result is tagged `kind: 'heuristic'`. Each criterion is a score from 0 to 1, higher is better.

| Criterion | Weight | Definition |
|---|---|---|
| Neighbourhood | 0.4 | For each point, the fraction of its 8 nearest neighbours by cosine similarity that are also among its 8 nearest neighbours on screen. Averaged over all points. |
| Separation | 0.3 | Mean silhouette `s` of the labelled topics in 2D, mapped as `clamp(0.5 + 0.5 · s / 0.6)`. A random scatter scores about 0.45; `s ≥ 0.6` scores 1. |
| Overlap | 0.2 | `1 − mean(max(0, 1 − d / 0.024))`, where `d` is each point's distance to its nearest neighbour and 0.024 is one point diameter. Exactly 1 when no two points touch. |
| Stability | 0.1 | `clamp(1 − mean displacement / 0.25)` relative to the best layout so far. 1 when nothing moved or there is no previous layout. |

The overall score is the weighted sum. A malformed layout (wrong length, non-finite or out-of-range
values) scores 0 with `valid: false` and a list of issues; the evaluator never throws on one.

### Loop (`src/loop/controller.js`)

Each iteration generates one candidate from the best layout, evaluates it, and accepts it only if
its overall score is strictly higher than the best so far. A rejected candidate never replaces the
best. After every completed iteration the stopping rules are checked, first match wins:

| Reason | Rule |
|---|---|
| **Converged** | Overall score ≥ 0.90 for three consecutive accepted iterations. Rejected iterations in between neither count nor reset the streak. |
| **Plateau** | The best score before and after the iteration differ by less than 0.005, for ten consecutive iterations. Any improvement of 0.005 or more resets the count. |
| **Iteration limit** | 100 attempted iterations. |
| **Time limit** | 10 seconds of active execution. Paused time does not count. |
| **Stopped** | Stop or Reset was pressed. Takes effect immediately. |

The iteration limit, time limit and cancellation are hard limits: they are checked again just
before each iteration, so none can start once a limit is reached. Pressing Start during a run does
nothing, so two loops never run at once. Reset invalidates the running loop with a run token, so a
stale loop can never write into the new state.

The loop waits 90 ms between iterations so each step is visible and the browser stays responsive.

### What to expect

Most seeds end on a **plateau** between 0.87 and 0.90 after 20 to 30 iterations; some **converge**
just above 0.90 (seeds 1 and 2 do). Neighbourhood preservation is the limiting criterion: flattening
16 dimensions into 2 always loses some neighbours, and the run ends honestly when the generator
stops finding improvements. Nothing in the scoring is adjusted to force convergence.

## Structure

```
index.html, styles.css     page shell and panel styling
serve.mjs                  zero-dependency local static server
src/contracts.js           shared data shapes, constants and validators (single source of truth)
src/core/                  seeded random numbers, nearest-neighbour helpers
src/data/dataset.js        the embedding collection
src/generator/             candidate layouts
src/discriminator/         heuristic scoring
src/loop/controller.js     iteration, acceptance, stopping rules, pause/resume/stop/reset
src/render/                particle canvas, zoom and pan, tooltip (knows nothing about scores or the loop)
src/ui/hud.js              control panel and score chart
src/main.js                the only file that wires the modules together
tests/                     node:test suites
```

The generator, evaluator, loop and renderer never import each other. They share only
`src/contracts.js` and `src/core/`, and `src/main.js` connects them.
