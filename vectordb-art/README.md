# Vector Garden — a VectorDB art experiment

A small vector database (216 embeddings in six topics) drawn as geometric marks on a dark drafting
sheet: one shape and colour per topic, with hairlines joining each vector to its nearest neighbours. A seeded
**generator** proposes 2D layouts of the embeddings, a **heuristic evaluator** scores each one, and a
**loop controller** keeps only the proposals that improve on the best so far. You watch the marks
settle from a tangle of crossing lines into tight meshes, one per topic.

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
npm test             # node --test, 136 tests, a couple of seconds
```

| File | Covers |
|---|---|
| `tests/controller.test.js` | Convergence, plateau, iteration limit, time limit, cancellation, rejected candidates, determinism, concurrent-loop prevention, pause/resume |
| `tests/generator.test.js` | Determinism, seeds, input immutability, bounds, semantic structure |
| `tests/discriminator.test.js` | Each criterion, validation of malformed layouts, weights |
| `tests/render.test.js` | Layout-to-screen mapping, easing, pointer force, hit testing, zoom, hover, selection, candidate overlay |
| `tests/camera.test.js` | Zoom and pan maths, word labels |
| `tests/integration.test.js` | The real modules wired together with a fake clock; neighbours against brute-force cosine; weights and thresholds unchanged |

## Using it

The screen has three parts.

- **Top bar.** A link back to the EmpoweredAI home page, the wordmark, the dataset (216 vectors, 16 dimensions, six topics, synthetic), the
  current status, the seed, and the Inspector toggle.
- **Canvas.** The artwork fills everything below the bar. The layout sits in a framed square on a
  grid that runs edge to edge.
- **Controls.** A readout rail on the left and an inspector on the right. On a phone the rail becomes
  a sheet along the bottom and the inspector stacks above it.

### Running the loop

- **Start** begins a run with the seed in the top bar. **Pause** and **Resume** suspend and continue
  it. **Stop** ends it and keeps the result. **Reset** clears everything. Only the buttons that apply
  right now are shown: Start when idle, Pause and Stop while running, Resume and Stop while paused.
- The rail shows the best score, the iteration, and the score history. While running, one line says
  what the loop is doing. When the run ends, a block names the reason, each with its own glyph and
  rule: converged (the only one in the accent colour), plateau, iteration limit, time limit, stopped.
- The same seed always produces the same run. Change the seed for a different one.

### The inspector

Detail is kept out of the way until it is useful. The inspector holds the score breakdown (bar: best
layout, tick: latest candidate; hover a criterion for its definition), the kept and rejected counts
and active time, the topic key, and the current selection. It opens when a run ends or a mark is
selected, and from the Inspector button at any time. Once you close it by hand it stays closed until
you open it again. On a phone it only opens from the button.

### Exploring the canvas

- **Hover a mark** to bracket it and see one tooltip with its word and topic.
- **Click a mark** (or tap it) to select it. Its five nearest neighbours are joined to it with clear
  lines, everything else recedes, and the inspector lists the neighbours with their cosine
  similarity. Hover one of those neighbours to see its similarity in the tooltip. Click another mark
  to move the selection; click the same mark, empty space, or press Esc to clear it.
- **Zoom in** to read the words: scroll the mouse wheel, pinch, double-click, or use the − / +
  controls in the corner. From about 250% every mark is labelled. **Drag** to pan while zoomed in.
  Click the percentage, or double-click again, to return to the full view.
- Every control works from the keyboard: Tab to reach it, Enter or Space to press it. Selecting a
  mark needs a pointer.

The interface never scrolls. On short or narrow windows it drops the least important detail first,
and the artwork refits to the space the controls leave.

### Visual system

All tokens are CSS custom properties at the top of `styles.css`.

- **Colour:** the EMPOWEREDAI site's warm charcoal, off-white and single rust accent. The accent is
  reserved for the primary action, the running indicator, the best-score line and a convergence.
- **Type:** Barlow Condensed for the wordmark and the words on the canvas; Barlow for controls and
  prose; IBM Plex Mono for every number. Small uppercase labels are used for secondary metadata only.
- **Shape:** hairlines, 2px corners, a 4px spacing grid, no shadows, blurs or gradients.

## The algorithm

### Data

`src/data/dataset.js` builds the collection from a fixed seed: six topic centres in 16 dimensions,
arranged in three related pairs, with 36 unit-length vectors around each. Inside a topic the vectors
vary mostly along two directions, plus a little noise in every dimension. The dataset is deeply
frozen, so no module can change a source vector.

Neighbours and similarity scores are real: they are the cosine similarity of the stored vectors,
computed in `src/core/knn.js` and never by the drawing code. What is synthetic is the data itself.

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

### What you see during a run

The loop waits 200 ms between iterations. On screen each iteration has two beats:

1. **Proposed.** The candidate appears as small ticks joined by hairlines to where the marks are now,
   and the panel reads "Candidate 12 · evaluating".
2. **Verdict**, 90 ms later. If the candidate was kept, the marks move to it and the panel reads
   "· kept". If it was rejected, the overlay turns rust and fades, the marks stay on the best layout,
   and the panel reads "· rejected, best kept". Ticks on the score breakdown show where the candidate
   scored against the bars for the best layout.

Generating and scoring a candidate takes under 3 ms, so the display is paced for the viewer: the
verdict already exists when the proposal is drawn. The pacing does not change results. Sixty seeds
were checked at 200 ms and each ends for the same reason, with the same history, as with no pause.

When a run ends the panel names the actual reason. A plateau reads "Plateau, not converged" and
states that the 0.90 target was not reached; only a real convergence is called one.

The marks also sway by about a pixel, slowly. That is added when drawing only: it never touches the
layout coordinates, the vectors, or anything the scores or neighbour rankings are computed from, and
it is switched off under the reduced-motion preference.

### What to expect

Most seeds end on a **plateau** between 0.87 and 0.90 after 20 to 30 iterations; some **converge**
just above 0.90 (seeds 1 and 2 do). Neighbourhood preservation is the limiting criterion: flattening
16 dimensions into 2 always loses some neighbours, and the run ends honestly when the generator
stops finding improvements. Nothing in the scoring is adjusted to force convergence.

## Structure

```
index.html, styles.css     page shell and the visual system (tokens at the top of styles.css)
serve.mjs                  zero-dependency local static server
src/contracts.js           shared data shapes, constants and validators (single source of truth)
src/core/                  seeded random numbers, nearest-neighbour helpers
src/data/dataset.js        the embedding collection
src/generator/             candidate layouts
src/discriminator/         heuristic scoring
src/loop/controller.js     iteration, acceptance, stopping rules, pause/resume/stop/reset
src/render/                canvas: marks, neighbour lines, zoom and pan, tooltip (knows nothing about scores or the loop)
src/ui/hud.js              top bar, readout rail, inspector and score chart
src/main.js                the only file that wires the modules together
tests/                     node:test suites
```

The generator, evaluator, loop and renderer never import each other. They share only
`src/contracts.js` and `src/core/`, and `src/main.js` connects them.
