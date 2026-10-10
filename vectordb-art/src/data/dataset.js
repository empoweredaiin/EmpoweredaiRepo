/**
 * The "vector database": a small synthetic collection of unit-length embeddings
 * grouped into named topics. Built deterministically from a seed and deeply
 * frozen, so no module can alter the source vectors.
 */
import { createRng, gaussian, hashSeed } from '../core/rng.js';

export const DATASET_DEFAULTS = Object.freeze({
  seed: 7,
  dim: 16,
  pointsPerCluster: 36,
  spread: 0.45,
  noise: 0.02,
});

const CLUSTER_NAMES = Object.freeze(['tides', 'circuits', 'orchards', 'archives', 'weather', 'cities']);

/**
 * A word for every point, by topic. The words are display labels attached to synthetic vectors:
 * the vectors are generated below, they are not embeddings computed from these words.
 */
const WORDS = Object.freeze({
  tides:
    'tide wave reef kelp harbour estuary current swell lagoon shoal coral driftwood anchor buoy gull brine surf undertow dune pier trawler lighthouse sandbar pearl oyster plankton whale tidepool spray foam shore inlet delta mooring jetty seabed'.split(' '),
  circuits:
    'resistor capacitor diode transistor inductor relay solder breadboard oscillator amplifier voltage ampere ohm ground fuse switch sensor antenna microchip logic latch register clock signal waveform firmware pinout trace via rectifier transformer battery electrode filament conductor insulator'.split(' '),
  orchards:
    'apple pear plum cherry quince apricot peach fig blossom graft rootstock pruning harvest cider windfall orchard trellis pollen bee hive mulch sapling bough ladder basket ripening russet pippin medlar mulberry walnut hazel damson greengage canopy compost'.split(' '),
  archives:
    'ledger manuscript folio catalogue index scroll parchment vellum binding archive record roll deed charter diary letter microfilm shelf vault registry annals chronicle codex footnote citation memoir almanac atlas dossier transcript facsimile provenance inventory minutes gazette pamphlet'.split(' '),
  weather:
    'rain drizzle sleet hail snow frost fog mist gale breeze squall thunder lightning monsoon cyclone drought humidity pressure isobar front cloud cumulus cirrus stratus dew thaw blizzard heatwave rainbow barometer forecast downpour overcast zephyr whirlwind sunshine'.split(' '),
  cities:
    'avenue alley boulevard plaza tram subway skyline tower bridge market kiosk crosswalk suburb district terminal platform flyover pavement lamppost billboard cafe arcade tenement courtyard fountain square viaduct overpass tunnel bazaar rooftop storefront sidewalk junction metro rickshaw'.split(' '),
});

function normalise(vector) {
  let norm = 0;
  for (const v of vector) norm += v * v;
  norm = Math.sqrt(norm) || 1;
  return vector.map((v) => v / norm);
}

/**
 * Each topic varies mostly along two directions of its own (`spread`), plus a little noise in
 * every dimension (`noise`). Like real embeddings, the collection has far fewer degrees of
 * freedom than dimensions, which is what makes a faithful 2D layout possible at all.
 *
 * @param {{ seed?: number, dim?: number, pointsPerCluster?: number, spread?: number, noise?: number }} [options]
 * @returns {import('../contracts.js').Dataset}
 */
export function createDataset(options = {}) {
  const { seed, dim, pointsPerCluster, spread, noise } = { ...DATASET_DEFAULTS, ...options };
  const rng = createRng(hashSeed(seed, 0x5eed));
  const randomUnit = () => normalise(Array.from({ length: dim }, () => gaussian(rng)));

  // Topics come in related pairs: each odd topic is a blend of the previous one
  // and a fresh direction, so some clusters are semantically closer than others.
  const centres = [];
  for (let c = 0; c < CLUSTER_NAMES.length; c++) {
    const fresh = randomUnit();
    centres.push(c % 2 === 1 ? normalise(fresh.map((v, i) => v + 0.7 * centres[c - 1][i])) : fresh);
  }

  const points = [];
  for (let c = 0; c < centres.length; c++) {
    const axisU = randomUnit();
    const axisV = randomUnit();
    for (let p = 0; p < pointsPerCluster; p++) {
      // Uniform sample of the unit disc in the topic's own plane.
      const radius = Math.sqrt(rng());
      const angle = 2 * Math.PI * rng();
      const u = radius * Math.cos(angle);
      const v = radius * Math.sin(angle);
      const vector = normalise(
        centres[c].map((centre, i) => centre + spread * (u * axisU[i] + v * axisV[i]) + noise * gaussian(rng)),
      );
      points.push(
        Object.freeze({
          id: `${CLUSTER_NAMES[c]}-${String(p + 1).padStart(2, '0')}`,
          vector: Object.freeze(vector),
          cluster: c,
          label: WORDS[CLUSTER_NAMES[c]][p] ?? `${CLUSTER_NAMES[c]} ${p + 1}`,
        }),
      );
    }
  }

  return Object.freeze({ seed, dim, points: Object.freeze(points), clusterNames: CLUSTER_NAMES });
}
