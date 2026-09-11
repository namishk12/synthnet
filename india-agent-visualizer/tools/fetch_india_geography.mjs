import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const outputDirectory = resolve(root, "public", "data");

const countrySource =
  "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_0_countries.geojson";
const statesSource =
  "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_1_states_provinces.geojson";

async function fetchGeoJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Unable to fetch ${url}: ${response.status}`);
  }
  return response.json();
}

function belongsToIndia(feature) {
  const properties = feature?.properties ?? {};
  return (
    properties.ADM0_A3 === "IND" ||
    properties.adm0_a3 === "IND" ||
    properties.sr_adm0_a3 === "IND" ||
    properties.ISO_A2 === "IN" ||
    properties.iso_a2 === "IN" ||
    properties.ADMIN === "India" ||
    properties.admin === "India" ||
    properties.geonunit === "India"
  );
}

function compactFeature(feature, properties) {
  return {
    type: "Feature",
    properties,
    geometry: feature.geometry,
  };
}

const countries = await fetchGeoJson(countrySource);
const country = countries.features.find(belongsToIndia);
if (!country) {
  throw new Error("India was not found in the Natural Earth country dataset.");
}

const states = await fetchGeoJson(statesSource);
const indiaStates = states.features.filter(belongsToIndia);
if (indiaStates.length === 0) {
  throw new Error("Indian states were not found in the Natural Earth dataset.");
}

await mkdir(outputDirectory, { recursive: true });
await writeFile(
  resolve(outputDirectory, "india-outline.geojson"),
  JSON.stringify(
    compactFeature(country, {
      name: "India",
      source: "Natural Earth",
    }),
  ),
);
await writeFile(
  resolve(outputDirectory, "india-states.geojson"),
  JSON.stringify({
    type: "FeatureCollection",
    features: indiaStates.map((feature) =>
      compactFeature(feature, {
        name:
          feature.properties?.name_en ??
          feature.properties?.name ??
          "Indian state or territory",
      }),
    ),
  }),
);

console.log(
  `Wrote India outline and ${indiaStates.length} state or territory boundaries.`,
);
