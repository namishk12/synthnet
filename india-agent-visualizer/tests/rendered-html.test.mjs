import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const projectRoot = new URL("../", import.meta.url);

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the Nexus India application shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(
    html,
    /<title>Nexus India \| Telecom Intelligence Platform<\/title>/i,
  );
  assert.match(html, /A unified local-first platform for synthetic telecom generation/i);
  assert.match(html, /Telecom intelligence, in one connected workspace/i);
  assert.match(html, /Run full pipeline/i);
  assert.match(html, /India explorer/i);
  assert.doesNotMatch(html, /codex-preview/i);
  assert.doesNotMatch(html, /Your site is taking shape|react-loading-skeleton/i);
});

test("ships the unified workspaces, refreshed dataset, and interactive India map", async () => {
  const [
    page,
    explorerPage,
    explorerWorkspace,
    layout,
    mapSource,
    packageJson,
    indexText,
  ] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/explorer/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../components/ExplorerWorkspace.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../components/IndiaMap.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
    readFile(new URL("../public/data/agent-index.json", import.meta.url), "utf8"),
  ]);
  const index = JSON.parse(indexText);

  assert.match(page, /HomeDashboard/);
  assert.match(explorerPage, /ExplorerWorkspace/);
  assert.match(explorerWorkspace, /IndiaAgentDashboard/);
  assert.match(layout, /Nexus India \| Telecom Intelligence Platform/);
  assert.match(mapSource, /fitEntireIndia\(map, indiaBounds, false\)/);
  assert.match(mapSource, /dragging:\s*true/);
  assert.match(mapSource, /scrollWheelZoom:\s*true/);
  assert.match(mapSource, /Reset map to show the whole of India/);
  assert.doesNotMatch(mapSource, /setMaxBounds|maxBoundsViscosity/);
  assert.match(packageJson, /"leaflet"/);
  assert.equal(index.totals.agents, 100);
  assert.equal(index.totals.calls, 100_000);
  assert.equal(index.totals.sessions, 100_000);
  assert.equal(index.agents.length, 100);
  assert.ok(
    index.agents.every(
      (agent) =>
        agent.location.lat >= 6 &&
        agent.location.lat <= 38.5 &&
        agent.location.lng >= 67 &&
        agent.location.lng <= 98.5,
    ),
  );

  const firstAgent = index.agents[0];
  const firstDetail = JSON.parse(
    await readFile(
      new URL(
        `../public/data/agents/${firstAgent.id}.json`,
        import.meta.url,
      ),
      "utf8",
    ),
  );
  assert.ok(firstDetail.profile.state_name);
  assert.ok(firstDetail.profile.state_code);
  assert.ok(firstDetail.schemas.callColumns.includes("caller_state_name"));
  assert.ok(firstDetail.schemas.callColumns.includes("receiver_state_name"));
  assert.ok(firstDetail.schemas.callColumns.includes("same_state_interaction"));
  assert.equal(firstDetail.calls.outgoing.length, 1_000);
  assert.equal(firstDetail.sessions.length, 1_000);

  for (const route of [
    "pipeline",
    "generator",
    "correlator",
    "community",
    "models",
    "explorer",
    "datasets",
    "reports",
  ]) {
    await access(new URL(`../app/${route}/page.tsx`, import.meta.url));
  }
  await assert.rejects(access(new URL("../app/_sites-preview", import.meta.url)));
  await assert.rejects(
    access(new URL("public/_sites-preview", projectRoot)),
  );
});
