"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { LatLngBounds, LayerGroup, Map as LeafletMap } from "leaflet";
import type { GeoJsonObject } from "geojson";
import type { AgentDetail, AgentSummary } from "./dashboard-types";

type LeafletModule = typeof import("leaflet");

const INDIA_OUTLINE_URL = "/data/india-outline.geojson";
const INDIA_STATES_URL = "/data/india-states.geojson";
const AGENT_FOCUS_ZOOM = 9;
const AGENT_FOCUS_DURATION = 1.05;
const INDIA_FIT_PADDING = 38;

type IndiaMapProps = {
  agents: AgentSummary[];
  selectedAgent: AgentSummary | null;
  detail: AgentDetail | null;
  focusRevision: number;
  onSelectAgent: (id: string) => void;
};

function escapeHtml(value: string) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function fitEntireIndia(
  map: LeafletMap,
  bounds: LatLngBounds,
  animate: boolean,
) {
  map.stop();
  map.invalidateSize({ pan: false });
  map.fitBounds(bounds, {
    paddingTopLeft: [INDIA_FIT_PADDING, INDIA_FIT_PADDING],
    paddingBottomRight: [INDIA_FIT_PADDING, INDIA_FIT_PADDING],
    animate,
    duration: animate ? 0.65 : undefined,
  });
}

export function IndiaMap({
  agents,
  selectedAgent,
  detail,
  focusRevision,
  onSelectAgent,
}: IndiaMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<LeafletMap | null>(null);
  const indiaBoundsRef = useRef<LatLngBounds | null>(null);
  const leafletRef = useRef<LeafletModule | null>(null);
  const agentsLayerRef = useRef<LayerGroup | null>(null);
  const networkLayerRef = useRef<LayerGroup | null>(null);
  const activityLayerRef = useRef<LayerGroup | null>(null);
  const previousSelectionRef = useRef("");
  const [ready, setReady] = useState(false);
  const [mapError, setMapError] = useState(false);

  const agentById = useMemo(
    () => new Map(agents.map((agent) => [agent.id, agent])),
    [agents],
  );

  useEffect(() => {
    let cancelled = false;
    let initialFitFrame = 0;

    async function initialise() {
      if (!containerRef.current || mapRef.current) return;
      try {
        const [L, outlineResponse, statesResponse] = await Promise.all([
          import("leaflet"),
          fetch(INDIA_OUTLINE_URL),
          fetch(INDIA_STATES_URL),
        ]);
        if (!outlineResponse.ok || !statesResponse.ok) {
          throw new Error("The India geography assets could not be loaded.");
        }
        const indiaOutline = (await outlineResponse.json()) as GeoJsonObject;
        const indiaStates = (await statesResponse.json()) as GeoJsonObject;
        if (cancelled || !containerRef.current) return;

        const map = L.map(containerRef.current, {
          zoomControl: false,
          attributionControl: false,
          dragging: true,
          scrollWheelZoom: true,
          doubleClickZoom: true,
          touchZoom: true,
          boxZoom: true,
          keyboard: true,
          minZoom: 3,
          maxZoom: 11,
          preferCanvas: true,
          zoomDelta: 0.5,
          zoomSnap: 0.5,
          wheelPxPerZoomLevel: 90,
        });

        const haloPane = map.createPane("india-halo");
        haloPane.style.zIndex = "180";
        const countryPane = map.createPane("india-country");
        countryPane.style.zIndex = "200";
        const statesPane = map.createPane("india-states");
        statesPane.style.zIndex = "210";

        L.geoJSON(indiaOutline, {
          pane: "india-halo",
          interactive: false,
          style: {
            color: "#49d6d0",
            weight: 18,
            opacity: 0.1,
            fillOpacity: 0,
          },
        }).addTo(map);

        const countryLayer = L.geoJSON(indiaOutline, {
          pane: "india-country",
          interactive: false,
          style: {
            color: "#4d7d86",
            weight: 1.8,
            opacity: 0.95,
            fillColor: "#102830",
            fillOpacity: 0.98,
          },
        }).addTo(map);

        L.geoJSON(indiaStates, {
          pane: "india-states",
          interactive: false,
          style: {
            color: "#52717a",
            weight: 0.72,
            opacity: 0.72,
            fillOpacity: 0,
          },
        }).addTo(map);

        const indiaBounds = countryLayer.getBounds();
        indiaBoundsRef.current = indiaBounds;
        fitEntireIndia(map, indiaBounds, false);

        L.control.zoom({ position: "bottomright" }).addTo(map);
        L.control.scale({ position: "bottomleft", imperial: false }).addTo(map);

        leafletRef.current = L;
        mapRef.current = map;
        agentsLayerRef.current = L.layerGroup().addTo(map);
        networkLayerRef.current = L.layerGroup().addTo(map);
        activityLayerRef.current = L.layerGroup().addTo(map);
        setReady(true);

        // Leaflet can measure the grid cell before the surrounding dashboard
        // has completed layout. Re-measure on the next paint and fit the whole
        // country again so India is always the initial viewport.
        initialFitFrame = window.requestAnimationFrame(() => {
          if (!cancelled) fitEntireIndia(map, indiaBounds, false);
        });
      } catch {
        setMapError(true);
      }
    }

    initialise();
    return () => {
      cancelled = true;
      window.cancelAnimationFrame(initialFitFrame);
      mapRef.current?.remove();
      mapRef.current = null;
      indiaBoundsRef.current = null;
    };
  }, []);

  function showEntireIndia() {
    const map = mapRef.current;
    const bounds = indiaBoundsRef.current;
    if (!map || !bounds) return;
    fitEntireIndia(map, bounds, true);
  }

  function fitActiveAgents() {
    const L = leafletRef.current;
    const map = mapRef.current;
    if (!L || !map) return;
    const coordinates = agents
      .map((agent) => [agent.location.lat, agent.location.lng] as [number, number])
      .filter(([latitude, longitude]) => (
        Number.isFinite(latitude)
        && Number.isFinite(longitude)
      ));
    if (!coordinates.length) {
      showEntireIndia();
      return;
    }
    if (coordinates.length === 1) {
      map.flyTo(coordinates[0], AGENT_FOCUS_ZOOM, {
        duration: AGENT_FOCUS_DURATION,
      });
      return;
    }
    map.stop();
    map.fitBounds(L.latLngBounds(coordinates), {
      padding: [50, 50],
      maxZoom: 8,
      animate: true,
      duration: 0.65,
    });
  }

  useEffect(() => {
    const L = leafletRef.current;
    const map = mapRef.current;
    const agentsLayer = agentsLayerRef.current;
    const networkLayer = networkLayerRef.current;
    const activityLayer = activityLayerRef.current;
    if (!ready || !L || !map || !agentsLayer || !networkLayer || !activityLayer) {
      return;
    }

    agentsLayer.clearLayers();
    networkLayer.clearLayers();
    activityLayer.clearLayers();

    for (const agent of agents) {
      const selected = agent.id === selectedAgent?.id;
      const marker = L.circleMarker([agent.location.lat, agent.location.lng], {
        radius: selected ? 10 : 5.5,
        color: selected ? "#fff3dc" : "#8f9cab",
        weight: selected ? 3 : 1,
        fillColor: selected ? "#ffb54a" : "#536575",
        fillOpacity: selected ? 1 : 0.78,
      });
      marker.bindTooltip(
        `<div class="map-tooltip"><strong>${escapeHtml(agent.name)}</strong><span>${escapeHtml(agent.id)} · ${escapeHtml(agent.msisdn)}</span></div>`,
        { direction: "top", offset: [0, -6] },
      );
      marker.on("click", () => onSelectAgent(agent.id));
      marker.addTo(agentsLayer);
    }

    if (selectedAgent && detail) {
      for (const point of detail.locations) {
        const radius = Math.min(14, 4 + Math.sqrt(point.count) * 0.7);
        const marker = L.circleMarker([point.lat, point.lng], {
          radius,
          color: "#ffb54a",
          weight: 1.4,
          fillColor: "#ffb54a",
          fillOpacity: 0.18,
        });
        marker.bindTooltip(
          `<div class="map-tooltip"><strong>${escapeHtml(point.tower)}</strong><span>${point.count.toLocaleString()} events · ${escapeHtml(point.lastSeen)}</span></div>`,
          { direction: "top" },
        );
        marker.addTo(activityLayer);
      }

      for (const contact of detail.contacts) {
        if (!contact.subscriberId) continue;
        const otherAgent = agentById.get(contact.subscriberId);
        if (!otherAgent) continue;
        const reciprocal = contact.incomingCount > 0 && contact.outgoingCount > 0;
        const incomingOnly = contact.incomingCount > 0 && contact.outgoingCount === 0;
        const colour = reciprocal ? "#b799ff" : incomingOnly ? "#49d6d0" : "#ffb54a";
        const total = contact.incomingCount + contact.outgoingCount;
        const line = L.polyline(
          [
            [selectedAgent.location.lat, selectedAgent.location.lng],
            [otherAgent.location.lat, otherAgent.location.lng],
          ],
          {
            color: colour,
            weight: Math.min(5, 1.2 + Math.log10(total + 1)),
            opacity: 0.72,
            dashArray: incomingOnly ? "5 7" : undefined,
          },
        );
        line.bindTooltip(
          `<div class="map-tooltip"><strong>${escapeHtml(contact.name)}</strong><span>${contact.outgoingCount} out · ${contact.incomingCount} in</span></div>`,
          { sticky: true },
        );
        line.addTo(networkLayer);
      }
    }

    const focusToken = selectedAgent
      ? `${selectedAgent.id}:${focusRevision}`
      : "";
    if (
      selectedAgent &&
      focusRevision > 0 &&
      previousSelectionRef.current !== focusToken
    ) {
      map.flyTo(
        [selectedAgent.location.lat, selectedAgent.location.lng],
        Math.max(AGENT_FOCUS_ZOOM, map.getZoom()),
        {
          duration: AGENT_FOCUS_DURATION,
          easeLinearity: 0.22,
        },
      );
      previousSelectionRef.current = focusToken;
    }
  }, [
    agentById,
    agents,
    detail,
    focusRevision,
    onSelectAgent,
    ready,
    selectedAgent,
  ]);

  return (
    <div className="map-shell" aria-label="India agent activity map">
      <div ref={containerRef} className="india-map" />
      {mapError ? (
        <div className="map-error">
          <strong>The India map could not be loaded.</strong>
          <span>The agent data and coordinates remain available in the profile panel.</span>
        </div>
      ) : null}
      <div className="map-readout">
        <span className="eyebrow">INDIA-ONLY NETWORK VIEW</span>
        <strong>{selectedAgent ? selectedAgent.name : "All synthetic agents"}</strong>
        <small>
          {selectedAgent
            ? `${selectedAgent.location.lat.toFixed(5)}, ${selectedAgent.location.lng.toFixed(5)}`
            : `${agents.length} mapped agents`}
        </small>
      </div>
      <button
        type="button"
        className="map-coverage-note"
        aria-label="Reset map to show the whole of India"
        onClick={showEntireIndia}
      >
        <span>SHOW ALL INDIA</span>
        <small>Reset map view</small>
      </button>
      <button
        type="button"
        className="map-agents-note"
        aria-label="Fit map to active agents"
        onClick={fitActiveAgents}
      >
        <span>FIT ACTIVE AGENTS</span>
        <small>{agents.length} mapped profiles</small>
      </button>
      <div className="map-navigation-hint">
        Drag to move <b>·</b> Scroll to zoom
      </div>
      <div className="map-legend" aria-label="Map legend">
        <span><i className="legend-dot selected" /> Selected</span>
        <span><i className="legend-line outgoing" /> Outgoing</span>
        <span><i className="legend-line incoming" /> Incoming</span>
        <span><i className="legend-line reciprocal" /> Both</span>
      </div>
    </div>
  );
}
