"use client";

import {
  Activity,
  ArrowDownLeft,
  ArrowUpRight,
  ChevronLeft,
  ChevronRight,
  CircleDot,
  Clock3,
  Database,
  Eye,
  FileDown,
  HardDrive,
  MapPin,
  Network,
  PhoneCall,
  RadioTower,
  Search,
  Server,
  ShieldCheck,
  Smartphone,
  Users,
  Wifi,
  X,
} from "lucide-react";
import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { IndiaMap } from "./IndiaMap";
import { nexusApiUrl } from "../lib/nexus-api";
import type {
  AgentDetail,
  Contact,
  DashboardIndex,
  FriendEdge,
} from "./dashboard-types";

type ViewTab = "profile" | "contacts" | "calls" | "sessions" | "friends";
type DirectionFilter = "all" | "incoming" | "outgoing" | "reciprocal";
type EventRecord = Record<string, string>;

const PAGE_SIZE = 24;

const tabs: Array<{ id: ViewTab; label: string }> = [
  { id: "profile", label: "Profile" },
  { id: "contacts", label: "Contacts" },
  { id: "calls", label: "CDR" },
  { id: "sessions", label: "IPDR" },
  { id: "friends", label: "Friends" },
];

function formatNumber(value: number) {
  return new Intl.NumberFormat("en-IN").format(value);
}

function formatDecimal(value: number, digits = 1) {
  return new Intl.NumberFormat("en-IN", {
    maximumFractionDigits: digits,
  }).format(value);
}

function formatData(megabytes: number) {
  if (megabytes >= 1_000_000) return `${formatDecimal(megabytes / 1_000_000, 2)} TB`;
  if (megabytes >= 1_000) return `${formatDecimal(megabytes / 1_000, 2)} GB`;
  return `${formatDecimal(megabytes, 1)} MB`;
}

function formatDuration(seconds: number) {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  if (hours) return `${hours}h ${minutes}m`;
  if (minutes) return `${minutes}m ${secs}s`;
  return `${secs}s`;
}

function formatTimestamp(value: string) {
  if (!value) return "—";
  const parsed = new Date(value.replace(" ", "T"));
  if (Number.isNaN(parsed.getTime())) return value;
  return new Intl.DateTimeFormat("en-IN", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(parsed);
}

function labelForField(value: string) {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase())
    .replace("Msisdn", "MSISDN")
    .replace("Imei", "IMEI")
    .replace("Imsi", "IMSI")
    .replace("Apn", "APN")
    .replace("Ip ", "IP ");
}

function initials(name: string) {
  const parts = name.split(/\s+/).filter(Boolean);
  if (!parts.length) return "NA";
  return `${parts[0][0] ?? ""}${parts[1]?.[0] ?? ""}`.toUpperCase();
}

function decodeRows(columns: string[], rows: string[][]): EventRecord[] {
  return rows.map((row) =>
    Object.fromEntries(columns.map((column, index) => [column, row[index] ?? ""])),
  );
}

function Stat({
  icon,
  label,
  value,
  accent,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  accent?: "amber" | "cyan" | "violet";
}) {
  return (
    <div className={`stat-block ${accent ? `stat-${accent}` : ""}`}>
      <span className="stat-icon">{icon}</span>
      <div>
        <small>{label}</small>
        <strong>{value}</strong>
      </div>
    </div>
  );
}

function Pagination({
  page,
  total,
  onPage,
}: {
  page: number;
  total: number;
  onPage: (page: number) => void;
}) {
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));
  if (pageCount <= 1) return null;
  return (
    <div className="pagination">
      <span>
        Page {page + 1} / {pageCount}
      </span>
      <div>
        <button
          type="button"
          aria-label="Previous page"
          disabled={page === 0}
          onClick={() => onPage(Math.max(0, page - 1))}
        >
          <ChevronLeft size={16} />
        </button>
        <button
          type="button"
          aria-label="Next page"
          disabled={page + 1 >= pageCount}
          onClick={() => onPage(Math.min(pageCount - 1, page + 1))}
        >
          <ChevronRight size={16} />
        </button>
      </div>
    </div>
  );
}

function HourlyChart({
  calls,
  sessions,
}: {
  calls: number[];
  sessions: number[];
}) {
  const maxValue = Math.max(1, ...calls, ...sessions);
  return (
    <div className="hour-chart" aria-label="Hourly call and session activity">
      <div className="chart-key">
        <span><i className="chart-dot calls" /> CDR</span>
        <span><i className="chart-dot sessions" /> IPDR</span>
      </div>
      <div className="hour-bars">
        {calls.map((value, hour) => (
          <div className="hour-group" key={hour} title={`${hour}:00 · ${value} calls · ${sessions[hour]} sessions`}>
            <div className="bar-pair">
              <i
                className="bar calls"
                style={{ height: `${Math.max(2, (value / maxValue) * 100)}%` }}
              />
              <i
                className="bar sessions"
                style={{ height: `${Math.max(2, (sessions[hour] / maxValue) * 100)}%` }}
              />
            </div>
            <span>{hour % 4 === 0 ? `${hour.toString().padStart(2, "0")}` : ""}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function RankedList({
  title,
  items,
}: {
  title: string;
  items: Array<{ label: string; value: number }>;
}) {
  const maxValue = Math.max(1, ...items.map((item) => item.value));
  return (
    <div className="ranked-list">
      <div className="section-kicker">{title}</div>
      {items.slice(0, 6).map((item) => (
        <div className="rank-row" key={item.label}>
          <span title={item.label}>{item.label}</span>
          <div><i style={{ width: `${(item.value / maxValue) * 100}%` }} /></div>
          <strong>{formatNumber(item.value)}</strong>
        </div>
      ))}
      {!items.length ? <div className="empty-inline">No values available</div> : null}
    </div>
  );
}

function DirectionBadge({ contact }: { contact: Contact }) {
  const incoming = contact.incomingCount > 0;
  const outgoing = contact.outgoingCount > 0;
  if (incoming && outgoing) {
    return <span className="direction-pill reciprocal">Both directions</span>;
  }
  if (incoming) return <span className="direction-pill incoming">Incoming</span>;
  return <span className="direction-pill outgoing">Outgoing</span>;
}

function LoadingPanel() {
  return (
    <div className="loading-panel">
      <div className="loading-orbit"><CircleDot size={28} /></div>
      <strong>Linking the network</strong>
      <span>Loading the selected agent’s full CDR and IPDR history…</span>
    </div>
  );
}

export function IndiaAgentDashboard({ runId = "" }: { runId?: string }) {
  const [indexData, setIndexData] = useState<DashboardIndex | null>(null);
  const [selectedId, setSelectedId] = useState("");
  const [focusRevision, setFocusRevision] = useState(0);
  const [detail, setDetail] = useState<AgentDetail | null>(null);
  const [search, setSearch] = useState("");
  const [activeTab, setActiveTab] = useState<ViewTab>("profile");
  const [contactFilter, setContactFilter] = useState<DirectionFilter>("all");
  const [callFilter, setCallFilter] = useState<"all" | "incoming" | "outgoing">("all");
  const [contactPage, setContactPage] = useState(0);
  const [callPage, setCallPage] = useState(0);
  const [sessionPage, setSessionPage] = useState(0);
  const [friendPage, setFriendPage] = useState(0);
  const [inspector, setInspector] = useState<{ title: string; record: EventRecord } | null>(null);
  const [loadError, setLoadError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    const suffix = runId ? `?run_id=${encodeURIComponent(runId)}` : "";
    fetch(nexusApiUrl(`/explorer/index${suffix}`), { signal: controller.signal })
      .then((response) => {
        if (!response.ok && !runId) return fetch("/data/agent-index.json", { signal: controller.signal });
        if (!response.ok) throw new Error("The selected run has no explorer dataset.");
        return response;
      })
      .then((response) => {
        if (!response.ok) throw new Error("The dataset index could not be loaded.");
        return response.json() as Promise<DashboardIndex>;
      })
      .then((payload) => {
        setIndexData(payload);
        setDetail(null);
        setSelectedId(payload.agents[0]?.id ?? "");
      })
      .catch((error: Error) => {
        if (error.name !== "AbortError") setLoadError(error.message);
      });
    return () => controller.abort();
  }, [runId]);

  useEffect(() => {
    if (!selectedId) return;
    const controller = new AbortController();
    const suffix = runId ? `?run_id=${encodeURIComponent(runId)}` : "";
    fetch(nexusApiUrl(`/explorer/agents/${selectedId}${suffix}`), { signal: controller.signal })
      .then((response) => {
        if (!response.ok && !runId) {
          return fetch(`/data/agents/${selectedId}.json`, { signal: controller.signal });
        }
        return response;
      })
      .then((response) => {
        if (!response.ok) throw new Error(`Agent profile ${selectedId} could not be loaded.`);
        return response.json() as Promise<AgentDetail>;
      })
      .then(setDetail)
      .catch((error: Error) => {
        if (error.name !== "AbortError") setLoadError(error.message);
      });
    return () => controller.abort();
  }, [runId, selectedId]);

  useEffect(() => {
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") setInspector(null);
    }
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, []);

  const selectAgent = useCallback((id: string) => {
    setDetail(null);
    setSelectedId(id);
    setFocusRevision((revision) => revision + 1);
    setActiveTab("profile");
    setContactPage(0);
    setCallPage(0);
    setSessionPage(0);
    setFriendPage(0);
    setInspector(null);
  }, []);

  const selectedAgent = useMemo(
    () => indexData?.agents.find((agent) => agent.id === selectedId) ?? null,
    [indexData, selectedId],
  );

  const visibleAgents = useMemo(() => {
    const query = search.trim().toLowerCase();
    if (!indexData) return [];
    if (!query) return indexData.agents;
    return indexData.agents.filter((agent) =>
      [agent.id, agent.name, agent.msisdn, agent.circle, agent.tower]
        .join(" ")
        .toLowerCase()
        .includes(query),
    );
  }, [indexData, search]);

  const outgoingRecords = useMemo(
    () =>
      detail
        ? decodeRows(detail.schemas.callColumns, detail.calls.outgoing).map((record) => ({
            ...record,
            _direction: "outgoing",
          }))
        : [],
    [detail],
  );
  const incomingRecords = useMemo(
    () =>
      detail
        ? decodeRows(detail.schemas.callColumns, detail.calls.incoming).map((record) => ({
            ...record,
            _direction: "incoming",
          }))
        : [],
    [detail],
  );
  const sessionRecords = useMemo(
    () => (detail ? decodeRows(detail.schemas.sessionColumns, detail.sessions) : []),
    [detail],
  );

  const filteredContacts = useMemo(() => {
    if (!detail) return [];
    if (contactFilter === "incoming") {
      return detail.contacts.filter((contact) => contact.incomingCount > 0);
    }
    if (contactFilter === "outgoing") {
      return detail.contacts.filter((contact) => contact.outgoingCount > 0);
    }
    if (contactFilter === "reciprocal") {
      return detail.contacts.filter(
        (contact) => contact.incomingCount > 0 && contact.outgoingCount > 0,
      );
    }
    return detail.contacts;
  }, [contactFilter, detail]);

  const filteredCalls = useMemo(() => {
    if (callFilter === "incoming") return incomingRecords;
    if (callFilter === "outgoing") return outgoingRecords;
    return [...outgoingRecords, ...incomingRecords].sort((left, right) =>
      (right.timestamp ?? "").localeCompare(left.timestamp ?? ""),
    );
  }, [callFilter, incomingRecords, outgoingRecords]);

  const pagedContacts = filteredContacts.slice(
    contactPage * PAGE_SIZE,
    (contactPage + 1) * PAGE_SIZE,
  );
  const pagedCalls = filteredCalls.slice(
    callPage * PAGE_SIZE,
    (callPage + 1) * PAGE_SIZE,
  );
  const pagedSessions = sessionRecords.slice(
    sessionPage * PAGE_SIZE,
    (sessionPage + 1) * PAGE_SIZE,
  );
  const pagedFriends = (detail?.friendEdges ?? []).slice(
    friendPage * PAGE_SIZE,
    (friendPage + 1) * PAGE_SIZE,
  );

  if (loadError) {
    return (
      <main className="fatal-state">
        <ShieldCheck size={32} />
        <h1>Nexus India could not load its data.</h1>
        <p>{loadError}</p>
      </main>
    );
  }

  if (!indexData) {
    return (
      <main className="boot-screen">
        <div className="boot-mark">NI</div>
        <span>Preparing India telecom intelligence</span>
      </main>
    );
  }

  return (
    <main className="dashboard">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">NI</span>
          <div>
            <strong>NEXUS INDIA</strong>
            <small>Synthetic telecom intelligence</small>
          </div>
        </div>
        <div className="top-metrics">
          <div><span>Agents</span><strong>{formatNumber(indexData.totals.agents)}</strong></div>
          <div><span>CDR events</span><strong>{formatNumber(indexData.totals.calls)}</strong></div>
          <div><span>IPDR sessions</span><strong>{formatNumber(indexData.totals.sessions)}</strong></div>
          <div><span>Graph edges</span><strong>{formatNumber(indexData.totals.friendEdges)}</strong></div>
        </div>
        <div className="dataset-status">
          <i />
          <span>{runId ? runId : "LOCAL SYNTHETIC DATA"}</span>
        </div>
      </header>

      <section className="workspace">
        <aside className="agent-rail">
          <div className="rail-heading">
            <div>
              <span className="eyebrow">AGENT DIRECTORY</span>
              <strong>{visibleAgents.length} profiles</strong>
            </div>
            <Users size={19} />
          </div>
          <label className="search-box">
            <Search size={17} />
            <input
              type="search"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Name, MSISDN or agent ID"
              aria-label="Search agents"
            />
            {search ? (
              <button type="button" aria-label="Clear search" onClick={() => setSearch("")}>
                <X size={14} />
              </button>
            ) : null}
          </label>
          <div className="agent-list">
            {visibleAgents.map((agent) => {
              const selected = agent.id === selectedId;
              return (
                <button
                  type="button"
                  className={`agent-row ${selected ? "selected" : ""}`}
                  key={agent.id}
                  onClick={() => selectAgent(agent.id)}
                  aria-pressed={selected}
                >
                  <span className="agent-avatar">{initials(agent.name)}</span>
                  <span className="agent-copy">
                    <strong>{agent.name}</strong>
                    <small>{agent.id} · {agent.circle}</small>
                  </span>
                  <span className="agent-activity">
                    {formatNumber(agent.stats.outgoingCalls + agent.stats.incomingCalls)}
                    <small>calls</small>
                  </span>
                </button>
              );
            })}
            {!visibleAgents.length ? (
              <div className="rail-empty">No agents match that search.</div>
            ) : null}
          </div>
          <div className="rail-footer">
            <ShieldCheck size={16} />
            <span>All identities and events shown here are synthetic.</span>
          </div>
        </aside>

        <section className="map-column">
          <IndiaMap
            agents={indexData.agents}
            selectedAgent={selectedAgent}
            detail={detail}
            focusRevision={focusRevision}
            onSelectAgent={selectAgent}
          />
          <div className="map-summary">
            <div>
              <span className="eyebrow">DATASET PULSE</span>
              <strong>{formatData(indexData.totals.totalDataMb)} transferred</strong>
            </div>
            <HourlyChart calls={indexData.hourlyCalls} sessions={indexData.hourlySessions} />
          </div>
        </section>

        <aside className="detail-panel">
          {!selectedAgent || !detail ? (
            <LoadingPanel />
          ) : (
            <>
              <div className="profile-head">
                <div className="profile-avatar">{initials(selectedAgent.name)}</div>
                <div className="profile-title">
                  <span className="eyebrow">SELECTED AGENT</span>
                  <h1>{selectedAgent.name}</h1>
                  <div>
                    <span>{selectedAgent.id}</span>
                    <i />
                    <span>{selectedAgent.msisdn}</span>
                  </div>
                </div>
                <a
                  className="dossier-link"
                  href={nexusApiUrl(
                    `/explorer/agents/${selectedAgent.id}/dossier${
                      runId ? `?run_id=${encodeURIComponent(runId)}` : ""
                    }`,
                  )}
                  target="_blank"
                  rel="noreferrer"
                  title="Download analytical PDF dossier"
                >
                  <FileDown size={15} />
                  PDF
                </a>
              </div>

              <div className="profile-strip">
                <span><MapPin size={14} /> {selectedAgent.circle}</span>
                <span><RadioTower size={14} /> {selectedAgent.tower}</span>
                <span><Smartphone size={14} /> {selectedAgent.device}</span>
              </div>

              <nav className="detail-tabs" aria-label="Agent detail sections">
                {tabs.map((tab) => (
                  <button
                    type="button"
                    className={activeTab === tab.id ? "active" : ""}
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                  >
                    {tab.label}
                  </button>
                ))}
              </nav>

              <div className="detail-scroll">
                {activeTab === "profile" ? (
                  <div className="tab-content">
                    <div className="stat-grid">
                      <Stat
                        icon={<ArrowUpRight size={17} />}
                        label="Outgoing"
                        value={formatNumber(detail.stats.outgoingCalls)}
                        accent="amber"
                      />
                      <Stat
                        icon={<ArrowDownLeft size={17} />}
                        label="Incoming"
                        value={formatNumber(detail.stats.incomingCalls)}
                        accent="cyan"
                      />
                      <Stat
                        icon={<Wifi size={17} />}
                        label="Sessions"
                        value={formatNumber(detail.stats.sessions)}
                      />
                      <Stat
                        icon={<HardDrive size={17} />}
                        label="Data"
                        value={formatData(detail.stats.totalDataMb)}
                        accent="violet"
                      />
                    </div>

                    <section className="detail-section">
                      <div className="section-heading">
                        <div>
                          <span className="eyebrow">ACTIVITY RHYTHM</span>
                          <strong>24-hour signal</strong>
                        </div>
                        <Clock3 size={18} />
                      </div>
                      <HourlyChart
                        calls={detail.charts.hourlyCalls}
                        sessions={detail.charts.hourlySessions}
                      />
                    </section>

                    <section className="detail-section">
                      <div className="section-heading">
                        <div>
                          <span className="eyebrow">IDENTITY RECORD</span>
                          <strong>Complete subscriber profile</strong>
                        </div>
                        <Database size={18} />
                      </div>
                      <dl className="profile-fields">
                        {Object.entries(detail.profile).map(([field, value]) => (
                          <div key={field}>
                            <dt>{labelForField(field)}</dt>
                            <dd>{value || "—"}</dd>
                          </div>
                        ))}
                      </dl>
                    </section>

                    <section className="split-rankings">
                      <RankedList title="TOP DESTINATION PORTS" items={detail.charts.destinationPorts} />
                      <RankedList title="ACCESS POINTS" items={detail.charts.apns} />
                    </section>

                    <section className="detail-section compact">
                      <div className="section-heading">
                        <div>
                          <span className="eyebrow">NETWORK SUMMARY</span>
                          <strong>Relationship coverage</strong>
                        </div>
                        <Network size={18} />
                      </div>
                      <div className="network-facts">
                        <div><span>Unique contacts</span><strong>{formatNumber(detail.stats.uniqueContacts)}</strong></div>
                        <div><span>Core agents reached</span><strong>{formatNumber(detail.stats.coreContacts)}</strong></div>
                        <div><span>Known friends</span><strong>{formatNumber(detail.stats.knownFriends)}</strong></div>
                        <div><span>CGAN relationships</span><strong>{formatNumber(detail.stats.friendEdges)}</strong></div>
                        <div><span>Observed locations</span><strong>{formatNumber(detail.stats.locationPoints)}</strong></div>
                        <div><span>Total call time</span><strong>{formatDuration(detail.stats.totalCallSeconds)}</strong></div>
                      </div>
                    </section>
                  </div>
                ) : null}

                {activeTab === "contacts" ? (
                  <div className="tab-content">
                    <div className="tab-intro">
                      <div>
                        <span className="eyebrow">ALL CONNECTIONS</span>
                        <strong>{formatNumber(filteredContacts.length)} incoming / outgoing contacts</strong>
                      </div>
                      <PhoneCall size={19} />
                    </div>
                    <div className="filter-pills">
                      {(["all", "incoming", "outgoing", "reciprocal"] as DirectionFilter[]).map((filter) => (
                        <button
                          type="button"
                          key={filter}
                          className={contactFilter === filter ? "active" : ""}
                          onClick={() => {
                            setContactFilter(filter);
                            setContactPage(0);
                          }}
                        >
                          {filter}
                        </button>
                      ))}
                    </div>
                    <div className="contact-list">
                      {pagedContacts.map((contact) => (
                        <article className="contact-card" key={contact.msisdn}>
                          <div className="contact-avatar">{initials(contact.name)}</div>
                          <div className="contact-main">
                            <div>
                              <strong>{contact.name || "Unnamed contact"}</strong>
                              <DirectionBadge contact={contact} />
                            </div>
                            <small>{contact.subscriberId || contact.contactType} · {contact.msisdn}</small>
                            <div className="contact-metrics">
                              <span><ArrowUpRight size={13} /> {contact.outgoingCount}</span>
                              <span><ArrowDownLeft size={13} /> {contact.incomingCount}</span>
                              <span><Clock3 size={13} /> {formatDuration(contact.durationSeconds)}</span>
                            </div>
                          </div>
                          {contact.subscriberId ? (
                            <button
                              type="button"
                              className="inspect-link"
                              onClick={() => selectAgent(contact.subscriberId)}
                              aria-label={`Open ${contact.name}`}
                            >
                              <Eye size={16} />
                            </button>
                          ) : null}
                        </article>
                      ))}
                    </div>
                    <Pagination
                      page={contactPage}
                      total={filteredContacts.length}
                      onPage={setContactPage}
                    />
                  </div>
                ) : null}

                {activeTab === "calls" ? (
                  <div className="tab-content">
                    <div className="tab-intro">
                      <div>
                        <span className="eyebrow">CALL DETAIL RECORDS</span>
                        <strong>{formatNumber(filteredCalls.length)} complete events</strong>
                      </div>
                      <Server size={19} />
                    </div>
                    <div className="filter-pills">
                      {(["all", "incoming", "outgoing"] as const).map((filter) => (
                        <button
                          type="button"
                          key={filter}
                          className={callFilter === filter ? "active" : ""}
                          onClick={() => {
                            setCallFilter(filter);
                            setCallPage(0);
                          }}
                        >
                          {filter}
                        </button>
                      ))}
                    </div>
                    <div className="event-list">
                      {pagedCalls.map((record, index) => {
                        const incoming = record._direction === "incoming";
                        const otherName = incoming ? record.caller_name : record.receiver_name;
                        const otherMsisdn = incoming ? record.caller_msisdn : record.receiver_msisdn;
                        return (
                          <article className="event-row" key={`${record.call_id}-${index}`}>
                            <span className={`event-direction ${incoming ? "incoming" : "outgoing"}`}>
                              {incoming ? <ArrowDownLeft size={16} /> : <ArrowUpRight size={16} />}
                            </span>
                            <div className="event-main">
                              <div><strong>{otherName || "Unknown contact"}</strong><span>{formatTimestamp(record.timestamp)}</span></div>
                              <small>{otherMsisdn} · {incoming ? "IN" : "OUT"} · {record.service_type || record.call_type || "Unknown service"} · {formatDuration(Number(record.duration_seconds || 0))}</small>
                              <span>{record.first_cell_desc || record.routing_tower || "No cell data"}</span>
                            </div>
                            <button
                              type="button"
                              className="inspect-link"
                              onClick={() => setInspector({
                                title: `${incoming ? "Incoming" : "Outgoing"} CDR · ${record.call_id}`,
                                record,
                              })}
                              aria-label={`Inspect ${record.call_id}`}
                            >
                              <Eye size={16} />
                            </button>
                          </article>
                        );
                      })}
                    </div>
                    <Pagination page={callPage} total={filteredCalls.length} onPage={setCallPage} />
                  </div>
                ) : null}

                {activeTab === "sessions" ? (
                  <div className="tab-content">
                    <div className="tab-intro">
                      <div>
                        <span className="eyebrow">INTERNET PROTOCOL DETAIL</span>
                        <strong>{formatNumber(sessionRecords.length)} complete sessions</strong>
                      </div>
                      <Wifi size={19} />
                    </div>
                    <div className="event-list">
                      {pagedSessions.map((record, index) => (
                        <article className="event-row session" key={`${record.session_id}-${index}`}>
                          <span className="event-direction session"><Activity size={16} /></span>
                          <div className="event-main">
                            <div><strong>{record.destination_ip || "Unknown destination"}</strong><span>{formatTimestamp(record.timestamp)}</span></div>
                            <small>Port {record.destination_port || "—"} · {record.apn || "No APN"} · {formatData(Number(record.megabytes_transferred || 0))}</small>
                            <span>{record.source_ip_address || "No source IP"} → {record.translated_ip_address || "No NAT translation"}</span>
                          </div>
                          <button
                            type="button"
                            className="inspect-link"
                            onClick={() => setInspector({
                              title: `IPDR session · ${record.session_id}`,
                              record,
                            })}
                            aria-label={`Inspect ${record.session_id}`}
                          >
                            <Eye size={16} />
                          </button>
                        </article>
                      ))}
                    </div>
                    <Pagination
                      page={sessionPage}
                      total={sessionRecords.length}
                      onPage={setSessionPage}
                    />
                  </div>
                ) : null}

                {activeTab === "friends" ? (
                  <div className="tab-content">
                    <div className="tab-intro">
                      <div>
                        <span className="eyebrow">CGAN RELATIONSHIP GRAPH</span>
                        <strong>{formatNumber(detail.friendEdges.length)} modeled friends</strong>
                      </div>
                      <Network size={19} />
                    </div>
                    <div className="friend-list">
                      {pagedFriends.map((edge: FriendEdge) => (
                        <article className="friend-row" key={edge.subscriberId}>
                          <div className="contact-avatar">{initials(edge.name)}</div>
                          <div>
                            <strong>{edge.name}</strong>
                            <small>{edge.subscriberId} · {edge.msisdn}</small>
                            <span>
                              {typeof edge.ipdrCorrelationScore === "number"
                                ? edge.ganInferred
                                  ? "IPDR evidence + Conditional GAN"
                                  : "IPDR-inferred interaction"
                                : edge.observed
                                  ? "Observed training edge"
                                  : "Conditional GAN inference"}
                            </span>
                            {typeof edge.ipdrCorrelationScore === "number" ? (
                              <small>IPDR correlation {edge.ipdrCorrelationScore.toFixed(3)}</small>
                            ) : null}
                          </div>
                          <div className="friend-score">
                            <small>GAN score</small>
                            <strong>{edge.score === null ? "—" : edge.score.toFixed(3)}</strong>
                          </div>
                          <button
                            type="button"
                            className="inspect-link"
                            onClick={() => setInspector({
                              title: `Relationship evidence · ${edge.name}`,
                              record: {
                                relationship_source: edge.source,
                                truth_source: edge.truthSource,
                                gan_friend_score: edge.score?.toFixed(6) ?? "",
                                ipdr_correlation_score: edge.ipdrCorrelationScore?.toFixed(6) ?? "",
                                same_state: String(edge.sameState),
                                ...edge.evidence,
                              },
                            })}
                            aria-label={`Inspect relationship evidence with ${edge.name}`}
                          >
                            <ShieldCheck size={16} />
                          </button>
                          <button
                            type="button"
                            className="inspect-link"
                            onClick={() => selectAgent(edge.subscriberId)}
                            aria-label={`Open ${edge.name}`}
                          >
                            <Eye size={16} />
                          </button>
                        </article>
                      ))}
                    </div>
                    <Pagination
                      page={friendPage}
                      total={detail.friendEdges.length}
                      onPage={setFriendPage}
                    />
                  </div>
                ) : null}
              </div>
            </>
          )}
        </aside>
      </section>

      {inspector ? (
        <div className="inspector-backdrop" role="presentation" onMouseDown={() => setInspector(null)}>
          <section
            className="record-inspector"
            role="dialog"
            aria-modal="true"
            aria-label={inspector.title}
            onMouseDown={(event) => event.stopPropagation()}
          >
            <header>
              <div>
                <span className="eyebrow">COMPLETE RAW RECORD</span>
                <h2>{inspector.title}</h2>
              </div>
              <button type="button" aria-label="Close record" onClick={() => setInspector(null)}>
                <X size={19} />
              </button>
            </header>
            <dl>
              {Object.entries(inspector.record)
                .filter(([field]) => field !== "_direction")
                .map(([field, value]) => (
                  <div key={field}>
                    <dt>{labelForField(field)}</dt>
                    <dd>{value || "—"}</dd>
                  </div>
                ))}
            </dl>
          </section>
        </div>
      ) : null}
    </main>
  );
}
