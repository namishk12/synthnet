export type AgentStats = {
  outgoingCalls: number;
  incomingCalls: number;
  totalCallSeconds: number;
  uniqueContacts: number;
  knownFriends: number;
  coreContacts: number;
  sessions: number;
  totalDataMb: number;
  totalUploadKb: number;
  totalDownloadKb: number;
  friendEdges: number;
  ganFriendEdges: number;
  locationPoints: number;
  latestActivity: string;
};

export type AgentSummary = {
  id: string;
  name: string;
  msisdn: string;
  circle: string;
  tower: string;
  device: string;
  connection: string;
  simType: string;
  location: { lat: number; lng: number };
  stats: AgentStats;
};

export type DashboardIndex = {
  generatedAt: string;
  dataset: string;
  totals: {
    agents: number;
    calls: number;
    sessions: number;
    friendEdges: number;
    incomingCoreCalls: number;
    totalDataMb: number;
  };
  hourlyCalls: number[];
  hourlySessions: number[];
  agents: AgentSummary[];
};

export type Contact = {
  msisdn: string;
  name: string;
  subscriberId: string;
  contactType: string;
  outgoingCount: number;
  incomingCount: number;
  durationSeconds: number;
  lastSeen: string;
  lastCallType: string;
  isKnownFriend: boolean;
  friendScore: number | null;
  friendSource: string;
};

export type FriendEdge = {
  subscriberId: string;
  name: string;
  msisdn: string;
  source: string;
  score: number | null;
  ipdrCorrelationScore: number | null;
  observed: boolean;
  ganInferred: boolean;
  sameState: boolean;
  truthSource: string;
  evidence: Record<string, string>;
};

export type LocationPoint = {
  lat: number;
  lng: number;
  tower: string;
  count: number;
  lastSeen: string;
};

export type AgentDetail = {
  profile: Record<string, string>;
  index: AgentSummary;
  stats: AgentStats;
  contacts: Contact[];
  friendEdges: FriendEdge[];
  schemas: {
    callColumns: string[];
    sessionColumns: string[];
  };
  calls: {
    outgoing: string[][];
    incoming: string[][];
  };
  sessions: string[][];
  locations: LocationPoint[];
  charts: {
    hourlyCalls: number[];
    hourlySessions: number[];
    callTypes: Array<{ label: string; value: number }>;
    destinationPorts: Array<{ label: string; value: number }>;
    destinationIps: Array<{ label: string; value: number }>;
    apns: Array<{ label: string; value: number }>;
  };
};
