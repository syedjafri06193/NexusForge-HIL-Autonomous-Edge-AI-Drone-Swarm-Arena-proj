import { create } from 'zustand';

export type TeamName = 'RED' | 'BLUE' | 'GREEN' | 'GOLD';

export interface DroneState {
  id: string;
  team: TeamName;
  state: string;
  position: { x: number; y: number };
  velocity: { x: number; y: number };
  heading: number;
  health: number;
  shield: number;
  battery_pct: number;
  kills: number;
  hits_taken: number;
  latency_ms: number;
  inference_ms: number;
  weapon_ready: boolean;
}

export interface Projectile {
  id: string; team: string;
  x: number; y: number;
}

export interface Hazard {
  id: string; type: string;
  x: number; y: number;
  radius: number; intensity: number; duration: number;
}

export interface ControlPoint {
  id: string; x: number; y: number; r: number;
  owner: string | null; capture: number;
}

export interface ArenaState {
  width: number; height: number;
  obstacles: { x: number; y: number; w: number; h: number }[];
  control_points: ControlPoint[];
  hazards: Hazard[];
}

export interface SimEvent {
  type: string; killer?: string; victim?: string;
  victim_team?: string; tick: number;
}

export interface HilTelemetry {
  drone_id: string; latency_ms: number;
  power?: { battery_pct: number; power_mw: number };
  compute?: { cpu_load_pct: number; inference_us: number };
  radio?: { rssi_dbm: number; packet_loss_pct: number };
}

export interface SwarmTeamStatus {
  mission: string; formation: string; alive: number;
}

interface NexusStore {
  // Connection
  sessionId: string | null;
  connected: boolean;
  ws: WebSocket | null;

  // Sim state
  tick: number;
  elapsed: number;
  scores: Record<TeamName, number>;
  drones: DroneState[];
  projectiles: Projectile[];
  arena: ArenaState | null;
  events: SimEvent[];
  leaderboard: any[];

  // HIL
  hilTelemetry: HilTelemetry[];
  hilFleet: Record<string, any>;

  // Swarm AI
  swarmStatus: { teams: Record<TeamName, SwarmTeamStatus>; recent_commands: any[] } | null;

  // UI state
  selectedDroneId: string | null;
  showTelemetry: boolean;
  showBenchmark: boolean;
  commandText: string;
  commandTeam: TeamName;
  paused: boolean;
  viewMode: '2d' | '3d';

  // Actions
  setSession: (id: string) => void;
  setConnected: (v: boolean) => void;
  setWs: (ws: WebSocket | null) => void;
  applySnapshot: (snap: any) => void;
  selectDrone: (id: string | null) => void;
  setCommandText: (t: string) => void;
  setCommandTeam: (t: TeamName) => void;
  setShowTelemetry: (v: boolean) => void;
  setShowBenchmark: (v: boolean) => void;
  setViewMode: (v: '2d' | '3d') => void;
  sendCommand: (text: string) => void;
  disconnect: () => void;
}

export const useNexusStore = create<NexusStore>((set, get) => ({
  sessionId: null,
  connected: false,
  ws: null,
  tick: 0,
  elapsed: 0,
  scores: { RED: 0, BLUE: 0, GREEN: 0, GOLD: 0 },
  drones: [],
  projectiles: [],
  arena: null,
  events: [],
  leaderboard: [],
  hilTelemetry: [],
  hilFleet: {},
  swarmStatus: null,
  selectedDroneId: null,
  showTelemetry: false,
  showBenchmark: false,
  commandText: '',
  commandTeam: 'RED',
  paused: false,
  viewMode: '2d',

  setSession: (id) => set({ sessionId: id }),
  setConnected: (v) => set({ connected: v }),
  setWs: (ws) => set({ ws }),

  applySnapshot: (snap) => set({
    tick: snap.tick ?? 0,
    elapsed: snap.elapsed ?? 0,
    scores: snap.scores ?? {},
    drones: snap.drones ?? [],
    projectiles: snap.projectiles ?? [],
    arena: snap.arena ?? null,
    events: snap.events ?? [],
    hilTelemetry: snap.hil_telemetry ?? [],
    hilFleet: snap.hil_fleet ?? {},
    swarmStatus: snap.swarm_status ?? null,
  }),

  selectDrone: (id) => set({ selectedDroneId: id }),
  setCommandText: (t) => set({ commandText: t }),
  setCommandTeam: (t) => set({ commandTeam: t }),
  setShowTelemetry: (v) => set({ showTelemetry: v }),
  setShowBenchmark: (v) => set({ showBenchmark: v }),
  setViewMode: (v) => set({ viewMode: v }),

  sendCommand: (text) => {
    const { ws, commandTeam } = get();
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: 'command', text, team: commandTeam }));
  },

  disconnect: () => {
    const { ws } = get();
    ws?.close();
    set({ ws: null, connected: false, sessionId: null });
  },
}));
