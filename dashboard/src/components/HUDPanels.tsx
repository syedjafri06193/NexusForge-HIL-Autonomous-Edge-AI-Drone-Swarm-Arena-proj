import { useState } from 'react';
import { useNexusStore, TeamName } from '../store/nexus';
import { clsx } from 'clsx';
import { Zap, Radio, Cpu, Battery, Crosshair, Terminal, Activity } from 'lucide-react';

const TEAM_COLORS: Record<TeamName, string> = {
  RED: '#ff4060', BLUE: '#4080ff', GREEN: '#00ff9d', GOLD: '#ffb800',
};

// ─── Score board ──────────────────────────────────────────────────────────────

export function ScoreBoard() {
  const { scores, drones, tick, elapsed, swarmStatus } = useNexusStore();
  const teams = Object.entries(scores) as [TeamName, number][];
  const sorted = [...teams].sort(([, a], [, b]) => b - a);

  return (
    <div className="hud-panel relative p-3 space-y-2 min-w-48">
      <div className="flex items-center gap-2 mb-3">
        <Zap size={14} className="text-cyan-400" />
        <span className="text-xs font-mono text-cyan-400 tracking-widest">NEXUSFORGE</span>
        <span className="ml-auto text-xs font-mono text-gray-500">T:{tick}</span>
      </div>

      {sorted.map(([team, score]) => {
        const alive = drones.filter(d => d.team === team && d.state !== 'dead').length;
        const total = drones.filter(d => d.team === team).length;
        const color = TEAM_COLORS[team] || '#aaa';
        const status = swarmStatus?.teams?.[team];
        return (
          <div key={team} className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: color, boxShadow: `0 0 6px ${color}` }} />
            <span className="text-xs font-semibold w-10" style={{ color }}>{team}</span>
            <div className="flex-1 h-1.5 bg-white/5 rounded overflow-hidden">
              <div className="h-full rounded transition-all" style={{ width: `${(alive / Math.max(total, 1)) * 100}%`, background: color }} />
            </div>
            <span className="text-xs font-mono text-gray-400 w-8 text-right">{alive}/{total}</span>
            <span className="text-xs font-mono font-bold w-10 text-right" style={{ color }}>{score}</span>
            {status && (
              <span className="text-xs text-gray-600 font-mono hidden xl:block w-20 truncate">{status.mission}</span>
            )}
          </div>
        );
      })}

      <div className="pt-1 border-t border-cyan-900/30 text-xs font-mono text-gray-600 flex justify-between">
        <span>{Math.floor(elapsed / 60)}:{String(Math.floor(elapsed % 60)).padStart(2, '0')}</span>
        <span>{drones.filter(d => d.state !== 'dead').length} alive</span>
      </div>
    </div>
  );
}


// ─── Drone inspector ─────────────────────────────────────────────────────────

export function DroneInspector() {
  const { drones, selectedDroneId, hilTelemetry } = useNexusStore();
  const drone = drones.find(d => d.id === selectedDroneId);

  if (!drone) {
    return (
      <div className="hud-panel relative p-3 text-xs font-mono text-gray-600 min-w-56">
        <div className="flex items-center gap-2 mb-2">
          <Crosshair size={12} className="text-cyan-800" />
          <span className="text-cyan-800">DRONE INSPECTOR</span>
        </div>
        Click a drone to inspect
      </div>
    );
  }

  const color = TEAM_COLORS[drone.team] || '#aaa';
  const hilData = hilTelemetry.find(t => t.drone_id === drone.id);

  const rows: [string, string][] = [
    ['ID',       drone.id],
    ['TEAM',     drone.team],
    ['STATE',    drone.state.toUpperCase()],
    ['HP',       `${drone.health.toFixed(0)} / 100`],
    ['SHIELD',   `${drone.shield.toFixed(0)} / 50`],
    ['BATTERY',  `${drone.battery_pct.toFixed(1)}%`],
    ['KILLS',    String(drone.kills)],
    ['HITS',     String(drone.hits_taken)],
    ['LATENCY',  `${drone.latency_ms.toFixed(1)} ms`],
    ['INFER',    `${drone.inference_ms.toFixed(2)} ms`],
    ['VEL',      `${Math.sqrt(drone.velocity.x**2 + drone.velocity.y**2).toFixed(1)} u/s`],
    ...(hilData ? [
      ['RSSI',   `${hilData.radio?.rssi_dbm ?? '?'} dBm`] as [string, string],
      ['PKT LOSS', `${hilData.radio?.packet_loss_pct?.toFixed(2) ?? '?'}%`] as [string, string],
      ['CPU',    `${hilData.compute?.cpu_load_pct?.toFixed(1) ?? '?'}%`] as [string, string],
    ] : []),
  ];

  return (
    <div className="hud-panel relative p-3 min-w-56">
      <div className="flex items-center gap-2 mb-3">
        <Crosshair size={12} style={{ color }} />
        <span className="text-xs font-mono tracking-wider" style={{ color }}>DRONE #{drone.id.slice(-4)}</span>
      </div>

      {/* HP bar */}
      <div className="mb-3">
        <div className="flex justify-between text-xs font-mono mb-1">
          <span className="text-gray-500">HULL</span>
          <span style={{ color: drone.health > 50 ? '#00ff9d' : drone.health > 25 ? '#ffb800' : '#ff4060' }}>
            {drone.health.toFixed(0)}%
          </span>
        </div>
        <div className="h-2 bg-white/5 rounded overflow-hidden">
          <div
            className="h-full rounded transition-all duration-300"
            style={{
              width: `${drone.health}%`,
              background: drone.health > 50 ? '#00ff9d' : drone.health > 25 ? '#ffb800' : '#ff4060',
            }}
          />
        </div>
        <div className="h-1.5 bg-white/5 rounded overflow-hidden mt-1">
          <div className="h-full rounded bg-cyan-500/70" style={{ width: `${(drone.shield / 50) * 100}%` }} />
        </div>
      </div>

      <div className="space-y-0.5">
        {rows.map(([label, val]) => (
          <div key={label} className="flex justify-between text-xs font-mono">
            <span className="text-gray-600">{label}</span>
            <span className="text-cyan-300">{val}</span>
          </div>
        ))}
      </div>
    </div>
  );
}


// ─── Kill feed ───────────────────────────────────────────────────────────────

export function KillFeed() {
  const { events } = useNexusStore();
  const kills = events.filter(e => e.type === 'kill').slice(-8).reverse();

  return (
    <div className="hud-panel relative p-3 min-w-52">
      <div className="flex items-center gap-2 mb-2">
        <Activity size={12} className="text-red-400" />
        <span className="text-xs font-mono text-red-400 tracking-widest">KILL FEED</span>
      </div>
      {kills.length === 0
        ? <p className="text-xs font-mono text-gray-700">No kills yet</p>
        : kills.map((ev, i) => (
          <div key={i} className="slide-in flex items-center gap-1.5 text-xs font-mono py-0.5">
            <span className="text-cyan-400">#{(ev.killer || '?').slice(-4)}</span>
            <span className="text-gray-600">→</span>
            <span style={{ color: TEAM_COLORS[ev.victim_team as TeamName] || '#aaa' }}>
              #{(ev.victim || '?').slice(-4)}
            </span>
            <span className="text-gray-700 ml-auto">T:{ev.tick}</span>
          </div>
        ))
      }
    </div>
  );
}


// ─── Command terminal ────────────────────────────────────────────────────────

const QUICK_COMMANDS = [
  "Attack the center in wedge formation",
  "Defend the nexus with circle formation",
  "Flank the blue team from the east",
  "Regroup at alpha point",
  "Scatter and capture all control points",
  "Surround the enemy formation",
];

export function CommandTerminal() {
  const { commandText, commandTeam, connected, setCommandText, setCommandTeam, sendCommand } = useNexusStore();
  const [history, setHistory] = useState<string[]>([]);

  const submit = () => {
    if (!commandText.trim()) return;
    sendCommand(commandText);
    setHistory(h => [`[${commandTeam}] ${commandText}`, ...h.slice(0, 7)]);
    setCommandText('');
  };

  return (
    <div className="hud-panel relative p-3 min-w-72">
      <div className="flex items-center gap-2 mb-3">
        <Terminal size={12} className="text-cyan-400" />
        <span className="text-xs font-mono text-cyan-400 tracking-widest">NLP COMMAND</span>
        <div className={clsx('ml-auto w-1.5 h-1.5 rounded-full', connected ? 'bg-green-400 pulse' : 'bg-red-500')} />
      </div>

      {/* Team selector */}
      <div className="flex gap-1.5 mb-2">
        {(['RED','BLUE','GREEN','GOLD'] as TeamName[]).map(t => (
          <button
            key={t}
            onClick={() => setCommandTeam(t)}
            className="text-xs font-mono px-2 py-1 rounded border transition-all"
            style={{
              color: TEAM_COLORS[t],
              borderColor: commandTeam === t ? TEAM_COLORS[t] : 'rgba(255,255,255,0.05)',
              background: commandTeam === t ? `${TEAM_COLORS[t]}20` : 'transparent',
            }}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Input */}
      <div className="flex gap-1.5 mb-2">
        <input
          className="flex-1 bg-white/5 border border-cyan-900/40 rounded px-3 py-1.5 text-xs font-mono text-cyan-200 outline-none focus:border-cyan-500/60"
          placeholder="Issue natural language command..."
          value={commandText}
          onChange={e => setCommandText(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && submit()}
        />
        <button
          onClick={submit}
          disabled={!connected}
          className="px-3 py-1.5 bg-cyan-600/20 hover:bg-cyan-600/40 border border-cyan-600/30 rounded text-xs font-mono text-cyan-300 transition-all disabled:opacity-30"
        >
          EXEC
        </button>
      </div>

      {/* Quick commands */}
      <div className="flex flex-wrap gap-1 mb-2">
        {QUICK_COMMANDS.slice(0, 3).map(cmd => (
          <button
            key={cmd}
            onClick={() => { setCommandText(cmd); }}
            className="text-xs font-mono px-1.5 py-0.5 bg-white/3 hover:bg-white/8 border border-white/5 rounded text-gray-500 hover:text-gray-300 transition-all truncate max-w-[140px]"
          >
            {cmd.slice(0, 22)}…
          </button>
        ))}
      </div>

      {/* History */}
      <div className="space-y-0.5 max-h-16 overflow-y-auto">
        {history.map((h, i) => (
          <div key={i} className="text-xs font-mono text-gray-600 truncate">» {h}</div>
        ))}
      </div>
    </div>
  );
}


// ─── HIL Fleet health ─────────────────────────────────────────────────────────

export function HILPanel() {
  const { hilFleet, drones } = useNexusStore();

  const avgInfer = drones.length
    ? (drones.reduce((s, d) => s + d.inference_ms, 0) / drones.length).toFixed(2)
    : '—';
  const avgLatency = drones.length
    ? (drones.reduce((s, d) => s + d.latency_ms, 0) / drones.length).toFixed(1)
    : '—';
  const avgBatt = drones.filter(d => d.state !== 'dead').length
    ? (drones.filter(d => d.state !== 'dead').reduce((s, d) => s + d.battery_pct, 0) /
       drones.filter(d => d.state !== 'dead').length).toFixed(1)
    : '—';

  return (
    <div className="hud-panel relative p-3 min-w-52">
      <div className="flex items-center gap-2 mb-3">
        <Radio size={12} className="text-purple-400" />
        <span className="text-xs font-mono text-purple-400 tracking-widest">HIL TELEMETRY</span>
      </div>

      <div className="space-y-2">
        <Metric icon={<Radio size={10} />} label="AVG LATENCY" value={`${avgLatency} ms`} color="#7c3aed" />
        <Metric icon={<Cpu size={10} />} label="AVG INFERENCE" value={`${avgInfer} ms`} color="#00d2ff" />
        <Metric icon={<Battery size={10} />} label="AVG BATTERY" value={`${avgBatt}%`} color="#00ff9d" />
        {hilFleet.command_delivery_rate && (
          <Metric icon={<Zap size={10} />} label="CMD DELIVERY" value={`${hilFleet.command_delivery_rate}%`} color="#ffb800" />
        )}
      </div>
    </div>
  );
}

function Metric({ icon, label, value, color }: { icon: React.ReactNode; label: string; value: string; color: string }) {
  return (
    <div className="flex items-center gap-2 text-xs font-mono">
      <span style={{ color }}>{icon}</span>
      <span className="text-gray-600 flex-1">{label}</span>
      <span style={{ color }}>{value}</span>
    </div>
  );
}


// ─── Leaderboard table ────────────────────────────────────────────────────────

export function LeaderboardPanel({ rows }: { rows: any[] }) {
  return (
    <div className="hud-panel relative p-3">
      <div className="text-xs font-mono text-cyan-400 tracking-widest mb-2">LEADERBOARD</div>
      <table className="w-full text-xs font-mono">
        <thead>
          <tr className="text-gray-600">
            <th className="text-left py-0.5">#</th>
            <th className="text-left">TEAM</th>
            <th className="text-right">K</th>
            <th className="text-right">D</th>
            <th className="text-right">DMG</th>
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 10).map((r, i) => {
            const color = TEAM_COLORS[r.team as TeamName] || '#aaa';
            return (
              <tr key={r.id} className={clsx(!r.alive && 'opacity-30')}>
                <td className="py-0.5 text-gray-600">{i + 1}</td>
                <td style={{ color }}>{r.team}</td>
                <td className="text-right text-green-400">{r.kills}</td>
                <td className="text-right text-red-400">{r.deaths}</td>
                <td className="text-right text-gray-400">{r.damage}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
