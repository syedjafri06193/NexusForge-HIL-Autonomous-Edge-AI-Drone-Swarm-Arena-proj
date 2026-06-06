import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import { Activity, Zap, Radio, AlertTriangle, Play, RotateCcw } from 'lucide-react';
import { useNexusStore } from '../store/nexus';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const CHART_STYLE = {
  contentStyle: { background: '#0d1117', border: '1px solid rgba(0,210,255,0.2)', borderRadius: 6, fontSize: 11 },
  itemStyle: { color: '#00d2ff' },
};

const TEAM_COLORS: Record<string, string> = {
  RED: '#ff4060', BLUE: '#4080ff', GREEN: '#00ff9d', GOLD: '#ffb800',
};

const FAULT_SCENARIOS = [
  { id: 'network_degradation', label: '📡 Network Degradation', desc: 'WiFi congestion, packet loss, latency spikes' },
  { id: 'battery_crisis',      label: '🔋 Battery Crisis',      desc: 'Accelerated discharge on subset of drones' },
  { id: 'inference_failure',   label: '🧠 Inference Failure',   desc: 'TinyML model corruption, deadline misses' },
  { id: 'cascade',             label: '⚡ Cascade Failure',     desc: 'Power + network + compute fault chain' },
  { id: 'rogue_unit',          label: '☠️ Rogue Unit',          desc: 'One drone goes rogue and attacks its team' },
];

const FAULT_TYPES = [
  'packet_loss_spike', 'latency_spike', 'battery_drain', 'brownout',
  'cpu_overload', 'inference_corrupt', 'sensor_freeze', 'split_brain', 'rogue_drone',
];

export default function AnalyticsPage({ onClose }: { onClose: () => void }) {
  const { sessionId, drones, scores } = useNexusStore();
  const [faultType, setFaultType] = useState('latency_spike');
  const [severity, setSeverity] = useState(0.5);
  const [duration, setDuration] = useState(10);
  const [activeTab, setActiveTab] = useState<'live' | 'faults' | 'rl'>('live');

  const { data: analytics } = useQuery({
    queryKey: ['analytics', sessionId],
    queryFn: () => fetch(`${API}/sessions/${sessionId}/analytics`).then(r => r.json()),
    enabled: !!sessionId,
    refetchInterval: 2000,
  });

  const { data: faultStatus, refetch: refetchFaults } = useQuery({
    queryKey: ['faults', sessionId],
    queryFn: () => fetch(`${API}/sessions/${sessionId}/faults`).then(r => r.json()),
    enabled: !!sessionId && activeTab === 'faults',
    refetchInterval: 1000,
  });

  const injectFault = useMutation({
    mutationFn: (req: any) => fetch(`${API}/sessions/${sessionId}/faults`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    }).then(r => r.json()),
    onSuccess: () => refetchFaults(),
  });

  const injectScenario = useMutation({
    mutationFn: (scenario: string) => fetch(`${API}/sessions/${sessionId}/faults/scenario`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scenario }),
    }).then(r => r.json()),
    onSuccess: () => refetchFaults(),
  });

  const clearFaults = useMutation({
    mutationFn: () => fetch(`${API}/sessions/${sessionId}/faults`, { method: 'DELETE' }).then(r => r.json()),
    onSuccess: () => refetchFaults(),
  });

  // Build per-team chart data
  const teamData = analytics?.by_team ? Object.entries(analytics.by_team).map(([team, stats]: any) => ({
    team,
    health: Math.round(stats.avg_health || 0),
    battery: Math.round(stats.avg_battery || 0),
    kills: stats.kills || 0,
    latency: Math.round(stats.avg_latency || 0),
    inference: +(stats.avg_inference || 0).toFixed(2),
    score: scores[team as any] || 0,
  })) : [];

  // Build per-drone latency chart data
  const droneLatencies = drones
    .filter(d => d.state !== 'dead')
    .slice(0, 16)
    .map(d => ({ id: d.id.slice(-4), latency: +d.latency_ms.toFixed(1), inference: +d.inference_ms.toFixed(2), battery: +d.battery_pct.toFixed(0) }));

  const radarData = teamData.map(t => ({
    team: t.team,
    Kills: t.kills,
    Health: t.health,
    Battery: t.battery,
    Score: Math.min(100, t.score),
  }));

  return (
    <div className="h-full overflow-auto p-4 font-mono">
      <div className="flex items-center gap-3 mb-5">
        <Activity size={18} className="text-cyan-400" />
        <h2 className="text-base font-bold text-cyan-300 tracking-widest">ANALYTICS & OBSERVABILITY</h2>
        <button onClick={onClose} className="ml-auto text-gray-500 hover:text-white text-xl leading-none">×</button>
      </div>

      {/* Tab bar */}
      <div className="flex gap-2 mb-5">
        {([['live', '📊 Live Stats'], ['faults', '⚡ Fault Injection'], ['rl', '🧠 RL Training']] as const).map(([v, l]) => (
          <button key={v} onClick={() => setActiveTab(v)}
            className={`px-4 py-1.5 rounded-lg text-xs font-semibold transition-all border ${activeTab === v ? 'bg-cyan-600/20 border-cyan-500/50 text-cyan-300' : 'border-white/5 text-gray-500 hover:text-gray-300'}`}>
            {l}
          </button>
        ))}
      </div>

      {/* ── Live stats ── */}
      {activeTab === 'live' && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {teamData.map(t => (
              <div key={t.team} className="hud-panel relative p-3 text-center">
                <div className="text-xs text-gray-600 mb-1">{t.team}</div>
                <div className="text-2xl font-bold mb-1" style={{ color: TEAM_COLORS[t.team] }}>{t.score}</div>
                <div className="text-xs text-gray-600">pts | {t.kills} kills</div>
              </div>
            ))}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="hud-panel relative p-4">
              <div className="text-xs text-gray-500 mb-3">HEALTH & BATTERY BY TEAM</div>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={teamData} barSize={18}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="team" tick={{ fill: '#5a7090', fontSize: 10 }} />
                  <YAxis tick={{ fill: '#5a7090', fontSize: 10 }} domain={[0, 100]} />
                  <Tooltip {...CHART_STYLE} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  <Bar dataKey="health" name="Health%" fill="#00ff9d" radius={[3,3,0,0]} />
                  <Bar dataKey="battery" name="Battery%" fill="#ffb800" radius={[3,3,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="hud-panel relative p-4">
              <div className="text-xs text-gray-500 mb-3">DRONE LATENCY (ms)</div>
              <ResponsiveContainer width="100%" height={180}>
                <AreaChart data={droneLatencies}>
                  <defs>
                    <linearGradient id="latGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#00d2ff" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#00d2ff" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="id" tick={{ fill: '#5a7090', fontSize: 9 }} />
                  <YAxis tick={{ fill: '#5a7090', fontSize: 10 }} />
                  <Tooltip {...CHART_STYLE} />
                  <Area type="monotone" dataKey="latency" name="Latency ms" stroke="#00d2ff" fill="url(#latGrad)" strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div className="hud-panel relative p-4">
              <div className="text-xs text-gray-500 mb-3">INFERENCE TIME (ms)</div>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={droneLatencies} barSize={14}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="id" tick={{ fill: '#5a7090', fontSize: 9 }} />
                  <YAxis tick={{ fill: '#5a7090', fontSize: 10 }} />
                  <Tooltip {...CHART_STYLE} />
                  <Bar dataKey="inference" name="Inference ms" fill="#7c3aed" radius={[3,3,0,0]} />
                  <Bar dataKey="latency" name="Latency ms" fill="#00d2ff" radius={[3,3,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="hud-panel relative p-4">
              <div className="text-xs text-gray-500 mb-3">TEAM RADAR</div>
              <ResponsiveContainer width="100%" height={180}>
                <RadarChart data={Object.keys(TEAM_COLORS).map(team => {
                  const td = teamData.find(t => t.team === team);
                  return { subject: team, Kills: td?.kills || 0, Health: td?.health || 0, Score: Math.min(100, td?.score || 0) };
                })}>
                  <PolarGrid stroke="rgba(255,255,255,0.08)" />
                  <PolarAngleAxis dataKey="subject" tick={{ fill: '#5a7090', fontSize: 10 }} />
                  <Radar name="Kills" dataKey="Kills" stroke="#ff4060" fill="#ff4060" fillOpacity={0.1} />
                  <Radar name="Health" dataKey="Health" stroke="#00ff9d" fill="#00ff9d" fillOpacity={0.1} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* ── Fault injection ── */}
      {activeTab === 'faults' && (
        <div className="space-y-4">
          {/* Scenario presets */}
          <div className="hud-panel relative p-4">
            <div className="flex items-center gap-2 mb-3">
              <AlertTriangle size={14} className="text-yellow-400" />
              <span className="text-xs text-yellow-400 tracking-widest">FAULT SCENARIOS</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
              {FAULT_SCENARIOS.map(s => (
                <button key={s.id}
                  onClick={() => injectScenario.mutate(s.id)}
                  disabled={!sessionId || injectScenario.isPending}
                  className="text-left p-3 bg-white/3 hover:bg-white/6 border border-white/5 hover:border-yellow-800/30 rounded-lg transition-all">
                  <div className="text-xs font-semibold text-yellow-400 mb-0.5">{s.label}</div>
                  <div className="text-xs text-gray-600">{s.desc}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Manual fault */}
          <div className="hud-panel relative p-4">
            <div className="text-xs text-gray-500 mb-3">MANUAL FAULT INJECTION</div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-3">
              <div>
                <label className="text-xs text-gray-600 block mb-1">FAULT TYPE</label>
                <select className="w-full bg-white/5 border border-white/10 rounded px-2 py-1.5 text-xs text-cyan-200 outline-none"
                  value={faultType} onChange={e => setFaultType(e.target.value)}>
                  {FAULT_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-600 block mb-1">SEVERITY: {severity.toFixed(1)}</label>
                <input type="range" min={0.1} max={1} step={0.1} value={severity}
                  onChange={e => setSeverity(Number(e.target.value))}
                  className="w-full accent-red-500 mt-1" />
              </div>
              <div>
                <label className="text-xs text-gray-600 block mb-1">DURATION: {duration}s</label>
                <input type="range" min={3} max={60} value={duration}
                  onChange={e => setDuration(Number(e.target.value))}
                  className="w-full accent-red-500 mt-1" />
              </div>
              <div className="flex items-end gap-2">
                <button onClick={() => injectFault.mutate({ fault_type: faultType, severity, duration_s: duration })}
                  disabled={!sessionId || injectFault.isPending}
                  className="flex-1 py-2 bg-red-900/20 hover:bg-red-900/40 border border-red-800/30 rounded text-xs text-red-400 font-semibold transition-all">
                  INJECT
                </button>
                <button onClick={() => clearFaults.mutate()}
                  disabled={!sessionId}
                  className="py-2 px-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded text-xs text-gray-500 transition-all">
                  <RotateCcw size={12} />
                </button>
              </div>
            </div>
          </div>

          {/* Active faults */}
          {faultStatus?.active?.length > 0 && (
            <div className="hud-panel relative p-4">
              <div className="text-xs text-red-400 tracking-widest mb-3">ACTIVE FAULTS ({faultStatus.active.length})</div>
              <div className="space-y-2">
                {faultStatus.active.map((f: any, i: number) => (
                  <div key={i} className="flex items-center gap-3 text-xs font-mono bg-red-900/10 border border-red-800/20 rounded p-2">
                    <span className="text-red-400 font-semibold">{f.type}</span>
                    <span className="text-gray-600">sev:{f.severity}</span>
                    <span className="text-gray-600">{f.targets?.length} drones</span>
                    <span className="ml-auto text-yellow-600">{f.remaining?.toFixed(1)}s remaining</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── RL Training ── */}
      {activeTab === 'rl' && (
        <div className="space-y-4">
          <div className="hud-panel relative p-4">
            <div className="text-xs text-gray-500 mb-3">SELF-PLAY RL TRAINING</div>
            <p className="text-xs text-gray-600 mb-4 leading-relaxed">
              Train a lightweight policy network (NumPy MLP, no GPU needed) via self-play.
              All teams share one policy — agents learn attack, evasion, flocking, and capture strategies.
              Run from CLI:
            </p>
            <div className="bg-black/40 border border-white/8 rounded px-4 py-3 font-mono text-xs text-green-400 mb-4">
              python -m agents.rl.trainer --episodes 200 --drones 4 --teams 2
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {[
                { label: 'Obs Dim',  value: '24',       desc: 'State features per drone' },
                { label: 'Act Dim',  value: '5',        desc: 'move_x, move_y, fire, evade, capture' },
                { label: 'Hidden',   value: '64',       desc: 'MLP hidden size' },
                { label: 'GAE λ',    value: '0.95',     desc: 'Advantage estimation' },
                { label: 'LR',       value: '3e-4',     desc: 'Adam learning rate' },
                { label: 'γ',        value: '0.99',     desc: 'Discount factor' },
                { label: 'Backend',  value: 'NumPy',    desc: 'No PyTorch/TF needed' },
                { label: 'Export',   value: 'JSON',     desc: 'Weights for frontend viz' },
              ].map(item => (
                <div key={item.label} className="bg-white/3 rounded p-3">
                  <div className="text-xs text-gray-600 mb-0.5">{item.label}</div>
                  <div className="text-sm font-bold text-cyan-300">{item.value}</div>
                  <div className="text-xs text-gray-700 mt-0.5">{item.desc}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="hud-panel relative p-4">
            <div className="text-xs text-gray-500 mb-3">OBSERVATION SPACE</div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {[
                ['[0,1]',   'Normalized position (x/W, y/H)'],
                ['[2,3]',   'Normalized velocity (vx/vmax, vy/vmax)'],
                ['[4]',     'Heading / π'],
                ['[5]',     'Health / 100'],
                ['[6]',     'Shield / 50'],
                ['[7]',     'Battery / 100'],
                ['[8]',     'Weapon ready (0/1)'],
                ['[9]',     'Stun timer (clamped)'],
                ['[10,11]', 'Nearest enemy relative position'],
                ['[12,13]', 'Enemy distance, health'],
                ['[14,15]', 'Nearest ally relative position'],
                ['[16,17]', 'Ally count, enemy count'],
                ['[18,19]', 'Nearest control point position'],
                ['[20]',    'Control point capture progress'],
                ['[21]',    'Is outnumbered (0/1)'],
                ['[22,23]', 'Team score, elapsed time'],
              ].map(([feat, desc]) => (
                <div key={feat} className="flex gap-2 text-xs font-mono">
                  <span className="text-cyan-600 w-14 flex-shrink-0">{feat}</span>
                  <span className="text-gray-600">{desc}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
