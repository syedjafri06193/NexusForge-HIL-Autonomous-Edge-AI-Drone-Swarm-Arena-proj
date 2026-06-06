import { useState } from 'react';
import { useNexusStore } from './store/nexus';
import { useSimWebSocket } from './hooks/useSimWebSocket';
import { ArenaCanvas } from './components/ArenaCanvas';
import {
  ScoreBoard, DroneInspector, KillFeed, CommandTerminal, HILPanel, LeaderboardPanel,
} from './components/HUDPanels';
import { BenchmarkPanel } from './components/BenchmarkPanel';
import { clsx } from 'clsx';
import { Zap, Play, Square, Pause, BarChart2, Layers, Activity, Settings } from 'lucide-react';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function App() {
  const store = useNexusStore();
  const { sessionId, connected, showBenchmark, setShowBenchmark } = store;
  useSimWebSocket(sessionId);

  if (!sessionId) return <LobbyScreen />;

  return (
    <div className="flex flex-col h-screen bg-bg overflow-hidden scanline">
      <TopBar />

      <div className="flex flex-1 min-h-0 gap-1 p-1">
        {/* Left sidebar */}
        <div className="flex flex-col gap-1 w-52 flex-shrink-0">
          <ScoreBoard />
          <HILPanel />
          <KillFeed />
        </div>

        {/* Arena */}
        <div className="flex-1 min-w-0 relative">
          <ArenaCanvas />
          {showBenchmark && (
            <div className="absolute inset-0 bg-black/80 backdrop-blur-sm z-10 overflow-auto">
              <BenchmarkPanel onClose={() => setShowBenchmark(false)} />
            </div>
          )}
        </div>

        {/* Right sidebar */}
        <div className="flex flex-col gap-1 w-64 flex-shrink-0">
          <CommandTerminal />
          <DroneInspector />
          <LeaderboardPanel rows={store.drones
            .sort((a, b) => b.kills - a.kills)
            .map(d => ({ id: d.id, team: d.team, kills: d.kills, deaths: d.hits_taken, damage: 0, alive: d.state !== 'dead' }))
          } />
        </div>
      </div>
    </div>
  );
}

// ─── Top bar ──────────────────────────────────────────────────────────────────

function TopBar() {
  const { sessionId, connected, elapsed, tick, setShowBenchmark, showBenchmark, disconnect } = useNexusStore();

  const pauseSession = async () => {
    await fetch(`${API}/sessions/${sessionId}/pause`, { method: 'POST' });
  };

  return (
    <div className="h-9 flex items-center gap-3 px-3 border-b border-cyan-900/30 bg-black/40 flex-shrink-0">
      <div className="flex items-center gap-1.5">
        <Zap size={14} className="text-cyan-400" />
        <span className="text-xs font-mono font-bold text-cyan-300 tracking-widest">NEXUSFORGE</span>
      </div>

      <div className="text-xs font-mono text-gray-600">SID:{sessionId}</div>

      <div className={clsx('flex items-center gap-1 text-xs font-mono', connected ? 'text-green-400' : 'text-red-500')}>
        <div className={clsx('w-1.5 h-1.5 rounded-full', connected ? 'bg-green-400 pulse' : 'bg-red-500')} />
        {connected ? 'LIVE' : 'DISCONNECTED'}
      </div>

      <div className="text-xs font-mono text-gray-600">
        {Math.floor(elapsed / 60)}:{String(Math.floor(elapsed % 60)).padStart(2, '0')} | T:{tick}
      </div>

      <div className="ml-auto flex items-center gap-1.5">
        <TopBtn icon={<Pause size={12} />} label="PAUSE" onClick={pauseSession} />
        <TopBtn icon={<BarChart2 size={12} />} label="BENCH"
          onClick={() => setShowBenchmark(!showBenchmark)}
          active={showBenchmark}
        />
        <TopBtn icon={<Square size={12} />} label="END" onClick={disconnect} danger />
      </div>
    </div>
  );
}

function TopBtn({ icon, label, onClick, active, danger }: any) {
  return (
    <button
      onClick={onClick}
      className={clsx(
        'flex items-center gap-1 px-2 py-1 rounded text-xs font-mono transition-all border',
        active ? 'bg-cyan-600/30 border-cyan-500/50 text-cyan-300'
        : danger ? 'border-red-800/30 text-red-500 hover:bg-red-900/20'
        : 'border-white/5 text-gray-500 hover:text-gray-300 hover:bg-white/5'
      )}
    >
      {icon}{label}
    </button>
  );
}

// ─── Lobby ────────────────────────────────────────────────────────────────────

function LobbyScreen() {
  const [teams, setTeams] = useState(2);
  const [perTeam, setPerTeam] = useState(8);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const { setSession } = useNexusStore();

  const launch = async () => {
    setLoading(true); setError('');
    try {
      const res = await fetch(`${API}/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ num_teams: teams, drones_per_team: perTeam }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setSession(data.session_id);
    } catch (e: any) {
      setError(e.message || 'Failed to create session');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-screen flex items-center justify-center bg-bg overflow-hidden">
      {/* Animated background */}
      <div className="absolute inset-0 scanline opacity-40" />
      <div className="absolute inset-0" style={{
        background: 'radial-gradient(ellipse at 30% 50%, rgba(0,210,255,0.04) 0%, transparent 60%), radial-gradient(ellipse at 70% 50%, rgba(124,58,237,0.04) 0%, transparent 60%)',
      }} />

      <div className="relative hud-panel p-10 w-full max-w-md text-center">
        <div className="mb-8">
          <div className="flex items-center justify-center gap-3 mb-2">
            <Zap size={32} className="text-cyan-400" style={{ filter: 'drop-shadow(0 0 12px #00d2ff)' }} />
          </div>
          <h1 className="text-4xl font-black tracking-wider" style={{
            fontFamily: 'Exo 2',
            background: 'linear-gradient(135deg, #00d2ff, #7c3aed)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}>
            NEXUSFORGE
          </h1>
          <p className="text-xs font-mono text-gray-500 mt-2 tracking-widest">
            HIL · AUTONOMOUS · EDGE-AI · DRONE SWARM ARENA
          </p>
        </div>

        <div className="space-y-4 mb-8">
          <div className="text-left">
            <label className="text-xs font-mono text-gray-500 block mb-1.5">TEAMS</label>
            <div className="flex gap-2">
              {[2, 3, 4].map(n => (
                <button key={n} onClick={() => setTeams(n)}
                  className={clsx(
                    'flex-1 py-2 rounded border text-xs font-mono font-bold transition-all',
                    teams === n
                      ? 'bg-cyan-600/20 border-cyan-500/50 text-cyan-300'
                      : 'border-white/5 text-gray-600 hover:text-gray-400 hover:border-white/10'
                  )}>
                  {n}
                </button>
              ))}
            </div>
          </div>

          <div className="text-left">
            <label className="text-xs font-mono text-gray-500 block mb-1.5">
              DRONES PER TEAM: <span className="text-cyan-400">{perTeam}</span>
            </label>
            <input type="range" min={1} max={32} value={perTeam}
              onChange={e => setPerTeam(Number(e.target.value))}
              className="w-full accent-cyan-500" />
            <div className="flex justify-between text-xs font-mono text-gray-700 mt-1">
              <span>1</span><span>8</span><span>16</span><span>32</span>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-2 text-xs font-mono text-gray-600">
            <div className="bg-white/3 rounded p-2 text-center">
              <div className="text-cyan-400 font-bold text-base">{teams * perTeam}</div>
              <div>DRONES</div>
            </div>
            <div className="bg-white/3 rounded p-2 text-center">
              <div className="text-purple-400 font-bold text-base">60</div>
              <div>FPS</div>
            </div>
            <div className="bg-white/3 rounded p-2 text-center">
              <div className="text-green-400 font-bold text-base">&lt;30</div>
              <div>MS EDGE</div>
            </div>
          </div>
        </div>

        {error && (
          <div className="mb-4 text-xs font-mono text-red-400 bg-red-900/10 border border-red-800/30 rounded px-3 py-2">
            ⚠ {error}
          </div>
        )}

        <button
          onClick={launch}
          disabled={loading}
          className="w-full py-3.5 rounded border border-cyan-500/50 bg-cyan-600/15 hover:bg-cyan-600/25 text-cyan-300 font-mono font-bold text-sm tracking-widest transition-all disabled:opacity-40"
          style={{ boxShadow: loading ? 'none' : '0 0 20px rgba(0,210,255,0.15)' }}
        >
          {loading ? '⟳ INITIALIZING...' : '▶ LAUNCH ARENA'}
        </button>

        <div className="mt-6 grid grid-cols-2 gap-2 text-xs font-mono text-gray-700">
          {[
            ['Behavior Trees', 'Multi-agent AI'],
            ['TinyML / ONNX', 'Edge inference'],
            ['HIL via MQTT', 'ESP32 / STM32'],
            ['NLP Commands', 'Swarm tactics'],
          ].map(([k, v]) => (
            <div key={k} className="flex items-center gap-1.5">
              <div className="w-1 h-1 rounded-full bg-cyan-800" />
              <span className="text-gray-500">{k}</span>
              <span className="text-gray-700 ml-auto">{v}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
