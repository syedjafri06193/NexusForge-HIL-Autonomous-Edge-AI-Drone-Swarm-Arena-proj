import { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, Legend } from 'recharts';
import { X, Cpu, Zap } from 'lucide-react';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const MODELS = ['obstacle_detector', 'threat_classifier', 'trajectory_predictor', 'swarm_coordinator'];
const MCUS   = ['esp32', 'esp32_s3', 'stm32f4', 'stm32h7', 'rpi_zero_2'];

export function BenchmarkPanel({ onClose }: { onClose: () => void }) {
  const [model, setModel] = useState('obstacle_detector');
  const [mcu, setMcu] = useState('esp32');
  const [bits, setBits] = useState(8);
  const [runs, setRuns] = useState(100);
  const [result, setResult] = useState<any>(null);
  const [compareData, setCompareData] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  const runBench = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API}/benchmark`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: model, mcu_type: mcu, bits, n_runs: runs }),
      });
      setResult(await res.json());
    } finally { setLoading(false); }
  };

  const runCompare = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API}/benchmark/compare?model_name=${model}&mcu_type=${mcu}`, {
        method: 'POST',
      });
      const data = await res.json();
      setCompareData(data.comparisons?.map((c: any) => ({
        bits: `${c.bits}-bit`,
        latency_p50: c.latency_ms?.p50,
        latency_p99: c.latency_ms?.p99,
        accuracy: (c.accuracy?.mean * 100).toFixed(1),
        energy_uj: c.energy_uj?.mean,
        budget_met: c.budget_met_pct,
        size_kb: c.model_size_kb,
      })) || []);
    } finally { setLoading(false); }
  };

  return (
    <div className="h-full overflow-auto p-4">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center gap-3 mb-6">
          <Cpu size={18} className="text-cyan-400" />
          <h2 className="text-base font-mono font-bold text-cyan-300 tracking-widest">EDGE AI BENCHMARK</h2>
          <button onClick={onClose} className="ml-auto text-gray-500 hover:text-white p-1">
            <X size={16} />
          </button>
        </div>

        {/* Controls */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
          <div>
            <label className="text-xs font-mono text-gray-500 block mb-1">MODEL</label>
            <select className="w-full bg-white/5 border border-white/10 rounded px-2 py-1.5 text-xs font-mono text-cyan-200 outline-none"
              value={model} onChange={e => setModel(e.target.value)}>
              {MODELS.map(m => <option key={m} value={m}>{m}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs font-mono text-gray-500 block mb-1">MCU</label>
            <select className="w-full bg-white/5 border border-white/10 rounded px-2 py-1.5 text-xs font-mono text-cyan-200 outline-none"
              value={mcu} onChange={e => setMcu(e.target.value)}>
              {MCUS.map(m => <option key={m} value={m}>{m}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs font-mono text-gray-500 block mb-1">QUANT BITS</label>
            <div className="flex gap-1">
              {[4, 8, 16, 32].map(b => (
                <button key={b} onClick={() => setBits(b)}
                  className={`flex-1 py-1.5 rounded text-xs font-mono border transition-all ${bits === b ? 'bg-cyan-600/20 border-cyan-500/50 text-cyan-300' : 'border-white/5 text-gray-600 hover:text-gray-400'}`}>
                  {b}
                </button>
              ))}
            </div>
          </div>
          <div>
            <label className="text-xs font-mono text-gray-500 block mb-1">RUNS: {runs}</label>
            <input type="range" min={10} max={500} step={10} value={runs}
              onChange={e => setRuns(Number(e.target.value))}
              className="w-full accent-cyan-500 mt-1" />
          </div>
        </div>

        <div className="flex gap-2 mb-6">
          <button onClick={runBench} disabled={loading}
            className="px-4 py-2 bg-cyan-600/15 hover:bg-cyan-600/30 border border-cyan-600/30 rounded text-xs font-mono text-cyan-300 transition-all disabled:opacity-30">
            {loading ? '⟳ RUNNING...' : '▶ RUN BENCHMARK'}
          </button>
          <button onClick={runCompare} disabled={loading}
            className="px-4 py-2 bg-purple-600/15 hover:bg-purple-600/30 border border-purple-600/30 rounded text-xs font-mono text-purple-300 transition-all disabled:opacity-30">
            ⧖ COMPARE QUANTIZATIONS
          </button>
        </div>

        {/* Single benchmark result */}
        {result && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
            {[
              { label: 'P50 LATENCY',  value: `${result.latency_ms?.p50} ms`,   color: '#00d2ff' },
              { label: 'P99 LATENCY',  value: `${result.latency_ms?.p99} ms`,   color: '#7c3aed' },
              { label: 'ACCURACY',     value: `${(result.accuracy?.mean * 100).toFixed(1)}%`, color: '#00ff9d' },
              { label: 'BUDGET MET',   value: `${result.budget_met_pct}%`,       color: '#ffb800' },
              { label: 'AVG ENERGY',   value: `${result.energy_uj?.mean} µJ`,    color: '#ff4060' },
              { label: 'MODEL SIZE',   value: `${result.model_size_kb} KB`,      color: '#00d2ff' },
              { label: 'TOTAL ENERGY', value: `${result.energy_uj?.total} µJ`,   color: '#7c3aed' },
              { label: 'MEMORY OK',    value: result.memory_ok ? 'YES' : 'NO ⚠', color: result.memory_ok ? '#00ff9d' : '#ff4060' },
            ].map(({ label, value, color }) => (
              <div key={label} className="hud-panel relative p-3 text-center">
                <div className="text-xs font-mono text-gray-600 mb-1">{label}</div>
                <div className="text-sm font-mono font-bold" style={{ color }}>{value}</div>
              </div>
            ))}
          </div>
        )}

        {/* Compare chart */}
        {compareData.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="hud-panel relative p-4">
              <div className="text-xs font-mono text-gray-500 mb-3">LATENCY vs QUANTIZATION</div>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={compareData} barSize={20}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="bits" tick={{ fill: '#5a7090', fontSize: 10 }} />
                  <YAxis tick={{ fill: '#5a7090', fontSize: 10 }} unit="ms" />
                  <Tooltip contentStyle={{ background: '#0d1117', border: '1px solid rgba(0,210,255,0.2)', borderRadius: 6 }} />
                  <Bar dataKey="latency_p50" name="P50" fill="#00d2ff" radius={[3,3,0,0]} />
                  <Bar dataKey="latency_p99" name="P99" fill="#7c3aed" radius={[3,3,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="hud-panel relative p-4">
              <div className="text-xs font-mono text-gray-500 mb-3">ACCURACY & BUDGET vs QUANTIZATION</div>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={compareData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="bits" tick={{ fill: '#5a7090', fontSize: 10 }} />
                  <YAxis tick={{ fill: '#5a7090', fontSize: 10 }} unit="%" />
                  <Tooltip contentStyle={{ background: '#0d1117', border: '1px solid rgba(0,210,255,0.2)', borderRadius: 6 }} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  <Line type="monotone" dataKey="accuracy" name="Accuracy%" stroke="#00ff9d" dot strokeWidth={2} />
                  <Line type="monotone" dataKey="budget_met" name="Budget Met%" stroke="#ffb800" dot strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
