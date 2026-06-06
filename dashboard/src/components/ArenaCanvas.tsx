import { useEffect, useRef, useCallback } from 'react';
import { useNexusStore, DroneState, TeamName } from '../store/nexus';

const TEAM_COLORS: Record<TeamName, string> = {
  RED:   '#ff4060',
  BLUE:  '#4080ff',
  GREEN: '#00ff9d',
  GOLD:  '#ffb800',
};

const TEAM_GLOW: Record<TeamName, string> = {
  RED:   'rgba(255,64,96,0.5)',
  BLUE:  'rgba(64,128,255,0.5)',
  GREEN: 'rgba(0,255,157,0.5)',
  GOLD:  'rgba(255,184,0,0.5)',
};

const HAZARD_COLORS: Record<string, string> = {
  plasma_storm:   'rgba(255,80,40,',
  gravity_well:   'rgba(140,40,255,',
  emp_pulse:      'rgba(0,255,220,',
  shield_disrupt: 'rgba(255,200,0,',
};

export function ArenaCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef   = useRef<number>();
  const store = useNexusStore();

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    const { arena, drones, projectiles, selectedDroneId } = store;

    if (!arena) {
      ctx.fillStyle = '#060810';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = 'rgba(0,210,255,0.3)';
      ctx.font = '18px Share Tech Mono';
      ctx.textAlign = 'center';
      ctx.fillText('Waiting for session...', canvas.width / 2, canvas.height / 2);
      return;
    }

    const scaleX = canvas.width  / arena.width;
    const scaleY = canvas.height / arena.height;
    const scale  = Math.min(scaleX, scaleY);
    const offX   = (canvas.width  - arena.width  * scale) / 2;
    const offY   = (canvas.height - arena.height * scale) / 2;

    const tx = (x: number) => x * scale + offX;
    const ty = (y: number) => y * scale + offY;
    const ts = (v: number) => v * scale;

    // ── Background ──────────────────────────────────────────────────────────
    ctx.fillStyle = '#06080e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Grid
    ctx.strokeStyle = 'rgba(0,210,255,0.04)';
    ctx.lineWidth = 1;
    const gridSize = 60 * scale;
    for (let x = offX % gridSize; x < canvas.width; x += gridSize) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
    }
    for (let y = offY % gridSize; y < canvas.height; y += gridSize) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
    }

    // Arena border glow
    ctx.strokeStyle = 'rgba(0,210,255,0.5)';
    ctx.lineWidth = 2;
    ctx.strokeRect(tx(0), ty(0), ts(arena.width), ts(arena.height));
    ctx.strokeStyle = 'rgba(0,210,255,0.1)';
    ctx.lineWidth = 8;
    ctx.strokeRect(tx(0), ty(0), ts(arena.width), ts(arena.height));

    // ── Hazards ─────────────────────────────────────────────────────────────
    for (const h of arena.hazards) {
      const cx = tx(h.x), cy = ty(h.y), r = ts(h.radius);
      const base = HAZARD_COLORS[h.type] || 'rgba(255,255,255,';
      const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
      const alpha = h.intensity * (0.3 + 0.1 * Math.sin(Date.now() / 500));
      grad.addColorStop(0,   base + (alpha * 0.8) + ')');
      grad.addColorStop(0.5, base + (alpha * 0.4) + ')');
      grad.addColorStop(1,   base + '0)');
      ctx.fillStyle = grad;
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill();

      // Pulsing ring
      ctx.strokeStyle = base + '0.6)';
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();
    }

    // ── Obstacles ──────────────────────────────────────────────────────────
    for (const obs of arena.obstacles) {
      ctx.fillStyle = 'rgba(0,30,50,0.95)';
      ctx.fillRect(tx(obs.x), ty(obs.y), ts(obs.w), ts(obs.h));
      ctx.strokeStyle = 'rgba(0,150,200,0.4)';
      ctx.lineWidth = 1;
      ctx.strokeRect(tx(obs.x), ty(obs.y), ts(obs.w), ts(obs.h));
    }

    // ── Control points ─────────────────────────────────────────────────────
    for (const cp of arena.control_points) {
      const cx = tx(cp.x), cy = ty(cp.y), r = ts(cp.r);
      const ownerColor = cp.owner ? TEAM_COLORS[cp.owner as TeamName] || '#888' : 'rgba(0,210,255,0.3)';
      const capture = cp.capture ?? 0;

      // Base ring
      ctx.strokeStyle = 'rgba(0,210,255,0.25)';
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();

      // Capture fill arc
      if (capture > 0) {
        ctx.strokeStyle = ownerColor;
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.arc(cx, cy, r - 4, -Math.PI / 2, -Math.PI / 2 + capture * Math.PI * 2);
        ctx.stroke();
      }

      // Label
      ctx.fillStyle = 'rgba(0,210,255,0.7)';
      ctx.font = `${Math.max(9, ts(10))}px Share Tech Mono`;
      ctx.textAlign = 'center';
      ctx.fillText(cp.id.toUpperCase(), cx, cy + ts(4));
    }

    // ── Projectiles ─────────────────────────────────────────────────────────
    for (const p of projectiles) {
      const color = TEAM_COLORS[p.team as TeamName] || '#fff';
      ctx.fillStyle = color;
      ctx.shadowBlur = 6;
      ctx.shadowColor = color;
      ctx.beginPath();
      ctx.arc(tx(p.x), ty(p.y), ts(3), 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.shadowBlur = 0;

    // ── Drones ──────────────────────────────────────────────────────────────
    for (const drone of drones) {
      const x = tx(drone.position.x);
      const y = ty(drone.position.y);
      const r = ts(12);
      const color = TEAM_COLORS[drone.team] || '#aaa';
      const glow  = TEAM_GLOW[drone.team]  || 'rgba(200,200,200,0.3)';
      const isDead = drone.state === 'dead';
      const isSelected = drone.id === selectedDroneId;
      const alpha = isDead ? 0.25 : 1.0;

      ctx.globalAlpha = alpha;

      // Selection ring
      if (isSelected) {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.arc(x, y, r + 8, 0, Math.PI * 2); ctx.stroke();
        ctx.setLineDash([]);
      }

      // Sensor range (for selected)
      if (isSelected) {
        ctx.strokeStyle = 'rgba(255,255,255,0.06)';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(x, y, ts(240), 0, Math.PI * 2); ctx.stroke();
      }

      // Glow
      if (!isDead) {
        ctx.shadowBlur  = isSelected ? 20 : 10;
        ctx.shadowColor = glow;
      }

      // Body hexagon
      ctx.fillStyle = isDead ? '#1a1a2e' : color + '33';
      ctx.strokeStyle = color;
      ctx.lineWidth = isDead ? 1 : 1.5;
      ctx.beginPath();
      for (let i = 0; i < 6; i++) {
        const angle = drone.heading + (i * Math.PI) / 3;
        const hx = x + Math.cos(angle) * r;
        const hy = y + Math.sin(angle) * r;
        i === 0 ? ctx.moveTo(hx, hy) : ctx.lineTo(hx, hy);
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // Direction arrow
      if (!isDead) {
        const arrowLen = r + 6;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(
          x + Math.cos(drone.heading) * arrowLen,
          y + Math.sin(drone.heading) * arrowLen,
        );
        ctx.stroke();
      }

      ctx.shadowBlur = 0;

      // Health/shield bars
      if (!isDead) {
        const barW = ts(22), barH = ts(3);
        const bx = x - barW / 2, by = y + r + ts(4);
        // Health
        ctx.fillStyle = 'rgba(0,0,0,0.6)';
        ctx.fillRect(bx, by, barW, barH);
        ctx.fillStyle = drone.health > 50 ? '#00ff9d' : drone.health > 25 ? '#ffb800' : '#ff4060';
        ctx.fillRect(bx, by, barW * (drone.health / 100), barH);
        // Shield
        ctx.fillStyle = 'rgba(0,0,0,0.6)';
        ctx.fillRect(bx, by + barH + 1, barW, barH - 1);
        ctx.fillStyle = 'rgba(100,200,255,0.8)';
        ctx.fillRect(bx, by + barH + 1, barW * (drone.shield / 50), barH - 1);
      }

      // Kill count badge
      if (drone.kills > 0 && !isDead) {
        ctx.fillStyle = color;
        ctx.font = `${ts(9)}px Share Tech Mono`;
        ctx.textAlign = 'center';
        ctx.fillText(`×${drone.kills}`, x, y - r - ts(3));
      }

      ctx.globalAlpha = 1.0;
    }

    ctx.textAlign = 'left';
  }, [store]);

  useEffect(() => {
    const loop = () => {
      draw();
      animRef.current = requestAnimationFrame(loop);
    };
    animRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animRef.current!);
  }, [draw]);

  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const { arena, drones } = store;
    if (!canvas || !arena) return;

    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;

    const scale = Math.min(canvas.width / arena.width, canvas.height / arena.height);
    const offX = (canvas.width - arena.width * scale) / 2;
    const offY = (canvas.height - arena.height * scale) / 2;

    const wx = (cx - offX) / scale;
    const wy = (cy - offY) / scale;

    const hit = drones.find(d => {
      const dx = d.position.x - wx, dy = d.position.y - wy;
      return Math.sqrt(dx * dx + dy * dy) < 16;
    });
    store.selectDrone(hit?.id ?? null);
  }, [store]);

  return (
    <canvas
      ref={canvasRef}
      width={900}
      height={675}
      onClick={handleClick}
      className="w-full h-full cursor-crosshair"
      style={{ display: 'block', background: '#06080e' }}
    />
  );
}
