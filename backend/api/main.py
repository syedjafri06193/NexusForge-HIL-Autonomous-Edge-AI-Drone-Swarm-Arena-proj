"""
NexusForge Backend API
FastAPI server with WebSocket broadcasting, REST control endpoints,
telemetry ingestion, replay system, and analytics.
"""

import asyncio
import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Set

import redis.asyncio as aioredis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation.engine.sim import (
    Simulation, TeamID, DroneConfig, DroneState, Vec2,
    run_simulation_loop, DT,
)
from agents.swarm.orchestrator import SwarmOrchestrator, SwarmCommand, MissionType, Formation
from firmware.protocols.hil_mqtt import HILManager, HILConfig
from firmware.fault_injection.injector import FaultInjector, FaultScenarios, FaultEvent, FaultType
from backend.telemetry.writer import TelemetryWriter
from firmware.tinyml.inference import DroneInferencePool


# ─── Global state ─────────────────────────────────────────────────────────────

sessions: Dict[str, "GameSession"] = {}
redis_client: Optional[aioredis.Redis] = None

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")


# ─── Session ──────────────────────────────────────────────────────────────────

class GameSession:
    def __init__(self, session_id: str, num_teams: int = 2, drones_per_team: int = 8):
        self.session_id = session_id
        self.sim = Simulation(num_teams=num_teams, drones_per_team=drones_per_team)
        self.orchestrator = SwarmOrchestrator(self.sim)
        self.hil = HILManager(session_id)
        self.inference_pool = DroneInferencePool()
        self.websockets: Set[WebSocket] = set()
        self._sim_task: Optional[asyncio.Task] = None
        self.fault_injector = FaultInjector()
        self.telemetry_writer = TelemetryWriter()
        self.created_at = time.time()

        # Register HIL nodes for all drones
        for drone_id in self.sim.drones:
            self.hil.register_drone(drone_id)

    async def start(self):
        await self.telemetry_writer.start()
        self.sim.running = True
        self._sim_task = asyncio.create_task(
            run_simulation_loop(self.sim, self._broadcast)
        )

    async def stop(self):
        self.sim.running = False
        await self.telemetry_writer.stop()
        if self._sim_task:
            self._sim_task.cancel()

    async def _broadcast(self, state: dict):
        """Broadcast simulation state to all connected WebSocket clients."""
        # Inject HIL telemetry into state
        state["hil_telemetry"] = self.hil.collect_telemetry()
        state["hil_fleet"] = self.hil.get_fleet_health()
        state["swarm_status"] = self.orchestrator.get_status()

        # Update HIL from sim
        self.hil.update_from_sim(state["drones"])

        # Apply fault injections
        self.fault_injector.apply_to_simulation(self.sim)

        # Record telemetry
        self.telemetry_writer.record_session_snapshot(self.session_id, state)

        # Run swarm AI
        self.orchestrator.update()

        if not self.websockets:
            return

        msg = json.dumps(state)
        dead = set()
        for ws in self.websockets:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        self.websockets -= dead

        # Publish to Redis for multi-server fan-out
        if redis_client:
            try:
                await redis_client.publish(f"nexusforge:{self.session_id}", msg)
            except Exception:
                pass


# ─── App lifespan ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client
    try:
        redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
        await redis_client.ping()
    except Exception:
        redis_client = None
    yield
    for session in sessions.values():
        await session.stop()
    if redis_client:
        await redis_client.close()


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NexusForge API",
    version="1.0.0",
    description="HIL Autonomous Edge-AI Drone Swarm Arena",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request models ───────────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    num_teams: int = Field(2, ge=2, le=4)
    drones_per_team: int = Field(8, ge=1, le=32)

class SwarmCommandRequest(BaseModel):
    team: str
    mission: str
    formation: Optional[str] = None
    target_x: Optional[float] = None
    target_y: Optional[float] = None
    source: str = "operator"

class NLPCommandRequest(BaseModel):
    text: str
    team: str = "RED"

class SpawnDroneRequest(BaseModel):
    team: str
    model_type: str = "behavior_tree"
    aggression: float = 0.5
    quantization_bits: int = 8
    mcu_type: str = "esp32"

class BenchmarkRequest(BaseModel):
    model_name: str = "obstacle_detector"
    mcu_type: str = "esp32"
    bits: int = 8
    n_runs: int = 100


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "sessions": len(sessions),
        "redis": redis_client is not None,
    }


# ─── Session management ───────────────────────────────────────────────────────

@app.post("/sessions", status_code=201)
async def create_session(req: CreateSessionRequest):
    sid = str(uuid.uuid4())[:8]
    session = GameSession(sid, req.num_teams, req.drones_per_team)
    sessions[sid] = session
    await session.start()
    return {
        "session_id": sid,
        "drones": len(session.sim.drones),
        "teams": req.num_teams,
        "arena": session.sim.arena.to_dict(),
    }

@app.get("/sessions")
async def list_sessions():
    return [
        {
            "session_id": sid,
            "tick": s.sim.tick,
            "elapsed": round(s.sim.elapsed, 1),
            "drones": len(s.sim.drones),
            "scores": s.sim.scores,
            "created_at": s.created_at,
        }
        for sid, s in sessions.items()
    ]

@app.get("/sessions/{sid}")
async def get_session(sid: str):
    s = _get_session(sid)
    return {
        **s.sim.get_state_snapshot(),
        "leaderboard": s.sim.get_leaderboard(),
        "swarm": s.orchestrator.get_status(),
        "hil": s.hil.get_fleet_health(),
    }

@app.delete("/sessions/{sid}")
async def delete_session(sid: str):
    s = _get_session(sid)
    await s.stop()
    del sessions[sid]
    return {"deleted": sid}

@app.post("/sessions/{sid}/pause")
async def pause_session(sid: str):
    s = _get_session(sid)
    s.sim.paused = not s.sim.paused
    return {"paused": s.sim.paused}


# ─── Swarm control ────────────────────────────────────────────────────────────

@app.post("/sessions/{sid}/command")
async def issue_command(sid: str, req: SwarmCommandRequest):
    s = _get_session(sid)
    try:
        team = TeamID[req.team.upper()]
        mission = MissionType(req.mission.lower())
        formation = Formation(req.formation.lower()) if req.formation else None
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))

    target = Vec2(req.target_x, req.target_y) if req.target_x is not None else None
    cmd = SwarmCommand(
        team=team, mission=mission, formation=formation,
        target_position=target, source=req.source,
    )
    s.orchestrator.issue_command(cmd)
    return {"issued": cmd.to_dict()}

@app.post("/sessions/{sid}/nlp")
async def nlp_command(sid: str, req: NLPCommandRequest):
    s = _get_session(sid)
    try:
        team = TeamID[req.team.upper()]
    except KeyError:
        raise HTTPException(400, f"Unknown team: {req.team}")
    cmd = s.orchestrator.issue_nlp_command(req.text, team)
    if not cmd:
        raise HTTPException(400, "Could not parse command")
    return {"parsed": cmd.to_dict()}

@app.post("/sessions/{sid}/auto_tactics")
async def toggle_auto_tactics(sid: str, enabled: bool = True):
    s = _get_session(sid)
    s.orchestrator._auto_tactics = enabled
    return {"auto_tactics": enabled}


# ─── Drone management ─────────────────────────────────────────────────────────

@app.post("/sessions/{sid}/drones")
async def spawn_drone(sid: str, req: SpawnDroneRequest):
    s = _get_session(sid)
    if len(s.sim.drones) >= 128:
        raise HTTPException(400, "Max 128 drones per session")
    try:
        team = TeamID[req.team.upper()]
    except KeyError:
        raise HTTPException(400, f"Unknown team: {req.team}")
    config = DroneConfig(
        model_type=req.model_type,
        aggression=req.aggression,
        quantization_bits=req.quantization_bits,
    )
    drone = s.sim.spawn_drone(team, config)
    s.hil.register_drone(drone.id)
    return {"drone": drone.to_dict()}

@app.get("/sessions/{sid}/drones/{drone_id}")
async def get_drone(sid: str, drone_id: str):
    s = _get_session(sid)
    drone = s.sim.drones.get(drone_id)
    if not drone:
        raise HTTPException(404, "Drone not found")
    return {
        "drone": drone.to_dict(),
        "telemetry": drone.get_telemetry().__dict__,
    }

@app.get("/sessions/{sid}/leaderboard")
async def leaderboard(sid: str):
    s = _get_session(sid)
    return {"leaderboard": s.sim.get_leaderboard(), "scores": s.sim.scores}


# ─── Telemetry & HIL ──────────────────────────────────────────────────────────

@app.get("/sessions/{sid}/telemetry")
async def get_telemetry(sid: str, n: int = Query(50, le=500)):
    s = _get_session(sid)
    return {
        "recent": s.hil.get_recent_telemetry(n),
        "fleet_health": s.hil.get_fleet_health(),
    }

@app.get("/sessions/{sid}/hil/commands")
async def get_hil_commands(sid: str, n: int = Query(20, le=200)):
    s = _get_session(sid)
    return {"commands": s.hil.get_recent_commands(n)}

@app.post("/sessions/{sid}/hil/inject")
async def inject_hil_telemetry(sid: str, payload: dict):
    """
    Accept real hardware telemetry from physical ESP32 boards.
    Merges real data with simulation drone state.
    """
    s = _get_session(sid)
    drone_id = payload.get("drone_id")
    if drone_id and drone_id in s.sim.drones:
        drone = s.sim.drones[drone_id]
        # Override sim state with real hardware values
        if "health" in payload:
            drone.health = float(payload["health"])
        if "pos" in payload:
            drone.position = Vec2(payload["pos"]["x"], payload["pos"]["y"])
        if "battery_pct" in payload:
            drone.battery_pct = float(payload["battery_pct"])
    return {"injected": True, "drone_id": drone_id}


# ─── Edge AI benchmarks ───────────────────────────────────────────────────────

@app.post("/benchmark")
async def run_benchmark(req: BenchmarkRequest):
    from firmware.tinyml.inference import EdgeInferenceEngine
    engine = EdgeInferenceEngine(req.mcu_type, req.bits, budget_ms=20.0)
    result = engine.benchmark(req.model_name, req.n_runs)
    return result

@app.post("/benchmark/compare")
async def compare_quantizations(model_name: str = "obstacle_detector", mcu_type: str = "esp32"):
    from firmware.tinyml.inference import EdgeInferenceEngine
    engine = EdgeInferenceEngine(mcu_type, 8)
    return {"comparisons": engine.compare_quantizations(model_name, n_runs=50)}

@app.get("/benchmark/models")
async def list_models():
    from firmware.tinyml.inference import MODELS, MCU_PROFILES
    return {
        "models": {
            name: {
                "params": m.param_count,
                "flops": m.flops_per_inference,
                "base_accuracy": m.base_accuracy,
                "size_8bit_kb": round(m.model_size_kb(8), 2),
                "size_4bit_kb": round(m.model_size_kb(4), 2),
            }
            for name, m in MODELS.items()
        },
        "mcus": list(MCU_PROFILES.keys()),
    }


# ─── Replay ───────────────────────────────────────────────────────────────────

@app.get("/sessions/{sid}/replay")
async def get_replay(sid: str, start: int = 0, end: int = -1):
    s = _get_session(sid)
    frames = s.sim.replay_frames
    if end == -1:
        end = len(frames)
    return {
        "frames": frames[start:end],
        "total_frames": len(frames),
        "fps": 10,
    }


# ─── Analytics ────────────────────────────────────────────────────────────────

@app.get("/sessions/{sid}/analytics")
async def session_analytics(sid: str):
    s = _get_session(sid)
    sim = s.sim
    drones = list(sim.drones.values())

    total_kills = sum(d.kills for d in drones)
    total_damage = sum(d.damage_dealt for d in drones)
    total_hits = sum(d.hits_taken for d in drones)

    by_team = {}
    for team in TeamID:
        team_drones = [d for d in drones if d.team == team]
        if not team_drones:
            continue
        by_team[team.name] = {
            "alive": sum(1 for d in team_drones if d.is_alive),
            "dead": sum(1 for d in team_drones if not d.is_alive),
            "kills": sum(d.kills for d in team_drones),
            "damage_dealt": round(sum(d.damage_dealt for d in team_drones), 1),
            "avg_health": round(sum(d.health for d in team_drones if d.is_alive) / max(1, sum(1 for d in team_drones if d.is_alive)), 1),
            "avg_battery": round(sum(d.battery_pct for d in team_drones) / len(team_drones), 1),
            "avg_latency_ms": round(sum(d.latency_ms for d in team_drones) / len(team_drones), 2),
            "avg_inference_ms": round(sum(d.inference_ms for d in team_drones) / len(team_drones), 3),
            "score": sim.scores.get(team.name, 0),
        }

    return {
        "tick": sim.tick,
        "elapsed_s": round(sim.elapsed, 2),
        "total_kills": total_kills,
        "total_damage": round(total_damage, 1),
        "total_hits": total_hits,
        "by_team": by_team,
        "hil": s.hil.get_fleet_health(),
        "inference": s.inference_pool.aggregate_stats(),
        "events": sim.events[-50:],
    }


# ─── WebSocket ────────────────────────────────────────────────────────────────

@app.websocket("/sessions/{sid}/ws")
async def websocket_endpoint(websocket: WebSocket, sid: str):
    s = sessions.get(sid)
    if not s:
        await websocket.close(code=4004)
        return

    await websocket.accept()
    s.websockets.add(websocket)

    try:
        # Send initial state
        await websocket.send_text(json.dumps({
            "type": "init",
            **s.sim.get_state_snapshot(),
            "swarm_status": s.orchestrator.get_status(),
        }))

        # Listen for client commands
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                msg = json.loads(data)

                if msg.get("type") == "command":
                    team = TeamID[msg.get("team", "RED").upper()]
                    s.orchestrator.issue_nlp_command(msg.get("text", ""), team)

                elif msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))

            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "ping"}))

    except WebSocketDisconnect:
        pass
    finally:
        s.websockets.discard(websocket)



# ─── Fault injection ──────────────────────────────────────────────────────────

class FaultRequest(BaseModel):
    fault_type: str
    drone_ids: Optional[List[str]] = None
    severity: float = Field(0.5, ge=0.0, le=1.0)
    duration_s: float = Field(10.0, ge=1.0, le=120.0)
    description: str = ""

class FaultScenarioRequest(BaseModel):
    scenario: str   # network_degradation | battery_crisis | inference_failure | cascade | rogue_unit

@app.post("/sessions/{sid}/faults")
async def inject_fault(sid: str, req: FaultRequest):
    s = _get_session(sid)
    try:
        fault_type = FaultType(req.fault_type)
    except ValueError:
        raise HTTPException(400, f"Unknown fault type: {req.fault_type}. "
                            f"Valid types: {[f.value for f in FaultType]}")
    target_ids = req.drone_ids or list(s.sim.drones.keys())
    fault = FaultEvent(
        fault_type=fault_type,
        target_drone_ids=target_ids,
        severity=req.severity,
        duration_s=req.duration_s,
        description=req.description or f"Manual {fault_type.value}",
    )
    s.fault_injector.inject(fault)
    return {"injected": fault.to_dict()}

@app.post("/sessions/{sid}/faults/scenario")
async def inject_scenario(sid: str, req: FaultScenarioRequest):
    s = _get_session(sid)
    drone_ids = list(s.sim.drones.keys())
    scenario_map = {
        "network_degradation": FaultScenarios.network_degradation,
        "battery_crisis":      FaultScenarios.battery_crisis,
        "inference_failure":   FaultScenarios.inference_failure,
        "cascade":             FaultScenarios.cascade_failure,
        "rogue_unit":          FaultScenarios.rogue_unit,
    }
    fn = scenario_map.get(req.scenario)
    if not fn:
        raise HTTPException(400, f"Unknown scenario. Valid: {list(scenario_map.keys())}")
    fn(s.fault_injector, drone_ids)
    return {"scenario": req.scenario, "status": s.fault_injector.get_status()}

@app.get("/sessions/{sid}/faults")
async def get_faults(sid: str):
    s = _get_session(sid)
    return s.fault_injector.get_status()

@app.delete("/sessions/{sid}/faults")
async def clear_faults(sid: str):
    s = _get_session(sid)
    s.fault_injector.clear_all()
    return {"cleared": True}

@app.get("/faults/types")
async def list_fault_types():
    return {"types": [f.value for f in FaultType]}


# ─── Advanced analytics ───────────────────────────────────────────────────────

@app.get("/sessions/{sid}/analytics/timeseries/{drone_id}")
async def drone_timeseries(sid: str, drone_id: str):
    s = _get_session(sid)
    data = await s.telemetry_writer.query_latency_timeseries(sid, drone_id)
    return {"drone_id": drone_id, "timeseries": data}

@app.get("/sessions/{sid}/analytics/power")
async def power_profile(sid: str):
    s = _get_session(sid)
    data = await s.telemetry_writer.query_power_profile(sid)
    return {"power_profile": data}

@app.get("/sessions/{sid}/analytics/db")
async def db_analytics(sid: str):
    s = _get_session(sid)
    data = await s.telemetry_writer.query_session_analytics(sid)
    return data

@app.get("/sessions/{sid}/analytics/cached")
async def cached_snapshot(sid: str):
    s = _get_session(sid)
    data = await s.telemetry_writer.get_cached_snapshot(sid)
    return {"cached": data}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_session(sid: str) -> GameSession:
    s = sessions.get(sid)
    if not s:
        raise HTTPException(404, f"Session '{sid}' not found")
    return s


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=1)
