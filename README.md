# NexusForge
### HIL Autonomous Edge-AI Drone Swarm Arena

> A real-time multi-agent combat simulator bridging **edge AI**, **distributed systems**, and **hardware-in-the-loop** from ESP32 firmware to orchestrated drone swarms.

---

## What It Is

NexusForge is a persistent arena where **autonomous drones** compete, coordinate, and adapt in real time. Each drone runs lightweight **TinyML inference** on a simulated (or real) MCU, a **behavior tree AI** handles per-drone decisions, a **swarm orchestrator** manages team tactics, and an **NLP command interface** lets operators issue natural-language orders.

**Core portfolio story:** bridging edge AI, distributed systems, and real-time simulation from embedded hardware all the way up to orchestrated swarms.

---

## Architecture

```
ESP32 / STM32                    Browser Dashboard
  | MQTT 20Hz telemetry               |
  v                                   | WebSocket 60 FPS
Mosquitto ──────> FastAPI Backend ────+
                      |               |
                 Simulation Loop      NLP Command Input
                 Python 60 FPS        |
                 Behavior Trees   Swarm Orchestrator
                 TinyML Simulator  Tactical Planner
                      |            Formation Control
                   Redis
                   TimescaleDB
```

| Layer | Technology |
|-------|-----------|
| Simulation | Python + NumPy (custom physics, 60 FPS, 128 agents) |
| Per-drone AI | Behavior trees: Attack, Evade, Flock, Patrol, Capture |
| Swarm AI | Formation control, tactical planner, NLP intent parser |
| Edge AI | ONNX/TFLite inference sim (4/8/16/32-bit quantization) |
| Firmware | C/C++ + FreeRTOS (ESP32, STM32, PlatformIO) |
| Backend | FastAPI + WebSockets + Redis pub/sub |
| Time-series | TimescaleDB + continuous aggregates |
| Message bus | MQTT via Mosquitto |
| Dashboard | React + Canvas 2D renderer + Recharts benchmark view |
| Infra | Docker Compose, Kubernetes + HPA, Prometheus, Grafana |

---

## Quick Start

```bash
# Docker Compose (everything)
docker compose up -d

# Dashboard:  http://localhost:3000
# API Docs:   http://localhost:8000/docs
# MQTT:       localhost:1883

# Local dev
pip install -r backend/requirements.txt
uvicorn backend.api.main:app --reload --port 8000
cd dashboard && npm install && npm run dev
```

---

## Key Features

### Simulation (`simulation/engine/sim.py`)
- 128 drones at 60 FPS in Python asyncio
- Full 2D physics: velocity, drag, collision detection, wall bounce
- Weapons with projectile lead-targeting
- Shield regen, battery drain model
- Dynamic hazards: plasma storms, gravity wells, EMP pulses, shield disruptors
- 5 capturable control points with per-team progress

### Per-drone Behavior Tree AI (`agents/behaviors/behavior_tree.py`)
- Composable BT nodes: Sequence, Selector, Inverter, AlwaysSuccess
- Conditions: HasEnemiesInSight, IsLowHealth, IsOutnumbered, WeaponReady
- Actions: AttackNearestEnemy, Evade, Regroup, FlockWithAllies (Reynolds boids), Patrol, CaptureControlPoint, PinceMovement
- Three profiles: aggressive, defensive, flanker
- Quantization noise model: 4-bit drones occasionally make wrong decisions

### Swarm Orchestrator (`agents/swarm/orchestrator.py`)
- 10 mission types: Attack, Defend, Capture, Flank, Surround, Scatter, Regroup, Kamikaze...
- 6 formation shapes: Wedge, Line, Circle, Diamond, Column, Spread
- Hungarian-algorithm-style slot assignment (nearest drone to each formation slot)
- NLP parser: keyword intent extraction, team/location/formation detection
- Tactical planner: re-evaluates every 2 seconds, switches missions based on health/numbers/score

### Edge AI Simulator (`firmware/tinyml/inference.py`)
- 4 model specs: MobileNetV1, SqueezeNet-Lite, TinyLSTM, TinyTransformer
- 5 MCU profiles: ESP32, ESP32-S3, STM32F4, STM32H7, RPi Zero 2
- Realistic latency with jitter, cache misses, interrupt latency
- Power model: active mW * latency_ms = energy in µJ
- Accuracy degradation: 4-bit (-6%), 8-bit (-2%), 16-bit (-0.5%)
- Built-in benchmark: p50/p95/p99 latency, budget_met%, model size

### HIL / Firmware (`firmware/protocols/hil_mqtt.py`, `firmware/esp32_sim/main.cpp`)
- SimulatedESP32: generates realistic telemetry with sensor noise, clock drift, RSSI variance
- Packet loss model, latency jitter, battery voltage curve
- HILManager: fleet health aggregation, telemetry log, command delivery tracking
- Real ESP32 firmware (C++/PlatformIO): connects over WiFi+MQTT, FreeRTOS tasks, real inference stub

### Backend API (`backend/api/main.py`)
- FastAPI with WebSocket broadcasting at 60 FPS
- Session management: create/pause/delete
- REST endpoints: spawn drones, issue commands, get telemetry, run benchmarks
- HIL injection: `/hil/inject` merges real hardware data with simulation
- Redis pub/sub for multi-server WebSocket fan-out
- Replay system: records frames at 10 FPS, seekable

### Dashboard (`dashboard/`)
- Lobby: configure teams/drones, launch session
- 2D canvas arena: hexagonal drones, team colors, glow effects, health bars, projectiles, hazard overlays, control point capture arcs
- HUD panels: scoreboard, drone inspector (HP/shield/battery/latency/inference), kill feed, NLP terminal, HIL fleet health
- Benchmark explorer: latency charts, accuracy vs quantization, energy cost
- Click-to-select any drone for detailed telemetry

---

## NLP Command Examples

```
"Red team, attack the center in wedge formation"
"Defend the nexus with circle formation"
"Flank the blue team from the east"
"All units, regroup at alpha point"
"Scatter and capture all control points"
"Surround the enemy — prioritize nexus"
"Kamikaze run on the gold team"
```

---

## Edge AI Benchmark Results (simulated)

| Quantization | Latency P50 (ESP32) | Accuracy | Energy/inference | Budget Met |
|-------------|---------------------|----------|-----------------|------------|
| 32-bit | 85 ms | 87.0% | 20.4 µJ | 0% |
| 16-bit | 47 ms | 86.5% | 11.3 µJ | 36% |
| **8-bit** | **24 ms** | **85.0%** | **5.8 µJ** | **78%** |
| 4-bit | 14 ms | 81.0% | 3.4 µJ | 98% |

8-bit is the sweet spot: 3.5x speedup, only 2% accuracy drop, fits in 520KB ESP32 RAM.

---

## Resume-Ready Metrics

- 128 autonomous drones at 60 FPS in Python
- 4 quantization levels benchmarked across 5 MCU profiles
- Edge inference decisions under 30ms at 8-bit on ESP32
- Real ESP32 HIL integration via MQTT + FreeRTOS
- NLP intent parsing -> formation assignment -> multi-agent execution
- TimescaleDB hypertable ingesting 2000+ telemetry rows/sec
- Kubernetes deployment with HPA (2–10 API replicas)

---

## License

MIT
