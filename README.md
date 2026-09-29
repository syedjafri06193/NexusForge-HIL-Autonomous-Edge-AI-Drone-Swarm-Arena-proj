# NexusForge
### HIL Autonomous Edge-AI Drone Swarm Arena

A real-time multi-agent combat simulator bridging **edge AI**, **distributed
systems** and **hardware-in-the-loop** — from ESP32 firmware up to
orchestrated drone swarms.

Each drone runs simulated **TinyML inference** on an MCU profile, a
**behavior tree** makes per-drone decisions, a **swarm orchestrator** handles
team tactics, and a **natural-language command interface** lets operators
issue orders like *"Red team, attack the center in wedge formation."*

## Architecture

```
ESP32 / STM32                    Browser dashboard (React, 2D canvas + 3D view)
  │ MQTT 20 Hz telemetry               ▲
  ▼                                    │ WebSocket 60 FPS
Mosquitto ──────► FastAPI backend ─────┤
                      │                │
                 Simulation loop       NLP command input
                 (Python, 60 FPS)      │
                 Behavior trees    Swarm orchestrator
                 TinyML simulator  Tactical planner · formations
                 Fault injection   RL-trained policies
                      │
                 Redis · TimescaleDB
```

| Layer | Technology |
|---|---|
| Simulation | Python + NumPy (custom 2D physics, 60 FPS, up to 128 agents) |
| Per-drone AI | Behavior trees: attack, evade, flock, patrol, capture |
| Swarm AI | Formation control, tactical planner, NLP intent parser, PPO-style RL self-play |
| Edge AI | Inference simulator: 4/8/16/32-bit quantization across 5 MCU profiles |
| Firmware | C++ / FreeRTOS on ESP32 (PlatformIO), simulated ESP32 fleet, fault injection |
| Backend | FastAPI + WebSockets + Redis pub/sub |
| Storage | TimescaleDB hypertables |
| Message bus | MQTT (Mosquitto) |
| Dashboard | React, Canvas 2D, three.js 3D view, Recharts analytics |
| Infra | Docker Compose, Kubernetes + HPA |

## Quick start

**Headless, no services needed:**

```bash
pip install -r backend/requirements.txt
pytest                                                       # 111 tests
python demos/run_demo.py --teams 4 --drones 16 --ticks 600 --faults
```

**Full stack with Docker Compose:**

```bash
docker compose up -d
# Dashboard  http://localhost:3000
# API docs   http://localhost:8000/docs
# MQTT       localhost:1883
```

**Local dev:**

```bash
uvicorn backend.api.main:app --reload --port 8000
cd dashboard && npm install && npm run dev
```

Set `ANTHROPIC_API_KEY` to use Claude for richer command parsing; without it
the NLP interface falls back to the built-in keyword parser.

## Repository layout

```
.
├── simulation/engine/     sim.py — physics, weapons, hazards, control points
├── agents/
│   ├── behaviors/         behavior tree nodes and drone profiles
│   ├── swarm/             orchestrator: missions, formations, tactical planner
│   ├── nlp/               natural-language command parser (Claude + keyword fallback)
│   ├── rl/                self-play policy trainer
│   └── models/            trained policies (generated, git-ignored)
├── firmware/
│   ├── esp32_sim/         real ESP32 firmware (C++, PlatformIO)
│   ├── protocols/         simulated ESP32 fleet + HIL MQTT manager
│   ├── tinyml/            edge inference latency / power / accuracy model
│   └── fault_injection/   hardware and network fault scenarios
├── backend/
│   ├── api/               FastAPI app: sessions, commands, HIL inject, replay, analytics
│   └── telemetry/         TimescaleDB writer + Redis cache
├── dashboard/             React frontend
├── demos/                 headless demo script
├── infra/                 Dockerfiles, Mosquitto config, TimescaleDB init, k8s manifests
├── tests/                 agents, firmware, simulation
└── docker-compose.yml
```

## Features

### Simulation — `simulation/engine/sim.py`
- Up to 128 drones at 60 FPS in Python asyncio
- 2D physics: velocity, drag, collisions, wall bounce
- Weapons with projectile lead-targeting, shield regen, battery drain
- Dynamic hazards: plasma storms, gravity wells, EMP pulses, shield disruptors
- 5 capturable control points with per-team progress

### Per-drone behavior trees — `agents/behaviors/behavior_tree.py`
- Composable nodes: Sequence, Selector, Inverter, AlwaysSuccess
- Conditions: HasEnemiesInSight, IsLowHealth, IsOutnumbered, WeaponReady
- Actions: attack, evade, regroup, flock (Reynolds boids), patrol, capture, pincer
- Profiles: aggressive, defensive, flanker
- Quantization noise: 4-bit drones occasionally make wrong decisions

### Swarm orchestrator — `agents/swarm/orchestrator.py`
- 10 mission types (attack, defend, capture, flank, surround, scatter, regroup, kamikaze…)
- 6 formations: wedge, line, circle, diamond, column, spread
- Nearest-drone slot assignment for formations
- Tactical planner re-evaluates every 2 s based on health, numbers and score

### Reinforcement learning — `agents/rl/trainer.py`
- CPU-only PPO-style self-play; trained policies plug back into the behavior tree as `rl_policy`

### Edge AI simulator — `firmware/tinyml/inference.py`
- Models: MobileNetV1, SqueezeNet-Lite, TinyLSTM, TinyTransformer
- MCUs: ESP32, ESP32-S3, STM32F4, STM32H7, RPi Zero 2
- Latency with jitter, cache misses and interrupt latency; energy = power × latency
- Accuracy loss by quantization; p50/p95/p99 benchmark with budget-met %

### Hardware-in-the-loop — `firmware/`
- Simulated ESP32s with sensor noise, clock drift, RSSI variance, packet loss, battery curve
- HIL manager: fleet health, telemetry log, command delivery tracking
- Real ESP32 firmware: Wi-Fi + MQTT, FreeRTOS tasks, inference stub
- Fault injection mirroring real ESP32/STM32 failures

### Backend — `backend/`
- WebSocket broadcast at 60 FPS; session create / pause / delete
- REST: spawn drones, issue commands, telemetry, benchmarks, replay, analytics
- `/hil/inject` merges real hardware data into the simulation
- Redis pub/sub for multi-server fan-out; TimescaleDB telemetry persistence

### Dashboard — `dashboard/`
- Lobby to configure teams and launch sessions
- 2D canvas arena and a three.js 3D view
- HUD: scoreboard, drone inspector, kill feed, NLP terminal, HIL fleet health
- Analytics and benchmark pages (latency, accuracy vs quantization, energy)

## NLP command examples

```
"Red team, attack the center in wedge formation"
"Defend the nexus with circle formation"
"Flank the blue team from the east"
"All units, regroup at alpha point"
"Scatter and capture all control points"
```

## Edge AI benchmark (simulated, ESP32)

| Quantization | Latency p50 | Accuracy | Energy / inference | Budget met |
|---|---|---|---|---|
| 32-bit | 85 ms | 87.0% | 20.4 µJ | 0% |
| 16-bit | 47 ms | 86.5% | 11.3 µJ | 36% |
| **8-bit** | **24 ms** | **85.0%** | **5.8 µJ** | **78%** |
| 4-bit | 14 ms | 81.0% | 3.4 µJ | 98% |

8-bit is the sweet spot: 3.5× faster than 32-bit for a 2-point accuracy drop.
These figures come from the simulator's model, not measurements on hardware.

## License

[MIT](LICENSE)
