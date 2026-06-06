"""
NexusForge Demo Script
Runs a headless simulation and prints real-time stats.
Good for CI, benchmarking, and quick sanity checks.

Usage:
  python demos/run_demo.py
  python demos/run_demo.py --teams 4 --drones 16 --ticks 600 --faults
"""

import sys
import os
import time
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.engine.sim import Simulation, TeamID, DT
from agents.swarm.orchestrator import SwarmOrchestrator, SwarmCommand, MissionType, Formation
from firmware.protocols.hil_mqtt import HILManager
from firmware.tinyml.inference import EdgeInferenceEngine
from firmware.fault_injection.injector import FaultInjector, FaultScenarios


def run_demo(args):
    print("\n" + "=" * 60)
    print("  NexusForge Headless Demo")
    print(f"  {args.teams} teams × {args.drones} drones = {args.teams * args.drones} total")
    print(f"  {args.ticks} ticks at 60 FPS = {args.ticks / 60:.1f}s")
    print("=" * 60 + "\n")

    # ── Boot simulation ──────────────────────────────────────────────────────
    sim = Simulation(num_teams=args.teams, drones_per_team=args.drones)
    orch = SwarmOrchestrator(sim)
    hil  = HILManager("demo")
    for did in sim.drones:
        hil.register_drone(did)

    fault_injector = FaultInjector() if args.faults else None

    # ── Edge AI benchmark (printed once) ────────────────────────────────────
    print("[Edge AI] Running quick benchmark on ESP32...")
    engine = EdgeInferenceEngine("esp32", bits=8)
    bench = engine.benchmark("trajectory_predictor", n_runs=50)
    print(f"  Model: {bench['model']} | MCU: {bench['mcu']} | Bits: {bench['bits']}")
    print(f"  Latency P50={bench['latency_ms']['p50']}ms  P99={bench['latency_ms']['p99']}ms")
    print(f"  Accuracy={bench['accuracy']['mean']*100:.1f}%  Budget met={bench['budget_met_pct']}%")
    print(f"  Energy per inference: {bench['energy_uj']['mean']}µJ\n")

    # ── NLP commands ────────────────────────────────────────────────────────
    print("[NLP] Parsing swarm commands...")
    test_commands = [
        ("Red team, attack the center in wedge formation", TeamID.RED),
        ("Defend the nexus with circle formation",         TeamID.BLUE),
        ("Flank the enemy from the east",                  TeamID.RED),
    ]
    for text, team in test_commands:
        cmd = orch.issue_nlp_command(text, team)
        if cmd:
            print(f"  '{text}'")
            print(f"  → {cmd.mission.value} | {cmd.formation} | src={cmd.source}")
    print()

    # ── Simulation loop ──────────────────────────────────────────────────────
    print("[Simulation] Running...")
    t0 = time.perf_counter()
    report_every = args.ticks // 10

    for tick in range(args.ticks):
        # Orchestrator auto-tactics
        orch.update()

        # Inject faults mid-simulation
        if fault_injector and tick == args.ticks // 3:
            print(f"\n  [Fault] Injecting cascade failure at tick {tick}!")
            FaultScenarios.cascade_failure(fault_injector, list(sim.drones.keys()))
        if fault_injector:
            fault_injector.apply_to_simulation(sim)

        sim.tick_once(DT)

        # HIL telemetry update
        hil.update_from_sim([d.to_dict() for d in sim.drones.values()])
        if tick % 3 == 0:  # 20Hz
            hil.collect_telemetry()

        if (tick + 1) % report_every == 0:
            alive = sum(1 for d in sim.drones.values() if d.is_alive)
            elapsed = (tick + 1) / 60
            score_str = " | ".join(f"{t}:{s}" for t, s in sorted(sim.scores.items(), key=lambda x: -x[1]))
            print(f"  T={elapsed:5.1f}s | Tick={tick+1:4d} | Alive={alive:3d}/{len(sim.drones)} | Scores: {score_str}")

    wall_time = time.perf_counter() - t0
    real_time  = args.ticks / 60
    print()
    print(f"[Done] Simulated {real_time:.1f}s of arena time in {wall_time:.2f}s wall time")
    print(f"       Speedup: {real_time / wall_time:.1f}x realtime\n")

    # ── Final stats ──────────────────────────────────────────────────────────
    print("── Final Scores ─────────────────────────────────────────────")
    sorted_scores = sorted(sim.scores.items(), key=lambda x: -x[1])
    for i, (team, score) in enumerate(sorted_scores):
        drones = [d for d in sim.drones.values() if d.team.name == team]
        total_kills = sum(d.kills for d in drones)
        alive = sum(1 for d in drones if d.is_alive)
        print(f"  {'🥇' if i==0 else '🥈' if i==1 else '🥉' if i==2 else '  '} "
              f"{team:5s}: {score:4d} pts | {total_kills} kills | {alive}/{len(drones)} alive")

    print("\n── Leaderboard (Top 5) ──────────────────────────────────────")
    lb = sim.get_leaderboard()[:5]
    for i, row in enumerate(lb):
        print(f"  {i+1}. [{row['team']:5s}] drone {row['id']} | "
              f"K:{row['kills']:2d} D:{row['deaths']:3d} DMG:{row['damage']:6.0f}")

    print("\n── HIL Fleet Health ─────────────────────────────────────────")
    fleet = hil.get_fleet_health()
    print(f"  Nodes: {fleet.get('node_count',0)} | "
          f"Avg latency: {fleet.get('avg_latency_ms',0):.1f}ms | "
          f"Telemetry packets: {fleet.get('telemetry_packets',0)}")
    print(f"  Command delivery: {fleet.get('command_delivery_rate',100):.1f}%")

    if fault_injector:
        fstatus = fault_injector.get_status()
        print(f"\n── Fault Injection ──────────────────────────────────────────")
        print(f"  Total injected: {fstatus['total_injected']} | Active: {len(fstatus['active'])}")

    print("\n── Swarm AI Status ──────────────────────────────────────────")
    swarm_status = orch.get_status()
    for team, info in swarm_status["teams"].items():
        if info["alive"] > 0:
            print(f"  {team:5s}: {info['mission']:12s} | {info['formation']:10s} | {info['alive']} alive")

    print("\n[NexusForge Demo Complete]\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NexusForge headless demo")
    parser.add_argument("--teams",  type=int, default=2,   help="Number of teams (2-4)")
    parser.add_argument("--drones", type=int, default=8,   help="Drones per team")
    parser.add_argument("--ticks",  type=int, default=600, help="Simulation ticks to run")
    parser.add_argument("--faults", action="store_true",   help="Enable fault injection demo")
    args = parser.parse_args()
    run_demo(args)
