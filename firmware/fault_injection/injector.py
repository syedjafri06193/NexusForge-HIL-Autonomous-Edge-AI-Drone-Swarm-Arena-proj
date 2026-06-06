"""
NexusForge Fault Injection System
Simulates real-world hardware and network failures to test swarm resilience.
Each fault type mirrors something that actually happens on ESP32/STM32 hardware.
"""

import random
import time
import math
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict
from enum import Enum


class FaultType(Enum):
    # Network faults
    PACKET_LOSS_SPIKE    = "packet_loss_spike"    # Sudden high packet loss
    LATENCY_SPIKE        = "latency_spike"         # RTT spikes to 200ms+
    BANDWIDTH_THROTTLE   = "bandwidth_throttle"    # Reduced uplink capacity
    MQTT_DISCONNECT      = "mqtt_disconnect"        # Broker connection lost

    # Power faults
    BATTERY_DRAIN        = "battery_drain"          # Accelerated discharge
    BROWNOUT             = "brownout"               # Voltage sag, reduced performance
    POWER_OFF            = "power_off"              # Complete power loss

    # Compute faults
    CPU_OVERLOAD         = "cpu_overload"           # Inference misses deadline
    MEMORY_PRESSURE      = "memory_pressure"        # Heap fragmentation
    INFERENCE_CORRUPT    = "inference_corrupt"      # Bad model weights / quantization errors
    WATCHDOG_RESET       = "watchdog_reset"         # MCU restarts

    # Sensor faults
    GPS_DRIFT            = "gps_drift"              # Position estimate diverges
    IMU_BIAS             = "imu_bias"               # Accelerometer/gyro bias
    SENSOR_FREEZE        = "sensor_freeze"          # Sensor stops updating

    # Swarm faults
    SPLIT_BRAIN          = "split_brain"            # Drones lose formation coherence
    COMMAND_FLOOD        = "command_flood"          # Too many commands overwhelm MCU
    ROGUE_DRONE          = "rogue_drone"            # Single drone ignores commands


@dataclass
class FaultEvent:
    fault_type: FaultType
    target_drone_ids: List[str]
    severity: float          # 0.0 - 1.0
    duration_s: float        # how long fault lasts
    start_time: float = field(default_factory=time.time)
    description: str = ""
    active: bool = True

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def remaining(self) -> float:
        return max(0.0, self.duration_s - self.elapsed)

    @property
    def is_expired(self) -> bool:
        return self.elapsed >= self.duration_s

    def to_dict(self) -> dict:
        return {
            "type":     self.fault_type.value,
            "targets":  self.target_drone_ids,
            "severity": round(self.severity, 2),
            "duration": self.duration_s,
            "elapsed":  round(self.elapsed, 1),
            "remaining": round(self.remaining, 1),
            "active":   self.active and not self.is_expired,
        }


class FaultInjector:
    """
    Applies fault scenarios to drones during simulation.
    Called each tick to modify drone state / telemetry based on active faults.
    """

    def __init__(self):
        self.active_faults: List[FaultEvent] = []
        self.fault_history: List[dict] = []
        self._on_fault_start: Optional[Callable] = None
        self._on_fault_end: Optional[Callable]   = None

    def inject(self, fault: FaultEvent):
        """Inject a new fault event."""
        self.active_faults.append(fault)
        self.fault_history.append({**fault.to_dict(), "injected_at": time.time()})
        if self._on_fault_start:
            self._on_fault_start(fault)

    def inject_random(self, drone_ids: List[str], max_targets: int = 3) -> FaultEvent:
        """Inject a random fault on a random subset of drones."""
        fault_type = random.choice(list(FaultType))
        n_targets  = random.randint(1, min(max_targets, len(drone_ids)))
        targets    = random.sample(drone_ids, n_targets)
        severity   = random.uniform(0.3, 0.9)
        duration   = random.uniform(3.0, 15.0)

        fault = FaultEvent(
            fault_type=fault_type,
            target_drone_ids=targets,
            severity=severity,
            duration_s=duration,
            description=f"Random {fault_type.value} on {n_targets} drone(s)",
        )
        self.inject(fault)
        return fault

    def apply_to_simulation(self, sim):
        """
        Apply all active faults to the simulation.
        Call this each tick before behavior trees run.
        """
        # Expire old faults
        expired = [f for f in self.active_faults if f.is_expired]
        for f in expired:
            f.active = False
            if self._on_fault_end:
                self._on_fault_end(f)
        self.active_faults = [f for f in self.active_faults if not f.is_expired]

        # Apply active faults
        for fault in self.active_faults:
            for drone_id in fault.target_drone_ids:
                drone = sim.drones.get(drone_id)
                if not drone or not drone.is_alive:
                    continue
                self._apply_fault(fault, drone, sim)

    def _apply_fault(self, fault: FaultEvent, drone, sim):
        s = fault.severity   # 0-1

        if fault.fault_type == FaultType.PACKET_LOSS_SPIKE:
            # Increase simulated latency
            drone.latency_ms = drone.latency_ms + s * 80 * random.gauss(1.0, 0.3)

        elif fault.fault_type == FaultType.LATENCY_SPIKE:
            drone.latency_ms = 50 + s * 200 * random.gauss(1.0, 0.2)

        elif fault.fault_type == FaultType.BATTERY_DRAIN:
            drain_rate = s * 2.0  # extra % per tick
            drone.battery_pct = max(0, drone.battery_pct - drain_rate / 60)
            if drone.battery_pct < 5:
                drone.stun_timer = max(drone.stun_timer, 0.3)

        elif fault.fault_type == FaultType.BROWNOUT:
            # Reduce max speed and increase inference time
            from simulation.engine.sim import MAX_SPEED
            speed = drone.velocity.length()
            if speed > MAX_SPEED * (1 - s * 0.5):
                drone.velocity = drone.velocity * (1 - s * 0.1)
            drone.inference_ms *= (1 + s * 2.0)

        elif fault.fault_type == FaultType.POWER_OFF:
            # Hard kill
            if random.random() < s * 0.02:
                drone.health = 0
                from simulation.engine.sim import DroneState
                drone.state = DroneState.DEAD

        elif fault.fault_type == FaultType.CPU_OVERLOAD:
            # Inflate inference time past budget
            drone.inference_ms = 20 + s * 40 + random.gauss(0, 5)
            # Stun slightly (missed control loop deadline)
            if random.random() < s * 0.3:
                drone.stun_timer = max(drone.stun_timer, 0.1)

        elif fault.fault_type == FaultType.MEMORY_PRESSURE:
            # Random stall behavior
            if random.random() < s * 0.05:
                drone.stun_timer = max(drone.stun_timer, 0.2)

        elif fault.fault_type == FaultType.INFERENCE_CORRUPT:
            # Force random action (corrupted model weights)
            from simulation.engine.sim import Vec2, MAX_SPEED
            if random.random() < s * 0.4:
                drone.apply_force(Vec2(
                    random.gauss(0, 1) * MAX_SPEED * s,
                    random.gauss(0, 1) * MAX_SPEED * s,
                ))

        elif fault.fault_type == FaultType.WATCHDOG_RESET:
            # Brief stun simulating MCU restart
            if random.random() < s * 0.01:
                drone.stun_timer = max(drone.stun_timer, 2.0)
                drone.latency_ms = 500  # reconnect delay
                drone.inference_ms = 0

        elif fault.fault_type == FaultType.GPS_DRIFT:
            # Add noise to reported position (doesn't affect actual position)
            from simulation.engine.sim import Vec2
            noise = Vec2(
                random.gauss(0, s * 30),
                random.gauss(0, s * 30),
            )
            drone.position = drone.position + noise * 0.02

        elif fault.fault_type == FaultType.IMU_BIAS:
            # Constant heading bias
            drone.heading += s * 0.05 * random.choice([-1, 1])

        elif fault.fault_type == FaultType.SENSOR_FREEZE:
            # Drone can't see neighbors or enemies
            if random.random() < s * 0.8:
                drone.neighbors = []
                drone.visible_enemies = []

        elif fault.fault_type == FaultType.SPLIT_BRAIN:
            # Force drone to move away from formation
            from simulation.engine.sim import Vec2, MAX_SPEED
            if drone.waypoint:
                # Move to wrong waypoint
                drone.waypoint = Vec2(
                    drone.waypoint.x + random.gauss(0, s * 100),
                    drone.waypoint.y + random.gauss(0, s * 100),
                )

        elif fault.fault_type == FaultType.ROGUE_DRONE:
            # Drone attacks its own team
            if drone.neighbors and random.random() < s * 0.3:
                target = random.choice(drone.neighbors)
                dist = drone.position.distance_to(target.position)
                if dist < 80 and drone.weapon_cooldown <= 0:
                    sim._fire_weapon(drone, target)

    def clear_all(self):
        self.active_faults.clear()

    def get_status(self) -> dict:
        return {
            "active":  [f.to_dict() for f in self.active_faults],
            "history": self.fault_history[-20:],
            "total_injected": len(self.fault_history),
        }


# ─── Scenario presets ─────────────────────────────────────────────────────────

class FaultScenarios:
    """Pre-built fault scenarios for testing swarm resilience."""

    @staticmethod
    def network_degradation(injector: FaultInjector, drone_ids: List[str]):
        """Simulate a WiFi congestion event affecting all drones."""
        injector.inject(FaultEvent(
            fault_type=FaultType.PACKET_LOSS_SPIKE,
            target_drone_ids=drone_ids,
            severity=0.6,
            duration_s=10.0,
            description="WiFi congestion: 30% packet loss across swarm",
        ))
        injector.inject(FaultEvent(
            fault_type=FaultType.LATENCY_SPIKE,
            target_drone_ids=random.sample(drone_ids, max(1, len(drone_ids) // 2)),
            severity=0.4,
            duration_s=8.0,
            description="RTT spikes on half the swarm",
        ))

    @staticmethod
    def battery_crisis(injector: FaultInjector, drone_ids: List[str]):
        """Simulate sudden battery drain on a subset of drones."""
        targets = random.sample(drone_ids, max(1, len(drone_ids) // 3))
        injector.inject(FaultEvent(
            fault_type=FaultType.BATTERY_DRAIN,
            target_drone_ids=targets,
            severity=0.85,
            duration_s=20.0,
            description=f"Critical battery drain on {len(targets)} drones",
        ))

    @staticmethod
    def inference_failure(injector: FaultInjector, drone_ids: List[str]):
        """Simulate TinyML model corruption on edge devices."""
        targets = random.sample(drone_ids, max(1, len(drone_ids) // 4))
        injector.inject(FaultEvent(
            fault_type=FaultType.INFERENCE_CORRUPT,
            target_drone_ids=targets,
            severity=0.7,
            duration_s=15.0,
            description="Quantization error: model weights corrupted",
        ))
        injector.inject(FaultEvent(
            fault_type=FaultType.CPU_OVERLOAD,
            target_drone_ids=targets,
            severity=0.5,
            duration_s=15.0,
            description="Inference deadline missed — fallback to rule-based",
        ))

    @staticmethod
    def cascade_failure(injector: FaultInjector, drone_ids: List[str]):
        """Multi-fault cascade: power + network + compute."""
        FaultScenarios.battery_crisis(injector, drone_ids)
        FaultScenarios.network_degradation(injector, drone_ids)
        FaultScenarios.inference_failure(injector, drone_ids)

    @staticmethod
    def rogue_unit(injector: FaultInjector, drone_ids: List[str]):
        """One drone goes rogue and attacks its own team."""
        target = [random.choice(drone_ids)]
        injector.inject(FaultEvent(
            fault_type=FaultType.ROGUE_DRONE,
            target_drone_ids=target,
            severity=1.0,
            duration_s=30.0,
            description=f"Drone {target[0]} compromised — friendly fire!",
        ))
