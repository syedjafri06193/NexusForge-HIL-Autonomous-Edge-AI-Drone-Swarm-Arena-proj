"""
NexusForge HIL Protocol Simulator
Simulates ESP32/STM32 hardware connected over MQTT.
Generates realistic telemetry with jitter, packet loss, and power modeling.
Real boards can connect to the same MQTT broker and inject real telemetry.
"""

import asyncio
import json
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

try:
    import aiomqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False


# ─── HIL config ───────────────────────────────────────────────────────────────

@dataclass
class HILConfig:
    broker_host: str = "localhost"
    broker_port: int = 1883
    # Simulated network characteristics
    base_latency_ms: float = 8.0
    latency_jitter_ms: float = 3.0
    packet_loss_rate: float = 0.005    # 0.5% loss
    bandwidth_kbps: float = 250.0      # BLE-ish
    # Telemetry rates
    telemetry_hz: float = 20.0         # 20 Hz telemetry uplink
    command_ack_hz: float = 50.0       # 50 Hz command reception


# ─── MQTT topic schema ────────────────────────────────────────────────────────

TOPIC_TELEMETRY  = "nexusforge/{session}/{drone_id}/telemetry"
TOPIC_COMMAND    = "nexusforge/{session}/{drone_id}/command"
TOPIC_ACK        = "nexusforge/{session}/{drone_id}/ack"
TOPIC_STATUS     = "nexusforge/{session}/status"
TOPIC_SWARM_CMD  = "nexusforge/{session}/swarm/command"
TOPIC_BROADCAST  = "nexusforge/{session}/broadcast"


# ─── Telemetry packet ─────────────────────────────────────────────────────────

@dataclass
class HWTelemetry:
    """Mirrors what a real ESP32 board would send over MQTT."""
    drone_id: str
    session_id: str
    seq: int                       # packet sequence number
    ts_device: float               # device timestamp (may drift)
    ts_server: float               # server receive time

    # IMU / navigation
    pos_x: float
    pos_y: float
    vel_x: float
    vel_y: float
    heading_deg: float
    accel_x: float
    accel_y: float

    # Power
    battery_mv: float              # millivolts
    battery_pct: float
    current_ma: float
    power_mw: float

    # Compute
    cpu_load_pct: float
    heap_free_kb: float
    inference_us: int              # microseconds

    # Radio
    rssi_dbm: int
    snr_db: float
    packet_loss_pct: float

    # Combat
    health: float
    shield: float
    kills: int
    state: str

    def to_dict(self) -> dict:
        return {
            "drone_id": self.drone_id,
            "session_id": self.session_id,
            "seq": self.seq,
            "ts": self.ts_server,
            "dt": round(self.ts_server - self.ts_device, 4),
            "pos": {"x": round(self.pos_x, 2), "y": round(self.pos_y, 2)},
            "vel": {"x": round(self.vel_x, 2), "y": round(self.vel_y, 2)},
            "heading_deg": round(self.heading_deg, 1),
            "accel": {"x": round(self.accel_x, 3), "y": round(self.accel_y, 3)},
            "power": {
                "battery_mv": round(self.battery_mv, 0),
                "battery_pct": round(self.battery_pct, 1),
                "current_ma": round(self.current_ma, 1),
                "power_mw": round(self.power_mw, 1),
            },
            "compute": {
                "cpu_load_pct": round(self.cpu_load_pct, 1),
                "heap_free_kb": round(self.heap_free_kb, 1),
                "inference_us": self.inference_us,
            },
            "radio": {
                "rssi_dbm": self.rssi_dbm,
                "snr_db": round(self.snr_db, 1),
                "packet_loss_pct": round(self.packet_loss_pct, 2),
            },
            "combat": {
                "health": round(self.health, 1),
                "shield": round(self.shield, 1),
                "kills": self.kills,
                "state": self.state,
            },
        }

    def to_bytes(self) -> bytes:
        return json.dumps(self.to_dict()).encode()


# ─── Simulated ESP32 node ─────────────────────────────────────────────────────

class SimulatedESP32:
    """
    Simulates an ESP32 board generating telemetry from a virtual drone.
    In real HIL deployment, replace with actual firmware over MQTT.
    """

    def __init__(self, drone_id: str, session_id: str, config: HILConfig):
        self.drone_id = drone_id
        self.session_id = session_id
        self.config = config
        self._seq = 0
        self._start_mv = random.uniform(3950, 4200)  # LiPo voltage
        self._base_heap = random.uniform(180, 320)
        self._rssi_base = random.randint(-65, -45)
        self._running = False

        # State injected from simulation
        self.pos_x = 600.0
        self.pos_y = 450.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.heading = 0.0
        self.health = 100.0
        self.shield = 50.0
        self.battery_pct = 100.0
        self.kills = 0
        self.state = "patrolling"
        self.inference_us = 8000

    def update_from_drone(self, drone_dict: dict):
        """Sync state from simulation drone dict."""
        self.pos_x = drone_dict.get("position", {}).get("x", self.pos_x)
        self.pos_y = drone_dict.get("position", {}).get("y", self.pos_y)
        self.vel_x = drone_dict.get("velocity", {}).get("x", self.vel_x)
        self.vel_y = drone_dict.get("velocity", {}).get("y", self.vel_y)
        self.heading = math.degrees(drone_dict.get("heading", 0.0))
        self.health = drone_dict.get("health", self.health)
        self.shield = drone_dict.get("shield", self.shield)
        self.battery_pct = drone_dict.get("battery_pct", self.battery_pct)
        self.kills = drone_dict.get("kills", self.kills)
        self.state = drone_dict.get("state", self.state)
        self.inference_us = int(drone_dict.get("inference_ms", 8.0) * 1000)

    def generate_telemetry(self) -> HWTelemetry:
        now = time.time()
        self._seq += 1

        # Simulate clock drift on device (±50ppm)
        drift = random.gauss(0, 0.00005)
        ts_device = now + drift * (now % 3600)

        # Power model
        speed = math.sqrt(self.vel_x ** 2 + self.vel_y ** 2)
        motion_load = speed / 280.0
        cpu_load = 30 + motion_load * 20 + random.gauss(0, 5)
        current_ma = 150 + motion_load * 200 + cpu_load * 0.8 + random.gauss(0, 10)
        power_mw = current_ma * (self._start_mv * (self.battery_pct / 100.0)) / 1000.0
        battery_mv = self._start_mv * max(0.6, self.battery_pct / 100.0)

        # Radio quality varies with distance from center
        dist_from_center = math.sqrt((self.pos_x - 600) ** 2 + (self.pos_y - 450) ** 2)
        rssi = self._rssi_base - int(dist_from_center / 50) + random.randint(-3, 3)
        snr = random.gauss(18.0, 2.5)
        pkt_loss = max(0.0, self.config.packet_loss_rate * 100 + dist_from_center / 2000 * 5)

        return HWTelemetry(
            drone_id=self.drone_id,
            session_id=self.session_id,
            seq=self._seq,
            ts_device=ts_device,
            ts_server=now,
            pos_x=self.pos_x + random.gauss(0, 0.2),   # sensor noise
            pos_y=self.pos_y + random.gauss(0, 0.2),
            vel_x=self.vel_x + random.gauss(0, 0.5),
            vel_y=self.vel_y + random.gauss(0, 0.5),
            heading_deg=self.heading + random.gauss(0, 0.3),
            accel_x=random.gauss(0, 2.0),
            accel_y=random.gauss(0, 2.0),
            battery_mv=battery_mv,
            battery_pct=self.battery_pct,
            current_ma=max(0, current_ma),
            power_mw=max(0, power_mw),
            cpu_load_pct=max(0, min(100, cpu_load)),
            heap_free_kb=max(10, self._base_heap - cpu_load * 0.3),
            inference_us=self.inference_us + random.randint(-500, 500),
            rssi_dbm=max(-120, min(-20, rssi)),
            snr_db=snr,
            packet_loss_pct=max(0, pkt_loss),
            health=self.health,
            shield=self.shield,
            kills=self.kills,
            state=self.state,
        )

    def should_drop_packet(self) -> bool:
        """Simulate packet loss."""
        return random.random() < self.config.packet_loss_rate

    def get_latency_ms(self) -> float:
        """Return simulated round-trip latency."""
        base = self.config.base_latency_ms
        jitter = random.gauss(0, self.config.latency_jitter_ms)
        # Occasionally spike (retransmit)
        spike = random.gauss(20, 5) if random.random() < 0.02 else 0
        return max(0.5, base + jitter + spike)


# ─── HIL Manager ──────────────────────────────────────────────────────────────

class HILManager:
    """
    Manages a pool of simulated (or real) ESP32 nodes.
    Handles telemetry ingestion, command dispatch, and latency accounting.
    """

    def __init__(self, session_id: str, config: Optional[HILConfig] = None):
        self.session_id = session_id
        self.config = config or HILConfig()
        self.nodes: Dict[str, SimulatedESP32] = {}
        self._telemetry_log: List[dict] = []
        self._command_log: List[dict] = []
        self._running = False

    def register_drone(self, drone_id: str) -> SimulatedESP32:
        node = SimulatedESP32(drone_id, self.session_id, self.config)
        self.nodes[drone_id] = node
        return node

    def update_from_sim(self, drones: List[dict]):
        """Sync all drone states from simulation snapshot."""
        for d in drones:
            node = self.nodes.get(d["id"])
            if node:
                node.update_from_drone(d)

    def collect_telemetry(self) -> List[dict]:
        """Generate telemetry from all nodes (simulating a poll cycle)."""
        packets = []
        for node in self.nodes.values():
            if node.should_drop_packet():
                continue
            # Simulate latency delay (just metadata, not actual sleep)
            telem = node.generate_telemetry()
            d = telem.to_dict()
            d["latency_ms"] = round(node.get_latency_ms(), 2)
            packets.append(d)

        self._telemetry_log.extend(packets)
        if len(self._telemetry_log) > 10_000:
            self._telemetry_log = self._telemetry_log[-10_000:]
        return packets

    def send_command(self, drone_id: str, command: dict) -> bool:
        """Dispatch a control command to a drone node."""
        node = self.nodes.get(drone_id)
        if not node:
            return False
        cmd_record = {
            "drone_id": drone_id,
            "command": command,
            "issued_at": time.time(),
            "latency_ms": node.get_latency_ms(),
            "delivered": not node.should_drop_packet(),
        }
        self._command_log.append(cmd_record)
        if len(self._command_log) > 2000:
            self._command_log = self._command_log[-2000:]
        return cmd_record["delivered"]

    def get_fleet_health(self) -> dict:
        """Aggregate health metrics across all nodes."""
        if not self.nodes:
            return {}
        recent_pkts = self._telemetry_log[-len(self.nodes) * 5:]
        latencies = [p.get("latency_ms", 8) for p in recent_pkts]
        return {
            "node_count": len(self.nodes),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
            "p99_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0, 2),
            "telemetry_packets": len(self._telemetry_log),
            "commands_sent": len(self._command_log),
            "command_delivery_rate": round(
                sum(1 for c in self._command_log[-100:] if c["delivered"]) /
                max(1, len(self._command_log[-100:])) * 100, 1
            ),
        }

    def get_recent_telemetry(self, n: int = 50) -> List[dict]:
        return self._telemetry_log[-n:]

    def get_recent_commands(self, n: int = 20) -> List[dict]:
        return self._command_log[-n:]
