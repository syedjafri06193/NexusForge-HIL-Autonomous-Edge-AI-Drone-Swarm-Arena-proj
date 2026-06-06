"""
NexusForge Simulation Engine
Core drone physics, arena management, and simulation loop.
Runs headless (for backend) or with Pygame rendering.
"""

import asyncio
import math
import time
import random
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import numpy as np


# ─── Constants ────────────────────────────────────────────────────────────────

SIM_FPS        = 60
DT             = 1.0 / SIM_FPS        # seconds per tick
ARENA_W        = 1200.0               # pixels / world units
ARENA_H        = 900.0
MAX_DRONES     = 128
GRAVITY        = 0.0                  # 2D top-down, no gravity
DRAG           = 0.92                 # velocity damping per tick
MAX_SPEED      = 280.0               # world units / sec
MAX_ACCEL      = 380.0
WEAPON_RANGE   = 80.0
WEAPON_DAMAGE  = 15.0
WEAPON_COOLDOWN = 0.6                 # seconds
SHIELD_REGEN   = 2.0                  # HP/sec when not hit
HEALTH_MAX     = 100.0
SHIELD_MAX     = 50.0
COLLISION_R    = 14.0                 # drone collision radius
SENSOR_RANGE   = 240.0               # neighbor detection radius


# ─── Enums ────────────────────────────────────────────────────────────────────

class DroneState(Enum):
    IDLE       = "idle"
    PATROLLING = "patrolling"
    PURSUING   = "pursuing"
    ATTACKING  = "attacking"
    EVADING    = "evading"
    REGROUPING = "regrouping"
    DEAD       = "dead"

class TeamID(Enum):
    RED   = 0
    BLUE  = 1
    GREEN = 2
    GOLD  = 3

class HazardType(Enum):
    PLASMA_STORM   = "plasma_storm"
    GRAVITY_WELL   = "gravity_well"
    EMP_PULSE      = "emp_pulse"
    SHIELD_DISRUPT = "shield_disrupt"


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def __add__(self, o: 'Vec2') -> 'Vec2':
        return Vec2(self.x + o.x, self.y + o.y)

    def __sub__(self, o: 'Vec2') -> 'Vec2':
        return Vec2(self.x - o.x, self.y - o.y)

    def __mul__(self, s: float) -> 'Vec2':
        return Vec2(self.x * s, self.y * s)

    def __truediv__(self, s: float) -> 'Vec2':
        return Vec2(self.x / s, self.y / s)

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)

    def normalized(self) -> 'Vec2':
        l = self.length()
        return Vec2(self.x / l, self.y / l) if l > 0.001 else Vec2(0, 0)

    def dot(self, o: 'Vec2') -> float:
        return self.x * o.x + self.y * o.y

    def distance_to(self, o: 'Vec2') -> float:
        return (self - o).length()

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def to_dict(self) -> dict:
        return {"x": round(self.x, 2), "y": round(self.y, 2)}

    @classmethod
    def random_in_arena(cls) -> 'Vec2':
        return cls(random.uniform(50, ARENA_W - 50), random.uniform(50, ARENA_H - 50))

    @classmethod
    def from_angle(cls, angle_rad: float, magnitude: float = 1.0) -> 'Vec2':
        return cls(math.cos(angle_rad) * magnitude, math.sin(angle_rad) * magnitude)


@dataclass
class TelemetryPacket:
    drone_id: str
    timestamp: float
    position: Vec2
    velocity: Vec2
    heading: float
    health: float
    shield: float
    state: str
    team: str
    latency_ms: float        # simulated HIL round-trip
    battery_pct: float       # edge power model
    inference_ms: float      # simulated TinyML inference time
    kills: int
    hits_taken: int


@dataclass
class ArenaHazard:
    id: str
    htype: HazardType
    position: Vec2
    radius: float
    intensity: float
    duration: float          # seconds remaining
    created_at: float


@dataclass
class Projectile:
    id: str
    owner_id: str
    team: TeamID
    position: Vec2
    velocity: Vec2
    damage: float
    lifetime: float          # seconds remaining


@dataclass
class DroneConfig:
    """Per-drone hardware / AI configuration"""
    # Edge AI simulation
    inference_budget_ms: float = 20.0   # max allowed inference time
    quantization_bits: int = 8          # 4, 8, 16, 32
    model_type: str = "behavior_tree"   # behavior_tree | rl_policy | rule_based
    # Hardware sim
    cpu_mhz: int = 240                  # ESP32 = 240 MHz
    ram_kb: int = 520                   # ESP32 typical
    battery_mwh: float = 2000.0
    # Weapon
    weapon_type: str = "laser"          # laser | missile | emp
    # Behavior
    aggression: float = 0.5            # 0=pacifist, 1=berserker
    cohesion: float = 0.6              # swarm cohesion weight
    separation: float = 0.4
    alignment: float = 0.3


# ─── Drone ────────────────────────────────────────────────────────────────────

class Drone:
    def __init__(self, team: TeamID, position: Optional[Vec2] = None, config: Optional[DroneConfig] = None):
        self.id = str(uuid.uuid4())[:8]
        self.team = team
        self.config = config or DroneConfig()
        self.position = position or Vec2.random_in_arena()
        self.velocity = Vec2()
        self.acceleration = Vec2()
        self.heading = random.uniform(0, math.tau)  # radians

        self.health = HEALTH_MAX
        self.shield = SHIELD_MAX
        self.state = DroneState.PATROLLING
        self.target_id: Optional[str] = None
        self.waypoint: Optional[Vec2] = None

        self.weapon_cooldown = 0.0
        self.stun_timer = 0.0           # EMP stun
        self.kills = 0
        self.hits_taken = 0
        self.damage_dealt = 0.0

        self.created_at = time.time()
        self.last_hit_at = 0.0
        self.battery_pct = 100.0

        # HIL simulation
        self.latency_ms = random.gauss(8.0, 2.0)   # simulated network jitter
        self.inference_ms = 0.0

        # Swarm neighbors (updated each tick)
        self.neighbors: List['Drone'] = []
        self.visible_enemies: List['Drone'] = []

    @property
    def is_alive(self) -> bool:
        return self.state != DroneState.DEAD

    @property
    def effective_hp(self) -> float:
        return self.health + self.shield

    def take_damage(self, amount: float, now: float):
        self.last_hit_at = now
        if self.shield > 0:
            absorbed = min(self.shield, amount)
            self.shield -= absorbed
            amount -= absorbed
        self.health -= amount
        self.hits_taken += 1
        if self.health <= 0:
            self.health = 0
            self.state = DroneState.DEAD

    def apply_force(self, force: Vec2):
        self.acceleration = self.acceleration + force

    def get_telemetry(self) -> TelemetryPacket:
        return TelemetryPacket(
            drone_id=self.id,
            timestamp=time.time(),
            position=self.position,
            velocity=self.velocity,
            heading=self.heading,
            health=self.health,
            shield=self.shield,
            state=self.state.value,
            team=self.team.name,
            latency_ms=self.latency_ms,
            battery_pct=self.battery_pct,
            inference_ms=self.inference_ms,
            kills=self.kills,
            hits_taken=self.hits_taken,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "team": self.team.name,
            "state": self.state.value,
            "position": self.position.to_dict(),
            "velocity": self.velocity.to_dict(),
            "heading": round(self.heading, 3),
            "health": round(self.health, 1),
            "shield": round(self.shield, 1),
            "battery_pct": round(self.battery_pct, 1),
            "kills": self.kills,
            "hits_taken": self.hits_taken,
            "latency_ms": round(self.latency_ms, 1),
            "inference_ms": round(self.inference_ms, 2),
            "weapon_ready": self.weapon_cooldown <= 0,
        }


# ─── Arena ────────────────────────────────────────────────────────────────────

class Arena:
    """Manages hazards, obstacles, and arena geometry"""

    def __init__(self):
        self.width = ARENA_W
        self.height = ARENA_H
        self.hazards: List[ArenaHazard] = []
        self.obstacles: List[dict] = self._generate_obstacles()
        self.control_points: List[dict] = self._generate_control_points()
        self._hazard_timer = 0.0
        self._hazard_interval = 15.0     # seconds between new hazards

    def _generate_obstacles(self) -> List[dict]:
        """Static rectangular obstacles"""
        return [
            {"x": 350, "y": 200, "w": 80, "h": 120},
            {"x": 750, "y": 200, "w": 80, "h": 120},
            {"x": 350, "y": 580, "w": 80, "h": 120},
            {"x": 750, "y": 580, "w": 80, "h": 120},
            {"x": 540, "y": 380, "w": 120, "h": 80},
            {"x": 100, "y": 400, "w": 60, "h": 100},
            {"x": 1040, "y": 400, "w": 60, "h": 100},
        ]

    def _generate_control_points(self) -> List[dict]:
        return [
            {"id": "alpha",   "x": 200,  "y": 450, "r": 60, "owner": None, "capture": 0.0},
            {"id": "beta",    "x": 600,  "y": 200, "r": 60, "owner": None, "capture": 0.0},
            {"id": "gamma",   "x": 600,  "y": 700, "r": 60, "owner": None, "capture": 0.0},
            {"id": "delta",   "x": 1000, "y": 450, "r": 60, "owner": None, "capture": 0.0},
            {"id": "nexus",   "x": 600,  "y": 450, "r": 80, "owner": None, "capture": 0.0},
        ]

    def update(self, dt: float, drones: List[Drone]):
        now = time.time()
        # Age hazards
        self.hazards = [h for h in self.hazards if h.duration > 0]
        for h in self.hazards:
            h.duration -= dt

        # Spawn new hazard periodically
        self._hazard_timer += dt
        if self._hazard_timer >= self._hazard_interval:
            self._hazard_timer = 0.0
            self._spawn_hazard(now)

        # Update control point captures
        for cp in self.control_points:
            cp_pos = Vec2(cp["x"], cp["y"])
            for team in TeamID:
                count = sum(
                    1 for d in drones
                    if d.is_alive and d.team == team
                    and d.position.distance_to(cp_pos) < cp["r"]
                )
                if count > 0:
                    cp["capture"] = min(1.0, cp["capture"] + dt * 0.1 * count)
                    cp["owner"] = team.name
                    break
            else:
                cp["capture"] = max(0.0, cp["capture"] - dt * 0.05)

    def _spawn_hazard(self, now: float):
        htype = random.choice(list(HazardType))
        hazard = ArenaHazard(
            id=str(uuid.uuid4())[:6],
            htype=htype,
            position=Vec2(
                random.uniform(100, ARENA_W - 100),
                random.uniform(100, ARENA_H - 100),
            ),
            radius=random.uniform(60, 140),
            intensity=random.uniform(0.4, 1.0),
            duration=random.uniform(8, 20),
            created_at=now,
        )
        self.hazards.append(hazard)

    def apply_hazards_to_drone(self, drone: Drone, dt: float):
        for h in self.hazards:
            dist = drone.position.distance_to(h.position)
            if dist > h.radius:
                continue
            factor = (1.0 - dist / h.radius) * h.intensity
            if h.htype == HazardType.PLASMA_STORM:
                drone.take_damage(8.0 * factor * dt, time.time())
            elif h.htype == HazardType.GRAVITY_WELL:
                toward = (h.position - drone.position).normalized()
                drone.apply_force(toward * 200 * factor)
            elif h.htype == HazardType.EMP_PULSE:
                drone.stun_timer = max(drone.stun_timer, 0.5 * factor)
            elif h.htype == HazardType.SHIELD_DISRUPT:
                drone.shield = max(0, drone.shield - 5.0 * factor * dt)

    def is_in_obstacle(self, pos: Vec2) -> bool:
        for obs in self.obstacles:
            if (obs["x"] <= pos.x <= obs["x"] + obs["w"] and
                    obs["y"] <= pos.y <= obs["y"] + obs["h"]):
                return True
        return False

    def clamp_to_arena(self, pos: Vec2) -> Vec2:
        return Vec2(
            max(COLLISION_R, min(ARENA_W - COLLISION_R, pos.x)),
            max(COLLISION_R, min(ARENA_H - COLLISION_R, pos.y)),
        )

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "obstacles": self.obstacles,
            "control_points": self.control_points,
            "hazards": [
                {
                    "id": h.id,
                    "type": h.htype.value,
                    "x": round(h.position.x, 1),
                    "y": round(h.position.y, 1),
                    "radius": round(h.radius, 1),
                    "intensity": round(h.intensity, 2),
                    "duration": round(h.duration, 1),
                }
                for h in self.hazards
            ],
        }


# ─── Simulation ───────────────────────────────────────────────────────────────

class Simulation:
    """
    Main simulation loop. Manages all drones, projectiles,
    physics integration, and state broadcasting.
    """

    def __init__(self, num_teams: int = 2, drones_per_team: int = 8):
        self.arena = Arena()
        self.drones: Dict[str, Drone] = {}
        self.projectiles: Dict[str, Projectile] = {}
        self.tick = 0
        self.elapsed = 0.0
        self.running = False
        self.paused = False

        self.scores: Dict[str, int] = {t.name: 0 for t in TeamID}
        self.events: List[dict] = []          # kill feed, captures, etc.
        self.replay_frames: List[dict] = []   # for replay system
        self._record_replay = True
        self._replay_interval = 6             # record every N ticks

        self.on_state_update: Optional[Callable] = None  # WebSocket broadcast hook

        # Spawn initial drones
        teams = list(TeamID)[:num_teams]
        for team in teams:
            for _ in range(drones_per_team):
                self.spawn_drone(team)

    # ─── Spawn ──────────────────────────────────────────────────────────────

    def spawn_drone(self, team: TeamID, config: Optional[DroneConfig] = None) -> Drone:
        # Spawn zones by team
        spawn_zones = {
            TeamID.RED:   (50,  50,  200, 200),
            TeamID.BLUE:  (950, 650, 1150, 850),
            TeamID.GREEN: (50,  650, 200,  850),
            TeamID.GOLD:  (950, 50,  1150, 200),
        }
        z = spawn_zones[team]
        pos = Vec2(random.uniform(z[0], z[2]), random.uniform(z[1], z[3]))
        drone = Drone(team=team, position=pos, config=config)
        self.drones[drone.id] = drone
        return drone

    # ─── Physics ────────────────────────────────────────────────────────────

    def _integrate_drone(self, drone: Drone, dt: float):
        if not drone.is_alive or drone.stun_timer > 0:
            drone.stun_timer = max(0, drone.stun_timer - dt)
            drone.velocity = drone.velocity * 0.7  # decelerate when stunned
            return

        # Apply acceleration
        drone.velocity = drone.velocity + drone.acceleration * dt
        # Clamp speed
        spd = drone.velocity.length()
        if spd > MAX_SPEED:
            drone.velocity = drone.velocity * (MAX_SPEED / spd)
        # Drag
        drone.velocity = drone.velocity * DRAG

        new_pos = drone.position + drone.velocity * dt

        # Obstacle avoidance push
        if self.arena.is_in_obstacle(new_pos):
            # Simple bounce: reverse velocity component
            new_pos = drone.position
            drone.velocity = drone.velocity * -0.5

        drone.position = self.arena.clamp_to_arena(new_pos)
        drone.acceleration = Vec2()  # reset each tick

        # Update heading toward velocity
        if drone.velocity.length() > 5.0:
            drone.heading = math.atan2(drone.velocity.y, drone.velocity.x)

        # Shield regen (if not recently hit)
        now = time.time()
        if now - drone.last_hit_at > 2.0 and drone.shield < SHIELD_MAX:
            drone.shield = min(SHIELD_MAX, drone.shield + SHIELD_REGEN * dt)

        # Battery drain (power model)
        accel_load = drone.acceleration.length() / MAX_ACCEL
        drone.battery_pct = max(0, drone.battery_pct - (0.003 + 0.002 * accel_load) * dt)

        # Cooldowns
        drone.weapon_cooldown = max(0, drone.weapon_cooldown - dt)

    def _update_projectiles(self, dt: float):
        dead_proj = []
        for pid, proj in self.projectiles.items():
            proj.position = proj.position + proj.velocity * dt
            proj.lifetime -= dt

            # Out of bounds
            if (proj.position.x < 0 or proj.position.x > ARENA_W or
                    proj.position.y < 0 or proj.position.y > ARENA_H or
                    proj.lifetime <= 0):
                dead_proj.append(pid)
                continue

            # Obstacle hit
            if self.arena.is_in_obstacle(proj.position):
                dead_proj.append(pid)
                continue

            # Drone hit
            for drone in self.drones.values():
                if not drone.is_alive or drone.team.value == proj.team.value or drone.id == proj.owner_id:
                    continue
                if proj.position.distance_to(drone.position) < COLLISION_R:
                    now = time.time()
                    drone.take_damage(proj.damage, now)
                    dead_proj.append(pid)
                    # Log kill
                    if not drone.is_alive:
                        shooter = self.drones.get(proj.owner_id)
                        if shooter:
                            shooter.kills += 1
                            self.scores[shooter.team.name] = self.scores.get(shooter.team.name, 0) + 10
                        self.events.append({
                            "type": "kill",
                            "killer": proj.owner_id,
                            "victim": drone.id,
                            "victim_team": drone.team.name,
                            "tick": self.tick,
                        })
                    break

        for pid in dead_proj:
            self.projectiles.pop(pid, None)

    def _drone_collision(self, drones: List[Drone]):
        """Simple O(n²) collision resolution — fine for 128 agents"""
        alive = [d for d in drones if d.is_alive]
        for i, a in enumerate(alive):
            for b in alive[i + 1:]:
                dist = a.position.distance_to(b.position)
                if dist < COLLISION_R * 2:
                    push = (a.position - b.position).normalized() * (COLLISION_R * 2 - dist) * 0.5
                    a.position = a.position + push
                    b.position = b.position - push

    # ─── Sensor update ──────────────────────────────────────────────────────

    def _update_sensors(self, drone: Drone, all_drones: List[Drone]):
        drone.neighbors = []
        drone.visible_enemies = []
        for other in all_drones:
            if other.id == drone.id or not other.is_alive:
                continue
            dist = drone.position.distance_to(other.position)
            if dist > SENSOR_RANGE:
                continue
            if other.team == drone.team:
                drone.neighbors.append(other)
            else:
                drone.visible_enemies.append(other)

    # ─── Fire weapon ────────────────────────────────────────────────────────

    def _fire_weapon(self, drone: Drone, target: Drone):
        if drone.weapon_cooldown > 0:
            return
        drone.weapon_cooldown = WEAPON_COOLDOWN

        # Lead the target
        dist = drone.position.distance_to(target.position)
        proj_speed = 400.0
        lead_t = dist / proj_speed
        predicted = target.position + target.velocity * lead_t
        direction = (predicted - drone.position).normalized()

        proj = Projectile(
            id=str(uuid.uuid4())[:6],
            owner_id=drone.id,
            team=drone.team,
            position=Vec2(drone.position.x, drone.position.y),
            velocity=direction * proj_speed,
            damage=WEAPON_DAMAGE,
            lifetime=0.5,
        )
        self.projectiles[proj.id] = proj
        drone.damage_dealt += WEAPON_DAMAGE

    # ─── Main tick ──────────────────────────────────────────────────────────

    def tick_once(self, dt: float = DT):
        if self.paused:
            return

        all_drones = list(self.drones.values())
        alive = [d for d in all_drones if d.is_alive]

        # Update sensors
        for drone in alive:
            self._update_sensors(drone, all_drones)

        # Run AI behaviors
        from agents.behaviors.behavior_tree import run_behavior_tree
        for drone in alive:
            t0 = time.perf_counter()
            run_behavior_tree(drone, self)
            drone.inference_ms = (time.perf_counter() - t0) * 1000

        # Physics
        for drone in all_drones:
            self._integrate_drone(drone, dt)

        # Hazard effects
        for drone in alive:
            self.arena.apply_hazards_to_drone(drone, dt)

        # Projectiles
        self._update_projectiles(dt)

        # Collisions
        self._drone_collision(alive)

        # Arena hazards + control points
        self.arena.update(dt, alive)

        # Respawn dead drones after 5 seconds (simplified)
        for drone in all_drones:
            if (drone.state == DroneState.DEAD and
                    time.time() - drone.last_hit_at > 5.0 and
                    drone.health <= 0):
                drone.health = HEALTH_MAX
                drone.shield = SHIELD_MAX
                drone.state = DroneState.PATROLLING
                spawn_zones = {
                    TeamID.RED:   (50,  50,  200, 200),
                    TeamID.BLUE:  (950, 650, 1150, 850),
                    TeamID.GREEN: (50,  650, 200,  850),
                    TeamID.GOLD:  (950, 50,  1150, 200),
                }
                z = spawn_zones[drone.team]
                drone.position = Vec2(random.uniform(z[0], z[2]), random.uniform(z[1], z[3]))
                drone.velocity = Vec2()
                drone.battery_pct = 100.0

        self.tick += 1
        self.elapsed += dt

        # Record replay frame
        if self._record_replay and self.tick % self._replay_interval == 0:
            self.replay_frames.append(self.get_state_snapshot())
            if len(self.replay_frames) > 3600:  # ~60s at 6 FPS
                self.replay_frames = self.replay_frames[-3600:]

    def get_state_snapshot(self) -> dict:
        return {
            "tick": self.tick,
            "elapsed": round(self.elapsed, 2),
            "scores": dict(self.scores),
            "drones": [d.to_dict() for d in self.drones.values()],
            "projectiles": [
                {
                    "id": p.id,
                    "x": round(p.position.x, 1),
                    "y": round(p.position.y, 1),
                    "team": p.team.name,
                }
                for p in self.projectiles.values()
            ],
            "arena": self.arena.to_dict(),
            "events": self.events[-20:],
        }

    def get_leaderboard(self) -> List[dict]:
        rows = []
        for drone in self.drones.values():
            rows.append({
                "id": drone.id,
                "team": drone.team.name,
                "kills": drone.kills,
                "deaths": drone.hits_taken,
                "damage": round(drone.damage_dealt, 0),
                "alive": drone.is_alive,
            })
        return sorted(rows, key=lambda r: r["kills"], reverse=True)


# ─── Async runner ─────────────────────────────────────────────────────────────

async def run_simulation_loop(sim: Simulation, broadcast_fn: Optional[Callable] = None):
    """Run simulation at target FPS, calling broadcast_fn each tick."""
    sim.running = True
    frame_time = 1.0 / SIM_FPS
    while sim.running:
        t0 = time.perf_counter()
        sim.tick_once(DT)
        if broadcast_fn:
            await broadcast_fn(sim.get_state_snapshot())
        elapsed = time.perf_counter() - t0
        sleep_t = max(0.0, frame_time - elapsed)
        await asyncio.sleep(sleep_t)
