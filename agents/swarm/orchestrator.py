"""
NexusForge Swarm Orchestrator
Higher-level swarm coordination: formation control, mission planning,
natural-language command parsing, and RL-based policy routing.
"""

import asyncio
import math
import random
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

from simulation.engine.sim import (
    Drone, Simulation, Vec2, TeamID, DroneState,
    ARENA_W, ARENA_H, MAX_SPEED,
)


# ─── Mission types ────────────────────────────────────────────────────────────

class MissionType(Enum):
    ATTACK       = "attack"
    DEFEND       = "defend"
    CAPTURE      = "capture"
    FLANK        = "flank"
    REGROUP      = "regroup"
    SCATTER      = "scatter"
    SURROUND     = "surround"
    PATROL       = "patrol"
    ESCORT       = "escort"
    KAMIKAZE     = "kamikaze"


# ─── Formation shapes ─────────────────────────────────────────────────────────

class Formation(Enum):
    WEDGE       = "wedge"
    LINE        = "line"
    CIRCLE      = "circle"
    SPREAD      = "spread"
    DIAMOND     = "diamond"
    COLUMN      = "column"


def compute_formation_positions(
    center: Vec2,
    heading: float,
    n: int,
    formation: Formation,
    spacing: float = 45.0,
) -> List[Vec2]:
    """Compute target positions for n drones in a given formation around center."""
    positions = []

    if formation == Formation.WEDGE:
        for i in range(n):
            row = i // 3
            col = i % 3 - 1
            angle = heading + math.pi  # behind center
            offset = Vec2(
                math.cos(angle) * row * spacing + math.cos(heading + math.pi / 2) * col * spacing,
                math.sin(angle) * row * spacing + math.sin(heading + math.pi / 2) * col * spacing,
            )
            positions.append(center + offset)

    elif formation == Formation.LINE:
        perp = heading + math.pi / 2
        start = -(n - 1) / 2 * spacing
        for i in range(n):
            offset = Vec2(
                math.cos(perp) * (start + i * spacing),
                math.sin(perp) * (start + i * spacing),
            )
            positions.append(center + offset)

    elif formation == Formation.CIRCLE:
        for i in range(n):
            angle = (math.tau / n) * i
            r = spacing * max(1, n / 6)
            positions.append(Vec2(
                center.x + math.cos(angle) * r,
                center.y + math.sin(angle) * r,
            ))

    elif formation == Formation.DIAMOND:
        offsets = [(0, -spacing), (spacing, 0), (0, spacing), (-spacing, 0)]
        for i in range(n):
            ox, oy = offsets[i % 4]
            ring = i // 4
            positions.append(Vec2(
                center.x + ox * (1 + ring * 0.5),
                center.y + oy * (1 + ring * 0.5),
            ))

    elif formation == Formation.SPREAD:
        for i in range(n):
            angle = (math.tau / n) * i
            r = random.uniform(spacing * 0.5, spacing * 2.0)
            positions.append(Vec2(
                center.x + math.cos(angle) * r,
                center.y + math.sin(angle) * r,
            ))

    elif formation == Formation.COLUMN:
        for i in range(n):
            offset = Vec2(
                math.cos(heading + math.pi) * i * spacing,
                math.sin(heading + math.pi) * i * spacing,
            )
            positions.append(center + offset)

    else:
        positions = [Vec2(
            center.x + random.uniform(-spacing, spacing),
            center.y + random.uniform(-spacing, spacing),
        ) for _ in range(n)]

    return positions


# ─── Swarm command ────────────────────────────────────────────────────────────

@dataclass
class SwarmCommand:
    team: TeamID
    mission: MissionType
    formation: Optional[Formation] = None
    target_position: Optional[Vec2] = None
    target_team: Optional[TeamID] = None
    priority: int = 5             # 1 (low) to 10 (critical)
    issued_at: float = field(default_factory=time.time)
    source: str = "operator"      # "operator" | "ai" | "nlp"
    raw_text: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "team": self.team.name,
            "mission": self.mission.value,
            "formation": self.formation.value if self.formation else None,
            "target_position": self.target_position.to_dict() if self.target_position else None,
            "target_team": self.target_team.name if self.target_team else None,
            "priority": self.priority,
            "source": self.source,
            "raw_text": self.raw_text,
            "issued_at": self.issued_at,
        }


# ─── NLP command parser ───────────────────────────────────────────────────────

# Intent rules: (keywords) -> (mission, formation, priority)
_INTENT_RULES = [
    (["attack", "assault", "destroy", "eliminate", "engage"],           MissionType.ATTACK,  None,               7),
    (["defend", "protect", "hold", "guard"],                            MissionType.DEFEND,  Formation.CIRCLE,   8),
    (["capture", "take", "seize", "control", "occupy"],                 MissionType.CAPTURE, None,               7),
    (["flank", "surround", "encircle", "pincer"],                       MissionType.FLANK,   Formation.SPREAD,   8),
    (["regroup", "retreat", "fall back", "withdraw", "rally"],          MissionType.REGROUP, Formation.COLUMN,   9),
    (["scatter", "disperse", "spread out", "fan out"],                  MissionType.SCATTER, Formation.SPREAD,   6),
    (["kamikaze", "suicide", "self-destruct", "sacrifice"],             MissionType.KAMIKAZE,None,               10),
    (["patrol", "sweep", "scout", "reconnoiter"],                       MissionType.PATROL,  Formation.LINE,     4),
    (["wedge", "arrow", "spear"],                                       MissionType.ATTACK,  Formation.WEDGE,    7),
    (["diamond", "diamond formation"],                                  MissionType.ATTACK,  Formation.DIAMOND,  6),
    (["surround", "encircle"],                                          MissionType.SURROUND,Formation.CIRCLE,   8),
]

_TEAM_KEYWORDS = {
    "red": TeamID.RED, "blue": TeamID.BLUE, "green": TeamID.GREEN, "gold": TeamID.GOLD,
    "alpha": TeamID.RED, "bravo": TeamID.BLUE,
}

_LOCATION_KEYWORDS = {
    "center": Vec2(ARENA_W / 2, ARENA_H / 2),
    "north":  Vec2(ARENA_W / 2, 80),
    "south":  Vec2(ARENA_W / 2, ARENA_H - 80),
    "east":   Vec2(ARENA_W - 80, ARENA_H / 2),
    "west":   Vec2(80, ARENA_H / 2),
    "nexus":  Vec2(600, 450),
    "alpha point":  Vec2(200, 450),
    "beta point":   Vec2(600, 200),
    "gamma point":  Vec2(600, 700),
    "delta point":  Vec2(1000, 450),
}


def parse_nlp_command(text: str, issuing_team: TeamID) -> Optional[SwarmCommand]:
    """
    Parse a natural-language command string into a SwarmCommand.
    E.g. "Red team, attack the center in wedge formation"
         "All drones, scatter and regroup at nexus"
         "Flank the blue team from the east"
    """
    t = text.lower().strip()

    # Detect team override
    team = issuing_team
    for kw, tid in _TEAM_KEYWORDS.items():
        if kw in t:
            team = tid
            break

    # Detect intent
    mission = MissionType.PATROL
    formation = None
    priority = 5

    for keywords, m, f, p in _INTENT_RULES:
        if any(kw in t for kw in keywords):
            mission = m
            if f and formation is None:
                formation = f
            priority = p
            break

    # Detect target team
    target_team = None
    for kw, tid in _TEAM_KEYWORDS.items():
        if kw in t and tid != team:
            target_team = tid
            break

    # Detect location
    target_pos = None
    for kw, pos in _LOCATION_KEYWORDS.items():
        if kw in t:
            target_pos = pos
            break

    # Detect explicit formation
    for f in Formation:
        if f.value in t:
            formation = f
            break

    return SwarmCommand(
        team=team,
        mission=mission,
        formation=formation,
        target_position=target_pos,
        target_team=target_team,
        priority=priority,
        source="nlp",
        raw_text=text,
    )


# ─── Tactical planner ─────────────────────────────────────────────────────────

class TacticalPlanner:
    """
    Higher-level tactical reasoning per team.
    Runs every N ticks and issues SwarmCommands automatically.
    """

    def __init__(self, team: TeamID):
        self.team = team
        self.current_mission = MissionType.PATROL
        self.current_formation = Formation.WEDGE
        self.formation_waypoint: Optional[Vec2] = None
        self._eval_interval = 2.0   # seconds between re-evaluation
        self._last_eval = 0.0

    def evaluate(self, sim: Simulation) -> Optional[SwarmCommand]:
        now = time.time()
        if now - self._last_eval < self._eval_interval:
            return None
        self._last_eval = now

        my_drones = [d for d in sim.drones.values() if d.team == self.team and d.is_alive]
        if not my_drones:
            return None

        # Gather intel
        enemies = [d for d in sim.drones.values() if d.team != self.team and d.is_alive]
        avg_health = sum(d.health for d in my_drones) / len(my_drones)
        my_score = sim.scores.get(self.team.name, 0)
        enemy_scores = {k: v for k, v in sim.scores.items() if k != self.team.name}
        losing = my_score < max(enemy_scores.values(), default=0)

        # Decision logic
        if avg_health < 30:
            return SwarmCommand(team=self.team, mission=MissionType.REGROUP,
                                formation=Formation.CIRCLE, priority=9, source="ai")

        if len(my_drones) < 3:
            return SwarmCommand(team=self.team, mission=MissionType.DEFEND,
                                formation=Formation.CIRCLE, priority=8, source="ai")

        if len(enemies) == 0:
            return SwarmCommand(team=self.team, mission=MissionType.CAPTURE,
                                formation=Formation.SPREAD, priority=5, source="ai")

        if len(my_drones) > len(enemies) * 1.5:
            # Outnumber enemy — surround
            if enemies:
                cx = sum(e.position.x for e in enemies) / len(enemies)
                cy = sum(e.position.y for e in enemies) / len(enemies)
                return SwarmCommand(team=self.team, mission=MissionType.SURROUND,
                                    target_position=Vec2(cx, cy),
                                    formation=Formation.CIRCLE, priority=8, source="ai")

        if losing:
            return SwarmCommand(team=self.team, mission=MissionType.ATTACK,
                                formation=Formation.WEDGE, priority=7, source="ai")

        return SwarmCommand(team=self.team, mission=MissionType.ATTACK,
                            formation=Formation.WEDGE, priority=6, source="ai")

    def apply_command(self, cmd: SwarmCommand, sim: Simulation):
        """Execute a SwarmCommand on the team's drones."""
        my_drones = [d for d in sim.drones.values() if d.team == self.team and d.is_alive]
        if not my_drones:
            return

        self.current_mission = cmd.mission
        if cmd.formation:
            self.current_formation = cmd.formation

        # Choose formation center
        center = cmd.target_position
        if center is None:
            if cmd.mission in (MissionType.ATTACK, MissionType.FLANK, MissionType.SURROUND):
                enemies = [d for d in sim.drones.values() if d.team != self.team and d.is_alive]
                if enemies:
                    target = min(enemies, key=lambda e: sum(
                        d.position.distance_to(e.position) for d in my_drones
                    ))
                    center = target.position
            if center is None:
                # Average drone position
                center = Vec2(
                    sum(d.position.x for d in my_drones) / len(my_drones),
                    sum(d.position.y for d in my_drones) / len(my_drones),
                )

        self.formation_waypoint = center

        # Compute target slots and assign
        heading = 0.0
        if cmd.mission in (MissionType.ATTACK, MissionType.CAPTURE):
            # Face toward target
            my_cx = sum(d.position.x for d in my_drones) / len(my_drones)
            my_cy = sum(d.position.y for d in my_drones) / len(my_drones)
            dx, dy = center.x - my_cx, center.y - my_cy
            heading = math.atan2(dy, dx)

        positions = compute_formation_positions(
            center, heading, len(my_drones),
            cmd.formation or Formation.SPREAD,
        )

        # Assign nearest drone to each slot
        assigned = set()
        for slot_pos in positions:
            best = None
            best_dist = float('inf')
            for d in my_drones:
                if d.id in assigned:
                    continue
                dist = d.position.distance_to(slot_pos)
                if dist < best_dist:
                    best_dist = dist
                    best = d
            if best:
                best.waypoint = slot_pos
                assigned.add(best.id)
                # Override state toward mission
                if cmd.mission in (MissionType.REGROUP, MissionType.DEFEND):
                    best.state = DroneState.REGROUPING
                elif cmd.mission in (MissionType.SCATTER,):
                    best.waypoint = Vec2(
                        random.uniform(50, ARENA_W - 50),
                        random.uniform(50, ARENA_H - 50),
                    )


# ─── Swarm orchestrator ───────────────────────────────────────────────────────

class SwarmOrchestrator:
    """Top-level orchestrator managing all teams."""

    def __init__(self, sim: Simulation):
        self.sim = sim
        self.planners: Dict[TeamID, TacticalPlanner] = {
            team: TacticalPlanner(team) for team in TeamID
        }
        self.command_log: List[dict] = []
        self._auto_tactics = True

    def issue_command(self, cmd: SwarmCommand):
        """Issue a manual or NLP-parsed command to a team."""
        planner = self.planners.get(cmd.team)
        if planner:
            planner.apply_command(cmd, self.sim)
            self.command_log.append({
                **cmd.to_dict(),
                "applied_at": time.time(),
            })

    def issue_nlp_command(self, text: str, issuing_team: TeamID) -> Optional[SwarmCommand]:
        """Parse text → command → apply."""
        cmd = parse_nlp_command(text, issuing_team)
        if cmd:
            self.issue_command(cmd)
        return cmd

    def update(self):
        """Called each simulation tick for autonomous tactical AI."""
        if not self._auto_tactics:
            return
        for team, planner in self.planners.items():
            cmd = planner.evaluate(self.sim)
            if cmd:
                planner.apply_command(cmd, self.sim)

    def get_status(self) -> dict:
        return {
            "auto_tactics": self._auto_tactics,
            "teams": {
                team.name: {
                    "mission": planner.current_mission.value,
                    "formation": planner.current_formation.value,
                    "alive": sum(
                        1 for d in self.sim.drones.values()
                        if d.team == team and d.is_alive
                    ),
                }
                for team, planner in self.planners.items()
            },
            "recent_commands": self.command_log[-10:],
        }
