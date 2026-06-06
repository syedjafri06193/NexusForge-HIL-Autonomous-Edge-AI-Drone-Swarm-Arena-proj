"""
Behavior Tree AI for NexusForge drones.
Each drone runs its own behavior tree every tick, simulating
edge-AI decision making under latency and power constraints.
"""

import math
import random
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation.engine.sim import Drone, Simulation, Vec2

from simulation.engine.sim import (
    DroneState, Vec2, MAX_SPEED, WEAPON_RANGE, SENSOR_RANGE,
    ARENA_W, ARENA_H, DroneConfig,
)


# ─── BT return types ─────────────────────────────────────────────────────────

class Status:
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"


# ─── Core node types ─────────────────────────────────────────────────────────

class BTNode:
    def tick(self, drone, sim) -> str:
        raise NotImplementedError

class Sequence(BTNode):
    """All children must succeed"""
    def __init__(self, *children):
        self.children = children
    def tick(self, drone, sim) -> str:
        for child in self.children:
            result = child.tick(drone, sim)
            if result != Status.SUCCESS:
                return result
        return Status.SUCCESS

class Selector(BTNode):
    """First successful child wins"""
    def __init__(self, *children):
        self.children = children
    def tick(self, drone, sim) -> str:
        for child in self.children:
            result = child.tick(drone, sim)
            if result != Status.FAILURE:
                return result
        return Status.FAILURE

class Inverter(BTNode):
    def __init__(self, child):
        self.child = child
    def tick(self, drone, sim) -> str:
        result = self.child.tick(drone, sim)
        if result == Status.SUCCESS:
            return Status.FAILURE
        if result == Status.FAILURE:
            return Status.SUCCESS
        return Status.RUNNING

class AlwaysSuccess(BTNode):
    def __init__(self, child):
        self.child = child
    def tick(self, drone, sim) -> str:
        self.child.tick(drone, sim)
        return Status.SUCCESS


# ─── Condition nodes ─────────────────────────────────────────────────────────

class HasEnemiesInSight(BTNode):
    def tick(self, drone, sim) -> str:
        return Status.SUCCESS if drone.visible_enemies else Status.FAILURE

class HasEnemyInWeaponRange(BTNode):
    def tick(self, drone, sim) -> str:
        for enemy in drone.visible_enemies:
            if drone.position.distance_to(enemy.position) <= WEAPON_RANGE:
                return Status.SUCCESS
        return Status.FAILURE

class IsLowHealth(BTNode):
    def __init__(self, threshold: float = 30.0):
        self.threshold = threshold
    def tick(self, drone, sim) -> str:
        return Status.SUCCESS if drone.health < self.threshold else Status.FAILURE

class IsLowBattery(BTNode):
    def __init__(self, threshold: float = 15.0):
        self.threshold = threshold
    def tick(self, drone, sim) -> str:
        return Status.SUCCESS if drone.battery_pct < self.threshold else Status.FAILURE

class HasAllyNearby(BTNode):
    def tick(self, drone, sim) -> str:
        return Status.SUCCESS if drone.neighbors else Status.FAILURE

class IsOutnumbered(BTNode):
    def tick(self, drone, sim) -> str:
        return Status.SUCCESS if len(drone.visible_enemies) > len(drone.neighbors) + 1 else Status.FAILURE

class WeaponReady(BTNode):
    def tick(self, drone, sim) -> str:
        return Status.SUCCESS if drone.weapon_cooldown <= 0 else Status.FAILURE

class IsStunned(BTNode):
    def tick(self, drone, sim) -> str:
        return Status.SUCCESS if drone.stun_timer > 0 else Status.FAILURE


# ─── Action nodes ─────────────────────────────────────────────────────────────

class AttackNearestEnemy(BTNode):
    def tick(self, drone, sim) -> str:
        if not drone.visible_enemies:
            return Status.FAILURE
        target = min(drone.visible_enemies, key=lambda e: drone.position.distance_to(e.position))
        drone.target_id = target.id
        drone.state = DroneState.ATTACKING

        dist = drone.position.distance_to(target.position)
        if dist <= WEAPON_RANGE and drone.weapon_cooldown <= 0:
            sim._fire_weapon(drone, target)
        elif dist > WEAPON_RANGE:
            # Move toward target
            direction = (target.position - drone.position).normalized()
            desired_velocity = direction * MAX_SPEED * drone.config.aggression
            steering = desired_velocity - drone.velocity
            drone.apply_force(steering * 4.0)
        return Status.RUNNING

class PursueTarget(BTNode):
    def tick(self, drone, sim) -> str:
        target = sim.drones.get(drone.target_id) if drone.target_id else None
        if not target or not target.is_alive:
            target = min(drone.visible_enemies, key=lambda e: drone.position.distance_to(e.position)) \
                if drone.visible_enemies else None
        if not target:
            return Status.FAILURE

        drone.target_id = target.id
        drone.state = DroneState.PURSUING
        direction = (target.position - drone.position).normalized()
        desired = direction * MAX_SPEED
        drone.apply_force((desired - drone.velocity) * 3.5)
        return Status.RUNNING

class Evade(BTNode):
    """Flee from nearest threat"""
    def tick(self, drone, sim) -> str:
        if not drone.visible_enemies:
            return Status.FAILURE
        drone.state = DroneState.EVADING
        nearest = min(drone.visible_enemies, key=lambda e: drone.position.distance_to(e.position))
        away = (drone.position - nearest.position).normalized()
        # Add random jitter to avoid lock-step
        jitter = Vec2(random.gauss(0, 0.3), random.gauss(0, 0.3))
        desired = (away + jitter).normalized() * MAX_SPEED
        drone.apply_force((desired - drone.velocity) * 5.0)
        return Status.RUNNING

class Regroup(BTNode):
    """Move toward team centroid"""
    def tick(self, drone, sim) -> str:
        if not drone.neighbors:
            return Status.FAILURE
        drone.state = DroneState.REGROUPING
        cx = sum(n.position.x for n in drone.neighbors) / len(drone.neighbors)
        cy = sum(n.position.y for n in drone.neighbors) / len(drone.neighbors)
        centroid = Vec2(cx, cy)
        direction = (centroid - drone.position).normalized()
        drone.apply_force((direction * MAX_SPEED * 0.7 - drone.velocity) * 2.0)
        return Status.RUNNING

class FlockWithAllies(BTNode):
    """Reynolds boids: separation + cohesion + alignment"""
    def tick(self, drone, sim) -> str:
        if not drone.neighbors:
            return Status.FAILURE

        cfg = drone.config
        sep = Vec2()
        coh = Vec2()
        ali = Vec2()

        for n in drone.neighbors:
            diff = drone.position - n.position
            dist = diff.length()
            if dist < 30 and dist > 0:
                sep = sep + diff.normalized() * (30 / dist)
            coh = coh + n.position
            ali = ali + n.velocity

        if drone.neighbors:
            coh = (coh / len(drone.neighbors) - drone.position).normalized()
            ali = (ali / len(drone.neighbors)).normalized()

        force = sep * cfg.separation + coh * cfg.cohesion + ali * cfg.alignment
        drone.apply_force(force * 60)
        return Status.SUCCESS

class Patrol(BTNode):
    """Random waypoint patrol"""
    def tick(self, drone, sim) -> str:
        drone.state = DroneState.PATROLLING
        if drone.waypoint is None or drone.position.distance_to(drone.waypoint) < 20:
            drone.waypoint = Vec2(
                random.uniform(60, ARENA_W - 60),
                random.uniform(60, ARENA_H - 60),
            )
        direction = (drone.waypoint - drone.position).normalized()
        desired = direction * MAX_SPEED * 0.5
        drone.apply_force((desired - drone.velocity) * 2.0)
        return Status.RUNNING

class HoldPosition(BTNode):
    def tick(self, drone, sim) -> str:
        drone.apply_force(drone.velocity * -3.0)
        return Status.SUCCESS

class CaptureNearestControlPoint(BTNode):
    def tick(self, drone, sim) -> str:
        cps = sim.arena.control_points
        unowned = [cp for cp in cps if cp["owner"] != drone.team.name]
        if not unowned:
            return Status.FAILURE
        nearest = min(unowned, key=lambda cp: drone.position.distance_to(Vec2(cp["x"], cp["y"])))
        target_pos = Vec2(nearest["x"], nearest["y"])
        dist = drone.position.distance_to(target_pos)
        if dist > 5:
            direction = (target_pos - drone.position).normalized()
            desired = direction * MAX_SPEED * 0.6
            drone.apply_force((desired - drone.velocity) * 2.5)
        return Status.RUNNING


# ─── Swarm tactics ───────────────────────────────────────────────────────────

class PinceMovement(BTNode):
    """Coordinate flanking: move to opposite side of target from allies"""
    def tick(self, drone, sim) -> str:
        if not drone.visible_enemies or not drone.neighbors:
            return Status.FAILURE
        target = drone.visible_enemies[0]
        # Find avg ally position
        ally_cx = sum(n.position.x for n in drone.neighbors) / len(drone.neighbors)
        ally_cy = sum(n.position.y for n in drone.neighbors) / len(drone.neighbors)
        ally_center = Vec2(ally_cx, ally_cy)
        # Go to opposite side
        to_target = (target.position - ally_center).normalized()
        flank_pos = target.position + to_target * 60
        direction = (flank_pos - drone.position).normalized()
        drone.apply_force((direction * MAX_SPEED - drone.velocity) * 3.0)
        return Status.RUNNING


# ─── Build behavior trees ─────────────────────────────────────────────────────

def _build_aggressive_tree() -> BTNode:
    return Selector(
        # If stunned, hold
        Sequence(IsStunned(), HoldPosition()),
        # If low health, evade and regroup
        Sequence(IsLowHealth(25), Selector(Evade(), Regroup())),
        # If outnumbered and low health, evade
        Sequence(IsLowHealth(50), IsOutnumbered(), Evade()),
        # If enemy in weapon range, attack
        Sequence(HasEnemiesInSight(), WeaponReady(), AttackNearestEnemy()),
        # If enemies visible, pursue
        Sequence(HasEnemiesInSight(), PursueTarget()),
        # Swarm tactics with allies
        Sequence(HasAllyNearby(), AlwaysSuccess(FlockWithAllies()), CaptureNearestControlPoint()),
        # Default: patrol
        AlwaysSuccess(Patrol()),
    )


def _build_defensive_tree() -> BTNode:
    return Selector(
        Sequence(IsStunned(), HoldPosition()),
        Sequence(IsLowBattery(10), HoldPosition()),
        Sequence(IsLowHealth(40), Regroup()),
        Sequence(HasEnemyInWeaponRange(), WeaponReady(), AttackNearestEnemy()),
        Sequence(HasAllyNearby(), FlockWithAllies()),
        Sequence(HasEnemiesInSight(), PursueTarget()),
        CaptureNearestControlPoint(),
        Patrol(),
    )


def _build_flanker_tree() -> BTNode:
    return Selector(
        Sequence(IsStunned(), HoldPosition()),
        Sequence(IsLowHealth(20), Evade()),
        Sequence(HasEnemiesInSight(), HasAllyNearby(), PinceMovement()),
        Sequence(HasEnemiesInSight(), WeaponReady(), AttackNearestEnemy()),
        Sequence(HasEnemiesInSight(), PursueTarget()),
        CaptureNearestControlPoint(),
        Patrol(),
    )


# Cache trees by model type
_TREES = {
    "aggressive": _build_aggressive_tree(),
    "defensive":  _build_defensive_tree(),
    "flanker":    _build_flanker_tree(),
    "behavior_tree": _build_aggressive_tree(),  # default
}


def run_behavior_tree(drone: 'Drone', sim: 'Simulation'):
    """Entry point called by simulation each tick for each drone."""
    if not drone.is_alive:
        return
    model = drone.config.model_type
    tree = _TREES.get(model, _TREES["behavior_tree"])
    # Simulate quantization latency: lower bit width = faster but noisier
    if drone.config.quantization_bits < 8:
        # 4-bit: adds noise to decisions
        if random.random() < 0.05:
            tree = _TREES["defensive"]   # occasionally confused
    tree.tick(drone, sim)
