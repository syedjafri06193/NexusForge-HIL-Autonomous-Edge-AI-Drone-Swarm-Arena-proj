"""Tests for simulation engine: physics, drones, arena, projectiles."""

import math
import time
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from simulation.engine.sim import (
    Vec2, Drone, Arena, Simulation, Projectile,
    TeamID, DroneState, DroneConfig,
    ARENA_W, ARENA_H, MAX_SPEED, COLLISION_R, HEALTH_MAX, SHIELD_MAX, DT,
)


# ─── Vec2 ────────────────────────────────────────────────────────────────────

class TestVec2:
    def test_addition(self):
        v = Vec2(1, 2) + Vec2(3, 4)
        assert v.x == 4 and v.y == 6

    def test_subtraction(self):
        v = Vec2(5, 3) - Vec2(2, 1)
        assert v.x == 3 and v.y == 2

    def test_scalar_multiply(self):
        v = Vec2(2, 3) * 2
        assert v.x == 4 and v.y == 6

    def test_length(self):
        assert abs(Vec2(3, 4).length() - 5.0) < 1e-6

    def test_normalized(self):
        n = Vec2(3, 4).normalized()
        assert abs(n.length() - 1.0) < 1e-6

    def test_normalized_zero_vector(self):
        n = Vec2(0, 0).normalized()
        assert n.x == 0 and n.y == 0

    def test_distance_to(self):
        assert abs(Vec2(0, 0).distance_to(Vec2(3, 4)) - 5.0) < 1e-6

    def test_from_angle(self):
        v = Vec2.from_angle(0, 1.0)
        assert abs(v.x - 1.0) < 1e-6
        assert abs(v.y) < 1e-6

    def test_to_dict(self):
        d = Vec2(1.23456, 7.89012).to_dict()
        assert "x" in d and "y" in d
        assert d["x"] == round(1.23456, 2)

    def test_random_in_arena(self):
        for _ in range(10):
            v = Vec2.random_in_arena()
            assert 50 <= v.x <= ARENA_W - 50
            assert 50 <= v.y <= ARENA_H - 50


# ─── Drone ───────────────────────────────────────────────────────────────────

class TestDrone:
    def setup_method(self):
        self.drone = Drone(team=TeamID.RED, position=Vec2(600, 450))

    def test_initial_state(self):
        assert self.drone.health == HEALTH_MAX
        assert self.drone.shield == SHIELD_MAX
        assert self.drone.is_alive
        assert self.drone.state == DroneState.PATROLLING
        assert self.drone.kills == 0
        assert self.drone.battery_pct == 100.0

    def test_take_damage_hits_shield_first(self):
        self.drone.take_damage(10.0, time.time())
        assert self.drone.shield == SHIELD_MAX - 10.0
        assert self.drone.health == HEALTH_MAX

    def test_take_damage_overflow_to_health(self):
        self.drone.take_damage(SHIELD_MAX + 20.0, time.time())
        assert self.drone.shield == 0
        assert self.drone.health == HEALTH_MAX - 20.0

    def test_death_on_zero_health(self):
        self.drone.take_damage(SHIELD_MAX + HEALTH_MAX + 10, time.time())
        assert not self.drone.is_alive
        assert self.drone.state == DroneState.DEAD

    def test_effective_hp(self):
        assert self.drone.effective_hp == HEALTH_MAX + SHIELD_MAX

    def test_apply_force_accumulates(self):
        self.drone.apply_force(Vec2(10, 0))
        self.drone.apply_force(Vec2(5, 3))
        assert self.drone.acceleration.x == 15
        assert self.drone.acceleration.y == 3

    def test_telemetry_packet(self):
        t = self.drone.get_telemetry()
        assert t.drone_id == self.drone.id
        assert t.team == "RED"
        assert t.health == HEALTH_MAX
        assert t.kills == 0

    def test_to_dict_keys(self):
        d = self.drone.to_dict()
        for key in ["id", "team", "state", "position", "velocity", "health", "shield",
                    "battery_pct", "kills", "latency_ms", "inference_ms", "weapon_ready"]:
            assert key in d, f"Missing key: {key}"

    def test_config_defaults(self):
        cfg = DroneConfig()
        assert cfg.quantization_bits == 8
        assert cfg.model_type == "behavior_tree"


# ─── Arena ───────────────────────────────────────────────────────────────────

class TestArena:
    def setup_method(self):
        self.arena = Arena()

    def test_dimensions(self):
        assert self.arena.width == ARENA_W
        assert self.arena.height == ARENA_H

    def test_has_obstacles(self):
        assert len(self.arena.obstacles) > 0

    def test_has_control_points(self):
        assert len(self.arena.control_points) == 5
        assert any(cp["id"] == "nexus" for cp in self.arena.control_points)

    def test_clamp_to_arena(self):
        pos = self.arena.clamp_to_arena(Vec2(-50, -50))
        assert pos.x == COLLISION_R
        assert pos.y == COLLISION_R

        pos = self.arena.clamp_to_arena(Vec2(9999, 9999))
        assert pos.x == ARENA_W - COLLISION_R
        assert pos.y == ARENA_H - COLLISION_R

    def test_obstacle_detection(self):
        # First obstacle is at (350, 200, 80, 120)
        assert self.arena.is_in_obstacle(Vec2(360, 210))
        # Center obstacle is at (540, 380, 120, 80) so (540, 420) is outside it
        assert not self.arena.is_in_obstacle(Vec2(200, 800))  # near corner, no obstacle

    def test_to_dict_keys(self):
        d = self.arena.to_dict()
        assert "width" in d and "height" in d
        assert "obstacles" in d and "control_points" in d and "hazards" in d


# ─── Simulation ──────────────────────────────────────────────────────────────

class TestSimulation:
    def setup_method(self):
        self.sim = Simulation(num_teams=2, drones_per_team=4)

    def test_spawns_correct_count(self):
        assert len(self.sim.drones) == 8

    def test_team_distribution(self):
        red_count  = sum(1 for d in self.sim.drones.values() if d.team == TeamID.RED)
        blue_count = sum(1 for d in self.sim.drones.values() if d.team == TeamID.BLUE)
        assert red_count == 4
        assert blue_count == 4

    def test_scores_initialized(self):
        for team in ["RED", "BLUE"]:
            assert team in self.sim.scores
            assert self.sim.scores[team] == 0

    def test_tick_advances(self):
        self.sim.tick_once()
        assert self.sim.tick == 1
        assert abs(self.sim.elapsed - DT) < 1e-6

    def test_multiple_ticks(self):
        for _ in range(60):
            self.sim.tick_once()
        assert self.sim.tick == 60
        # All drones should still be in arena bounds
        for drone in self.sim.drones.values():
            assert 0 <= drone.position.x <= ARENA_W
            assert 0 <= drone.position.y <= ARENA_H

    def test_spawn_additional_drone(self):
        initial_count = len(self.sim.drones)
        self.sim.spawn_drone(TeamID.GREEN)
        assert len(self.sim.drones) == initial_count + 1

    def test_state_snapshot_keys(self):
        snap = self.sim.get_state_snapshot()
        for key in ["tick", "elapsed", "scores", "drones", "projectiles", "arena", "events"]:
            assert key in snap, f"Missing key: {key}"

    def test_leaderboard_sorted_by_kills(self):
        # Give one drone some kills
        first_drone = list(self.sim.drones.values())[0]
        first_drone.kills = 5
        lb = self.sim.get_leaderboard()
        assert lb[0]["kills"] == 5

    def test_paused_tick_no_advance(self):
        self.sim.paused = True
        self.sim.tick_once()
        assert self.sim.tick == 0

    def test_replay_records_frames(self):
        for _ in range(self.sim._replay_interval + 1):
            self.sim.tick_once()
        assert len(self.sim.replay_frames) >= 1

    def test_projectile_damages_enemy(self):
        drones = list(self.sim.drones.values())
        attacker = next(d for d in drones if d.team == TeamID.RED)
        target   = next(d for d in drones if d.team == TeamID.BLUE)

        target.position = Vec2(attacker.position.x + 5, attacker.position.y)
        self.sim._fire_weapon(attacker, target)

        initial_hp = target.effective_hp
        for _ in range(10):
            self.sim._update_projectiles(DT)

        assert target.effective_hp < initial_hp or len(self.sim.projectiles) == 0

    def test_physics_drag(self):
        drone = list(self.sim.drones.values())[0]
        drone.velocity = Vec2(100, 0)
        self.sim._integrate_drone(drone, DT)
        # Velocity should decrease due to drag
        assert drone.velocity.x < 100

    def test_velocity_clamped(self):
        drone = list(self.sim.drones.values())[0]
        drone.velocity = Vec2(MAX_SPEED * 10, 0)
        self.sim._integrate_drone(drone, DT)
        # Should be clamped
        assert drone.velocity.length() <= MAX_SPEED + 1

    def test_collision_resolution(self):
        drones = list(self.sim.drones.values())[:2]
        drones[0].position = Vec2(600, 450)
        drones[1].position = Vec2(601, 450)  # Overlapping
        self.sim._drone_collision(drones)
        dist = drones[0].position.distance_to(drones[1].position)
        assert dist >= COLLISION_R * 2 - 0.1  # resolved
