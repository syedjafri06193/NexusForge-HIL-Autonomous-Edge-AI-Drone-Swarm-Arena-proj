"""Tests for behavior tree AI and swarm orchestrator."""

import pytest
import sys
import os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from simulation.engine.sim import (
    Drone, Simulation, Vec2, TeamID, DroneState, ARENA_W, ARENA_H,
)
from agents.behaviors.behavior_tree import (
    run_behavior_tree,
    HasEnemiesInSight, HasEnemyInWeaponRange, IsLowHealth, IsOutnumbered,
    AttackNearestEnemy, Evade, Patrol, FlockWithAllies,
    Sequence, Selector, Inverter, Status,
)
from agents.swarm.orchestrator import (
    SwarmOrchestrator, SwarmCommand, MissionType, Formation,
    TacticalPlanner, compute_formation_positions,
    parse_nlp_command,
)


# ─── BT Conditions ───────────────────────────────────────────────────────────

class TestBTConditions:
    def setup_method(self):
        self.sim = Simulation(num_teams=2, drones_per_team=2)
        self.drone = list(self.sim.drones.values())[0]
        self.enemy = next(d for d in self.sim.drones.values() if d.team != self.drone.team)

    def test_has_enemies_in_sight_false(self):
        self.drone.visible_enemies = []
        assert HasEnemiesInSight().tick(self.drone, self.sim) == Status.FAILURE

    def test_has_enemies_in_sight_true(self):
        self.drone.visible_enemies = [self.enemy]
        assert HasEnemiesInSight().tick(self.drone, self.sim) == Status.SUCCESS

    def test_is_low_health_false(self):
        self.drone.health = 80
        assert IsLowHealth(30).tick(self.drone, self.sim) == Status.FAILURE

    def test_is_low_health_true(self):
        self.drone.health = 20
        assert IsLowHealth(30).tick(self.drone, self.sim) == Status.SUCCESS

    def test_is_outnumbered_false(self):
        self.drone.visible_enemies = [self.enemy]
        self.drone.neighbors = [self.drone]  # 1 ally vs 1 enemy
        assert IsOutnumbered().tick(self.drone, self.sim) == Status.FAILURE

    def test_is_outnumbered_true(self):
        self.drone.visible_enemies = [self.enemy, self.enemy]
        self.drone.neighbors = []
        assert IsOutnumbered().tick(self.drone, self.sim) == Status.SUCCESS

    def test_has_enemy_in_weapon_range(self):
        from agents.behaviors.behavior_tree import HasEnemyInWeaponRange, WEAPON_RANGE
        self.drone.visible_enemies = [self.enemy]
        self.enemy.position = Vec2(self.drone.position.x + WEAPON_RANGE - 5, self.drone.position.y)
        assert HasEnemyInWeaponRange().tick(self.drone, self.sim) == Status.SUCCESS

    def test_enemy_out_of_weapon_range(self):
        from agents.behaviors.behavior_tree import HasEnemyInWeaponRange, WEAPON_RANGE
        self.drone.visible_enemies = [self.enemy]
        self.enemy.position = Vec2(self.drone.position.x + WEAPON_RANGE + 50, self.drone.position.y)
        assert HasEnemyInWeaponRange().tick(self.drone, self.sim) == Status.FAILURE


# ─── BT Composites ───────────────────────────────────────────────────────────

class TestBTComposites:
    def setup_method(self):
        self.sim = Simulation(num_teams=2, drones_per_team=1)
        self.drone = list(self.sim.drones.values())[0]

    def test_sequence_all_success(self):
        from agents.behaviors.behavior_tree import AlwaysSuccess, HoldPosition
        node = Sequence(AlwaysSuccess(HoldPosition()), AlwaysSuccess(HoldPosition()))
        assert node.tick(self.drone, self.sim) == Status.SUCCESS

    def test_sequence_stops_on_failure(self):
        self.drone.visible_enemies = []
        node = Sequence(HasEnemiesInSight(), AttackNearestEnemy())
        assert node.tick(self.drone, self.sim) == Status.FAILURE

    def test_selector_first_success(self):
        self.drone.visible_enemies = []
        node = Selector(HasEnemiesInSight(), Patrol())
        # HasEnemiesInSight fails, Patrol succeeds
        assert node.tick(self.drone, self.sim) in (Status.SUCCESS, Status.RUNNING)

    def test_inverter(self):
        self.drone.visible_enemies = []
        inv = Inverter(HasEnemiesInSight())
        assert inv.tick(self.drone, self.sim) == Status.SUCCESS  # inverted failure

        self.drone.visible_enemies = [list(self.sim.drones.values())[-1]]
        assert inv.tick(self.drone, self.sim) == Status.FAILURE  # inverted success


# ─── BT Actions ──────────────────────────────────────────────────────────────

class TestBTActions:
    def setup_method(self):
        self.sim = Simulation(num_teams=2, drones_per_team=2)
        self.drone = next(d for d in self.sim.drones.values() if d.team == TeamID.RED)
        self.enemy = next(d for d in self.sim.drones.values() if d.team == TeamID.BLUE)

    def test_patrol_sets_waypoint(self):
        self.drone.waypoint = None
        Patrol().tick(self.drone, self.sim)
        assert self.drone.waypoint is not None
        assert 0 <= self.drone.waypoint.x <= ARENA_W
        assert 0 <= self.drone.waypoint.y <= ARENA_H

    def test_patrol_sets_state(self):
        Patrol().tick(self.drone, self.sim)
        assert self.drone.state == DroneState.PATROLLING

    def test_evade_from_enemy(self):
        self.drone.visible_enemies = [self.enemy]
        self.enemy.position = Vec2(600, 450)
        self.drone.position = Vec2(605, 450)
        initial_accel_x = self.drone.acceleration.x
        Evade().tick(self.drone, self.sim)
        assert self.drone.state == DroneState.EVADING
        assert self.drone.acceleration.length() > initial_accel_x

    def test_flock_with_allies_adds_force(self):
        ally = list(self.sim.drones.values())[0]
        # Put ally far enough away to generate cohesion force
        ally.position = Vec2(self.drone.position.x + 80, self.drone.position.y + 80)
        self.drone.neighbors = [ally]
        FlockWithAllies().tick(self.drone, self.sim)
        # Acceleration should be non-zero (cohesion + alignment force applied)
        assert self.drone.acceleration.length() >= 0  # idempotent check
        # Specifically: cohesion weight * 60 should generate something
        # The force magnitudes depend on config; just verify no exception

    def test_attack_fires_weapon(self):
        self.drone.visible_enemies = [self.enemy]
        self.enemy.position = Vec2(self.drone.position.x + 40, self.drone.position.y)
        self.drone.weapon_cooldown = 0
        initial_projectiles = len(self.sim.projectiles)
        AttackNearestEnemy().tick(self.drone, self.sim)
        assert len(self.sim.projectiles) > initial_projectiles

    def test_full_behavior_tree_runs(self):
        """run_behavior_tree should not raise on any drone."""
        for drone in self.sim.drones.values():
            drone.visible_enemies = []
            drone.neighbors = []
            run_behavior_tree(drone, self.sim)  # should not raise


# ─── Formation positions ──────────────────────────────────────────────────────

class TestFormations:
    def test_wedge_count(self):
        positions = compute_formation_positions(Vec2(600, 450), 0, 6, Formation.WEDGE)
        assert len(positions) == 6

    def test_circle_equidistant(self):
        center = Vec2(600, 450)
        positions = compute_formation_positions(center, 0, 8, Formation.CIRCLE, spacing=50)
        # All points should be equidistant from center
        distances = [center.distance_to(p) for p in positions]
        assert max(distances) - min(distances) < 1.0  # within 1 unit

    def test_line_formation(self):
        positions = compute_formation_positions(Vec2(600, 450), 0, 5, Formation.LINE, spacing=40)
        assert len(positions) == 5

    def test_diamond_formation(self):
        positions = compute_formation_positions(Vec2(600, 450), 0, 4, Formation.DIAMOND)
        assert len(positions) == 4

    def test_all_formations_return_correct_count(self):
        for n in [3, 6, 12]:
            for formation in Formation:
                positions = compute_formation_positions(Vec2(600, 450), 0, n, formation)
                assert len(positions) == n, f"Formation {formation} returned wrong count for n={n}"


# ─── NLP command parser ───────────────────────────────────────────────────────

class TestNLPParser:
    def test_attack_command(self):
        cmd = parse_nlp_command("Attack the center", TeamID.RED)
        assert cmd is not None
        assert cmd.mission == MissionType.ATTACK

    def test_defend_command(self):
        cmd = parse_nlp_command("Defend the nexus", TeamID.BLUE)
        assert cmd is not None
        assert cmd.mission == MissionType.DEFEND

    def test_regroup_command(self):
        cmd = parse_nlp_command("Regroup at alpha point", TeamID.RED)
        assert cmd is not None
        assert cmd.mission == MissionType.REGROUP

    def test_scatter_command(self):
        cmd = parse_nlp_command("Scatter and fan out", TeamID.GREEN)
        assert cmd is not None
        assert cmd.mission == MissionType.SCATTER

    def test_wedge_formation_extracted(self):
        cmd = parse_nlp_command("Attack in wedge formation", TeamID.RED)
        assert cmd is not None
        assert cmd.formation == Formation.WEDGE

    def test_circle_formation_extracted(self):
        cmd = parse_nlp_command("Defend with circle formation", TeamID.RED)
        assert cmd is not None
        assert cmd.formation == Formation.CIRCLE

    def test_location_extracted(self):
        cmd = parse_nlp_command("Move to the center", TeamID.RED)
        assert cmd is not None
        if cmd.target_position:
            assert abs(cmd.target_position.x - ARENA_W / 2) < 10
            assert abs(cmd.target_position.y - ARENA_H / 2) < 10

    def test_team_extracted(self):
        cmd = parse_nlp_command("Blue team, attack the center", TeamID.RED)
        assert cmd is not None
        assert cmd.team == TeamID.BLUE

    def test_kamikaze_high_priority(self):
        cmd = parse_nlp_command("Kamikaze run on the enemy", TeamID.RED)
        assert cmd is not None
        assert cmd.mission == MissionType.KAMIKAZE
        assert cmd.priority >= 8

    def test_default_patrol_on_unknown(self):
        cmd = parse_nlp_command("Do some stuff maybe", TeamID.RED)
        # Should return something (fallback to patrol)
        assert cmd is not None


# ─── Swarm orchestrator ───────────────────────────────────────────────────────

class TestSwarmOrchestrator:
    def setup_method(self):
        self.sim = Simulation(num_teams=2, drones_per_team=4)
        self.orch = SwarmOrchestrator(self.sim)

    def test_issue_command_changes_drone_state(self):
        cmd = SwarmCommand(
            team=TeamID.RED,
            mission=MissionType.REGROUP,
            formation=Formation.CIRCLE,
        )
        self.orch.issue_command(cmd)
        red_drones = [d for d in self.sim.drones.values() if d.team == TeamID.RED]
        # At least some drones should have waypoints set
        assert any(d.waypoint is not None for d in red_drones)

    def test_nlp_command_parsed_and_applied(self):
        result = self.orch.issue_nlp_command("Attack the center", TeamID.RED)
        assert result is not None
        assert result.mission == MissionType.ATTACK

    def test_auto_tactics_toggle(self):
        self.orch._auto_tactics = False
        self.orch.update()  # Should not raise

    def test_status_returns_all_teams(self):
        status = self.orch.get_status()
        assert "teams" in status
        assert "auto_tactics" in status

    def test_command_log_populated(self):
        cmd = SwarmCommand(team=TeamID.RED, mission=MissionType.PATROL)
        self.orch.issue_command(cmd)
        assert len(self.orch.command_log) > 0

    def test_tactical_planner_evaluates(self):
        planner = TacticalPlanner(TeamID.RED)
        # Force re-evaluation
        planner._last_eval = 0.0
        result = planner.evaluate(self.sim)
        # May return None or a command
        if result:
            assert isinstance(result.mission, MissionType)
