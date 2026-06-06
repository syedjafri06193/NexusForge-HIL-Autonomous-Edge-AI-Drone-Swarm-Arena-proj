"""Shared pytest fixtures for NexusForge test suite."""
import sys
import os
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.engine.sim import Simulation, TeamID, Vec2


@pytest.fixture
def sim_2v2():
    """2-team, 4-drone-per-team simulation."""
    return Simulation(num_teams=2, drones_per_team=4)

@pytest.fixture
def sim_4v4():
    """4-team, 4-drone-per-team simulation."""
    return Simulation(num_teams=4, drones_per_team=4)

@pytest.fixture
def red_drone(sim_2v2):
    return next(d for d in sim_2v2.drones.values() if d.team == TeamID.RED)

@pytest.fixture
def blue_drone(sim_2v2):
    return next(d for d in sim_2v2.drones.values() if d.team == TeamID.BLUE)

@pytest.fixture
def center():
    return Vec2(600, 450)
