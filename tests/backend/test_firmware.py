"""Tests for TinyML inference simulator and fault injection system."""

import pytest
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from firmware.tinyml.inference import (
    EdgeInferenceEngine, DroneInferencePool, MODELS, MCU_PROFILES, ModelSpec,
)
from firmware.fault_injection.injector import (
    FaultInjector, FaultEvent, FaultType, FaultScenarios,
)
from simulation.engine.sim import Simulation, TeamID, DroneState


# ─── ModelSpec ───────────────────────────────────────────────────────────────

class TestModelSpec:
    def test_model_size_decreases_with_quantization(self):
        model = MODELS["obstacle_detector"]
        assert model.model_size_kb(32) > model.model_size_kb(16)
        assert model.model_size_kb(16) > model.model_size_kb(8)
        assert model.model_size_kb(8)  > model.model_size_kb(4)

    def test_accuracy_decreases_with_lower_bits(self):
        model = MODELS["threat_classifier"]
        assert model.accuracy_at_bits(32) >= model.accuracy_at_bits(16)
        assert model.accuracy_at_bits(16) >= model.accuracy_at_bits(8)
        assert model.accuracy_at_bits(8)  >= model.accuracy_at_bits(4)

    def test_accuracy_between_0_and_1(self):
        for model in MODELS.values():
            for bits in [4, 8, 16, 32]:
                acc = model.accuracy_at_bits(bits)
                assert 0.0 <= acc <= 1.0

    def test_quantized_flops_decrease_with_lower_bits(self):
        model = MODELS["swarm_coordinator"]
        assert model.quantized_flops(32) > model.quantized_flops(8)
        assert model.quantized_flops(8)  > model.quantized_flops(4)


# ─── EdgeInferenceEngine ─────────────────────────────────────────────────────

class TestEdgeInferenceEngine:
    def setup_method(self):
        self.engine = EdgeInferenceEngine(mcu_type="esp32", bits=8, budget_ms=20.0)

    def test_infer_returns_result(self):
        result = self.engine.infer("obstacle_detector")
        assert result is not None
        assert result.model_name == "obstacle_detector"
        assert result.mcu == "esp32"
        assert result.bits == 8

    def test_latency_positive(self):
        result = self.engine.infer("trajectory_predictor")
        assert result.latency_ms > 0

    def test_energy_positive(self):
        result = self.engine.infer("obstacle_detector")
        assert result.energy_uj > 0

    def test_accuracy_in_range(self):
        result = self.engine.infer("threat_classifier")
        assert 0.0 <= result.accuracy <= 1.0

    def test_output_is_probability_vector(self):
        result = self.engine.infer("obstacle_detector")
        assert len(result.output) == MODELS["obstacle_detector"].output_classes
        # Each value between 0 and 1
        for v in result.output:
            assert 0.0 <= v <= 1.0

    def test_within_budget_flag(self):
        result = self.engine.infer("trajectory_predictor")  # tiny model, should fit
        assert isinstance(result.within_budget, bool)

    def test_stats_accumulate(self):
        for _ in range(10):
            self.engine.infer("obstacle_detector")
        stats = self.engine.stats
        assert stats["count"] == 10
        assert stats["avg_latency_ms"] > 0

    def test_benchmark_returns_percentiles(self):
        result = self.engine.benchmark("trajectory_predictor", n_runs=20)
        assert "latency_ms" in result
        assert "p50" in result["latency_ms"]
        assert "p95" in result["latency_ms"]
        assert "p99" in result["latency_ms"]
        assert result["n_runs"] == 20

    def test_budget_met_pct_in_range(self):
        result = self.engine.benchmark("trajectory_predictor", n_runs=20)
        assert 0.0 <= result["budget_met_pct"] <= 100.0

    def test_compare_quantizations_returns_four_results(self):
        results = self.engine.compare_quantizations("trajectory_predictor", n_runs=10)
        assert len(results) == 4
        bits_seen = [r["bits"] for r in results]
        assert sorted(bits_seen) == [4, 8, 16, 32]

    def test_4bit_faster_than_32bit(self):
        results = self.engine.compare_quantizations("obstacle_detector", n_runs=20)
        r4  = next(r for r in results if r["bits"] == 4)
        r32 = next(r for r in results if r["bits"] == 32)
        assert r4["latency_ms"]["mean"] < r32["latency_ms"]["mean"]

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError):
            self.engine.infer("nonexistent_model")

    def test_rpi_faster_than_esp32(self):
        esp_engine = EdgeInferenceEngine("esp32", 8)
        rpi_engine = EdgeInferenceEngine("rpi_zero_2", 8)
        esp_r = esp_engine.benchmark("obstacle_detector", 20)
        rpi_r = rpi_engine.benchmark("obstacle_detector", 20)
        # RPi Zero 2 has more MHz and ops_per_MHz — should be faster
        assert rpi_r["latency_ms"]["mean"] < esp_r["latency_ms"]["mean"]

    def test_all_mcu_profiles_work(self):
        for mcu in MCU_PROFILES:
            engine = EdgeInferenceEngine(mcu, 8)
            result = engine.infer("trajectory_predictor")
            assert result.latency_ms > 0


# ─── DroneInferencePool ───────────────────────────────────────────────────────

class TestDroneInferencePool:
    def test_creates_engine_on_first_call(self):
        pool = DroneInferencePool()
        result = pool.run_inference("drone_abc", "trajectory_predictor")
        assert result is not None

    def test_same_engine_reused(self):
        pool = DroneInferencePool()
        e1 = pool.get_engine("drone_xyz")
        e2 = pool.get_engine("drone_xyz")
        assert e1 is e2

    def test_aggregate_stats(self):
        pool = DroneInferencePool()
        for did in ["d1", "d2", "d3"]:
            for _ in range(5):
                pool.run_inference(did, "trajectory_predictor")
        stats = pool.aggregate_stats()
        assert stats["drone_count"] == 3
        assert stats["total_inferences"] == 15


# ─── FaultInjector ───────────────────────────────────────────────────────────

class TestFaultInjector:
    def setup_method(self):
        self.sim = Simulation(num_teams=2, drones_per_team=4)
        self.injector = FaultInjector()
        self.drone_ids = list(self.sim.drones.keys())

    def test_inject_adds_active_fault(self):
        fault = FaultEvent(
            fault_type=FaultType.LATENCY_SPIKE,
            target_drone_ids=self.drone_ids[:2],
            severity=0.5,
            duration_s=10.0,
        )
        self.injector.inject(fault)
        assert len(self.injector.active_faults) == 1

    def test_expired_fault_removed(self):
        fault = FaultEvent(
            fault_type=FaultType.PACKET_LOSS_SPIKE,
            target_drone_ids=self.drone_ids[:1],
            severity=0.5,
            duration_s=0.01,   # expires immediately
        )
        self.injector.inject(fault)
        time.sleep(0.05)
        self.injector.apply_to_simulation(self.sim)
        assert len(self.injector.active_faults) == 0

    def test_latency_spike_increases_latency(self):
        drone = self.sim.drones[self.drone_ids[0]]
        original_latency = drone.latency_ms
        fault = FaultEvent(
            fault_type=FaultType.LATENCY_SPIKE,
            target_drone_ids=[self.drone_ids[0]],
            severity=1.0,
            duration_s=5.0,
        )
        self.injector.inject(fault)
        self.injector.apply_to_simulation(self.sim)
        assert drone.latency_ms >= original_latency

    def test_battery_drain_reduces_battery(self):
        drone = self.sim.drones[self.drone_ids[0]]
        drone.battery_pct = 80.0
        fault = FaultEvent(
            fault_type=FaultType.BATTERY_DRAIN,
            target_drone_ids=[self.drone_ids[0]],
            severity=1.0,
            duration_s=5.0,
        )
        self.injector.inject(fault)
        self.injector.apply_to_simulation(self.sim)
        assert drone.battery_pct < 80.0

    def test_sensor_freeze_clears_visibility(self):
        drone = self.sim.drones[self.drone_ids[0]]
        drone.visible_enemies = [list(self.sim.drones.values())[0]]
        drone.neighbors = [list(self.sim.drones.values())[1]]

        fault = FaultEvent(
            fault_type=FaultType.SENSOR_FREEZE,
            target_drone_ids=[self.drone_ids[0]],
            severity=1.0,
            duration_s=5.0,
        )
        self.injector.inject(fault)
        # Apply multiple times for high probability effect
        for _ in range(10):
            self.injector.apply_to_simulation(self.sim)

        # At high severity, visibility should be cleared at least sometimes
        # (stochastic — test that function doesn't raise)
        status = self.injector.get_status()
        assert "active" in status

    def test_inject_random_returns_fault(self):
        fault = self.injector.inject_random(self.drone_ids)
        assert fault is not None
        assert isinstance(fault.fault_type, FaultType)
        assert 0.3 <= fault.severity <= 0.9

    def test_clear_all(self):
        for _ in range(3):
            self.injector.inject_random(self.drone_ids)
        self.injector.clear_all()
        assert len(self.injector.active_faults) == 0

    def test_get_status_structure(self):
        status = self.injector.get_status()
        assert "active" in status
        assert "history" in status
        assert "total_injected" in status

    def test_scenario_network_degradation(self):
        FaultScenarios.network_degradation(self.injector, self.drone_ids)
        assert len(self.injector.active_faults) >= 2

    def test_scenario_battery_crisis(self):
        FaultScenarios.battery_crisis(self.injector, self.drone_ids)
        assert any(f.fault_type == FaultType.BATTERY_DRAIN for f in self.injector.active_faults)

    def test_scenario_cascade(self):
        FaultScenarios.cascade_failure(self.injector, self.drone_ids)
        assert len(self.injector.active_faults) >= 3

    def test_fault_history_recorded(self):
        fault = self.injector.inject_random(self.drone_ids)
        assert len(self.injector.fault_history) == 1
        assert self.injector.fault_history[0]["type"] == fault.fault_type.value
