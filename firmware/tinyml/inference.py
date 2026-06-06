"""
NexusForge Edge AI Simulator
Simulates quantized TinyML inference on constrained MCUs (ESP32/STM32).
Models power draw, latency, and accuracy degradation at different bit widths.
"""

import math
import random
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─── MCU Profiles ─────────────────────────────────────────────────────────────

MCU_PROFILES = {
    "esp32": {
        "cpu_mhz": 240,
        "ram_kb": 520,
        "flash_mb": 4,
        "mw_idle": 20,
        "mw_active": 240,
        "mw_wifi": 160,
        "ops_per_mhz": 1.2,          # scalar multiply-adds per MHz per ms
    },
    "esp32_s3": {
        "cpu_mhz": 240,
        "ram_kb": 512,
        "flash_mb": 8,
        "mw_idle": 22,
        "mw_active": 260,
        "mw_wifi": 170,
        "ops_per_mhz": 1.8,          # vector ops via SIMD
    },
    "stm32f4": {
        "cpu_mhz": 168,
        "ram_kb": 192,
        "flash_mb": 1,
        "mw_idle": 5,
        "mw_active": 80,
        "mw_wifi": 0,                # separate radio module
        "ops_per_mhz": 1.0,
    },
    "stm32h7": {
        "cpu_mhz": 480,
        "ram_kb": 1024,
        "flash_mb": 2,
        "mw_idle": 8,
        "mw_active": 150,
        "mw_wifi": 0,
        "ops_per_mhz": 2.0,          # FPU + DSP extensions
    },
    "rpi_zero_2": {
        "cpu_mhz": 1000,
        "ram_kb": 512 * 1024,
        "flash_mb": 0,
        "mw_idle": 100,
        "mw_active": 400,
        "mw_wifi": 50,
        "ops_per_mhz": 4.0,
    },
}


# ─── Model specs ──────────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    name: str
    param_count: int
    flops_per_inference: int      # multiply-accumulate ops
    input_shape: Tuple[int, ...]
    output_classes: int
    base_accuracy: float          # float32 accuracy (0-1)

    # Computed per quantization level
    def model_size_kb(self, bits: int) -> float:
        return self.param_count * bits / (8 * 1024)

    def quantized_flops(self, bits: int) -> int:
        """Lower-bit ops are cheaper but introduce noise."""
        speedup = {32: 1.0, 16: 1.8, 8: 3.5, 4: 6.0}.get(bits, 1.0)
        return int(self.flops_per_inference / speedup)

    def accuracy_at_bits(self, bits: int) -> float:
        degradation = {32: 0.0, 16: 0.005, 8: 0.02, 4: 0.06}.get(bits, 0.1)
        return max(0.0, self.base_accuracy - degradation)


# Predefined edge-AI models for drone tasks
MODELS = {
    "obstacle_detector": ModelSpec(
        name="MobileNetV1-0.25",
        param_count=475_000,
        flops_per_inference=41_000_000,
        input_shape=(96, 96, 3),
        output_classes=10,
        base_accuracy=0.87,
    ),
    "threat_classifier": ModelSpec(
        name="SqueezeNet-Lite",
        param_count=730_000,
        flops_per_inference=350_000_000,
        input_shape=(128, 128, 3),
        output_classes=4,
        base_accuracy=0.91,
    ),
    "trajectory_predictor": ModelSpec(
        name="TinyLSTM",
        param_count=48_000,
        flops_per_inference=960_000,
        input_shape=(10, 6),
        output_classes=2,
        base_accuracy=0.83,
    ),
    "swarm_coordinator": ModelSpec(
        name="TinyTransformer",
        param_count=210_000,
        flops_per_inference=4_200_000,
        input_shape=(8, 12),
        output_classes=8,
        base_accuracy=0.88,
    ),
}


# ─── Inference result ─────────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    model_name: str
    mcu: str
    bits: int
    latency_ms: float
    power_mw: float
    energy_uj: float          # microjoules
    accuracy: float
    output: List[float]       # softmax probabilities
    within_budget: bool
    memory_used_kb: float
    dropped: bool = False     # true if latency > budget


# ─── Inference engine ─────────────────────────────────────────────────────────

class EdgeInferenceEngine:
    """
    Simulates quantized TinyML inference on a given MCU.
    Returns latency, power draw, energy cost, and (simulated) accuracy.
    """

    def __init__(self, mcu_type: str = "esp32", bits: int = 8, budget_ms: float = 20.0):
        self.mcu_type = mcu_type
        self.mcu = MCU_PROFILES.get(mcu_type, MCU_PROFILES["esp32"])
        self.bits = bits
        self.budget_ms = budget_ms
        self._inference_count = 0
        self._total_energy_uj = 0.0
        self._total_latency_ms = 0.0
        self._dropped = 0

    def infer(self, model_name: str, input_data: Optional[np.ndarray] = None) -> InferenceResult:
        model = MODELS.get(model_name)
        if not model:
            raise ValueError(f"Unknown model: {model_name}")

        # Memory check
        model_kb = model.model_size_kb(self.bits)
        ram_kb = self.mcu["ram_kb"]
        activation_kb = math.prod(model.input_shape) * self.bits / (8 * 1024) * 4
        total_memory = model_kb + activation_kb

        # Latency calculation
        flops = model.quantized_flops(self.bits)
        ops_per_ms = self.mcu["cpu_mhz"] * self.mcu["ops_per_mhz"] * 1000
        base_latency = flops / ops_per_ms

        # Add jitter (cache misses, interrupt latency, DMA)
        jitter = random.gauss(0, base_latency * 0.08)
        latency = max(0.1, base_latency + jitter)

        # Power during inference
        power_mw = self.mcu["mw_active"]

        # Energy cost
        energy_uj = power_mw * latency * 1000  # mW * ms = µJ

        # Accuracy
        accuracy = model.accuracy_at_bits(self.bits)
        # Add small stochastic noise to simulate real quantization effects
        accuracy += random.gauss(0, 0.01)
        accuracy = max(0.0, min(1.0, accuracy))

        # Simulate output (softmax probabilities)
        raw = np.random.dirichlet(np.ones(model.output_classes) * (accuracy * 10 + 0.5))
        # Bias toward correct class proportional to accuracy
        raw[0] += accuracy * 0.3
        raw /= raw.sum()

        within_budget = latency <= self.budget_ms
        dropped = not within_budget or total_memory > ram_kb

        self._inference_count += 1
        self._total_energy_uj += energy_uj
        self._total_latency_ms += latency
        if dropped:
            self._dropped += 1

        return InferenceResult(
            model_name=model_name,
            mcu=self.mcu_type,
            bits=self.bits,
            latency_ms=round(latency, 3),
            power_mw=round(power_mw, 1),
            energy_uj=round(energy_uj, 2),
            accuracy=round(accuracy, 4),
            output=raw.tolist(),
            within_budget=within_budget,
            memory_used_kb=round(total_memory, 2),
            dropped=dropped,
        )

    def benchmark(self, model_name: str, n_runs: int = 100) -> dict:
        """Run n_runs inferences and collect performance stats."""
        results = [self.infer(model_name) for _ in range(n_runs)]
        latencies = [r.latency_ms for r in results]
        energies = [r.energy_uj for r in results]
        accuracies = [r.accuracy for r in results]
        return {
            "model": model_name,
            "mcu": self.mcu_type,
            "bits": self.bits,
            "n_runs": n_runs,
            "latency_ms": {
                "mean": round(np.mean(latencies), 3),
                "p50": round(np.percentile(latencies, 50), 3),
                "p95": round(np.percentile(latencies, 95), 3),
                "p99": round(np.percentile(latencies, 99), 3),
                "max": round(max(latencies), 3),
            },
            "energy_uj": {
                "mean": round(np.mean(energies), 2),
                "total": round(sum(energies), 2),
            },
            "accuracy": {
                "mean": round(np.mean(accuracies), 4),
                "min": round(min(accuracies), 4),
            },
            "budget_met_pct": round(
                sum(1 for r in results if r.within_budget) / n_runs * 100, 1
            ),
            "model_size_kb": round(MODELS[model_name].model_size_kb(self.bits), 2),
            "memory_ok": MODELS[model_name].model_size_kb(self.bits) < self.mcu["ram_kb"],
        }

    def compare_quantizations(self, model_name: str, n_runs: int = 50) -> List[dict]:
        """Compare 4-bit, 8-bit, 16-bit, 32-bit for a given model."""
        results = []
        for bits in [4, 8, 16, 32]:
            self.bits = bits
            results.append(self.benchmark(model_name, n_runs))
        self.bits = 8  # restore
        return results

    @property
    def stats(self) -> dict:
        if self._inference_count == 0:
            return {"count": 0}
        return {
            "count": self._inference_count,
            "dropped": self._dropped,
            "drop_rate_pct": round(self._dropped / self._inference_count * 100, 2),
            "avg_latency_ms": round(self._total_latency_ms / self._inference_count, 3),
            "total_energy_uj": round(self._total_energy_uj, 2),
            "avg_energy_uj": round(self._total_energy_uj / self._inference_count, 3),
        }


# ─── Per-drone engine pool ────────────────────────────────────────────────────

class DroneInferencePool:
    """
    Manages one EdgeInferenceEngine per drone.
    Provides telemetry aggregation across the swarm.
    """

    def __init__(self):
        self._engines: Dict[str, EdgeInferenceEngine] = {}

    def get_engine(self, drone_id: str, mcu: str = "esp32", bits: int = 8) -> EdgeInferenceEngine:
        if drone_id not in self._engines:
            self._engines[drone_id] = EdgeInferenceEngine(mcu, bits)
        return self._engines[drone_id]

    def run_inference(self, drone_id: str, model_name: str, **kwargs) -> InferenceResult:
        engine = self.get_engine(drone_id, **kwargs)
        return engine.infer(model_name)

    def aggregate_stats(self) -> dict:
        if not self._engines:
            return {}
        all_stats = [e.stats for e in self._engines.values() if e._inference_count > 0]
        if not all_stats:
            return {}
        return {
            "drone_count": len(all_stats),
            "total_inferences": sum(s["count"] for s in all_stats),
            "avg_latency_ms": round(
                sum(s["avg_latency_ms"] for s in all_stats) / len(all_stats), 3
            ),
            "total_energy_uj": round(sum(s["total_energy_uj"] for s in all_stats), 2),
            "avg_drop_rate_pct": round(
                sum(s["drop_rate_pct"] for s in all_stats) / len(all_stats), 2
            ),
        }
