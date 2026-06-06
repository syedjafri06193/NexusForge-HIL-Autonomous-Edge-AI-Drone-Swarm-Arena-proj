"""
NexusForge Telemetry Persistence
Writes drone telemetry and events to TimescaleDB (via asyncpg)
and caches recent data in Redis for the dashboard.
"""

import asyncio
import json
import os
import time
from typing import List, Optional, Dict, Any

try:
    import asyncpg
    _PG_AVAILABLE = True
except ImportError:
    _PG_AVAILABLE = False

try:
    import redis.asyncio as aioredis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


DB_URL    = os.getenv("TIMESCALE_URL", "postgresql://nexus:nexusforge_dev@localhost:5432/nexusforge")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


class TelemetryWriter:
    """
    Batched async writes to TimescaleDB.
    Buffers inserts and flushes every N records or T seconds.
    """

    def __init__(self, flush_interval: float = 1.0, batch_size: int = 500):
        self.flush_interval = flush_interval
        self.batch_size     = batch_size
        self._pool: Optional[asyncpg.Pool] = None
        self._redis: Optional[Any] = None
        self._drone_buf: List[dict] = []
        self._hil_buf:   List[dict] = []
        self._event_buf: List[dict] = []
        self._running    = False
        self._flush_task: Optional[asyncio.Task] = None

    async def start(self):
        if _PG_AVAILABLE:
            try:
                self._pool = await asyncpg.create_pool(DB_URL, min_size=1, max_size=5)
            except Exception as e:
                print(f"[Telemetry] TimescaleDB unavailable: {e}")

        if _REDIS_AVAILABLE:
            try:
                self._redis = aioredis.from_url(REDIS_URL, decode_responses=True)
                await self._redis.ping()
            except Exception as e:
                print(f"[Telemetry] Redis unavailable: {e}")

        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self):
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
        await self._flush_all()
        if self._pool:
            await self._pool.close()
        if self._redis:
            await self._redis.close()

    # ─── Buffer writes ───────────────────────────────────────────────────────

    def record_drone_state(self, session_id: str, drone: dict):
        self._drone_buf.append({
            "session_id": session_id,
            "drone_id":   drone["id"],
            "team":       drone["team"],
            "pos_x":      drone["position"]["x"],
            "pos_y":      drone["position"]["y"],
            "vel_x":      drone["velocity"]["x"],
            "vel_y":      drone["velocity"]["y"],
            "heading":    drone["heading"],
            "health":     drone["health"],
            "shield":     drone["shield"],
            "battery_pct": drone["battery_pct"],
            "kills":      drone["kills"],
            "inference_ms": drone.get("inference_ms", 0),
            "latency_ms": drone.get("latency_ms", 0),
            "state":      drone["state"],
        })

    def record_hil_packet(self, session_id: str, packet: dict):
        self._hil_buf.append({
            "session_id":    session_id,
            "drone_id":      packet.get("drone_id"),
            "seq":           packet.get("seq"),
            "battery_mv":    packet.get("power", {}).get("battery_mv"),
            "battery_pct":   packet.get("power", {}).get("battery_pct"),
            "current_ma":    packet.get("power", {}).get("current_ma"),
            "power_mw":      packet.get("power", {}).get("power_mw"),
            "cpu_load_pct":  packet.get("compute", {}).get("cpu_load_pct"),
            "heap_free_kb":  packet.get("compute", {}).get("heap_free_kb"),
            "inference_us":  packet.get("compute", {}).get("inference_us"),
            "rssi_dbm":      packet.get("radio", {}).get("rssi_dbm"),
            "snr_db":        packet.get("radio", {}).get("snr_db"),
            "packet_loss_pct": packet.get("radio", {}).get("packet_loss_pct"),
            "latency_ms":    packet.get("latency_ms"),
        })

    def record_event(self, session_id: str, event: dict):
        self._event_buf.append({
            "session_id": session_id,
            "event_type": event.get("type"),
            "data": json.dumps(event),
        })

    def record_session_snapshot(self, session_id: str, snapshot: dict):
        """Record all drones and events from a sim snapshot."""
        for drone in snapshot.get("drones", []):
            self.record_drone_state(session_id, drone)
        for packet in snapshot.get("hil_telemetry", []):
            self.record_hil_packet(session_id, packet)
        for event in snapshot.get("events", []):
            self.record_event(session_id, event)

        # Cache latest state in Redis
        if self._redis:
            asyncio.create_task(self._cache_snapshot(session_id, snapshot))

    # ─── Redis cache ─────────────────────────────────────────────────────────

    async def _cache_snapshot(self, session_id: str, snapshot: dict):
        """Cache latest sim state in Redis for low-latency reads."""
        try:
            key = f"nexusforge:session:{session_id}:latest"
            await self._redis.set(key, json.dumps({
                "tick":    snapshot.get("tick"),
                "elapsed": snapshot.get("elapsed"),
                "scores":  snapshot.get("scores"),
                "drones":  snapshot.get("drones", [])[:20],  # sample
            }), ex=300)
        except Exception:
            pass

    async def get_cached_snapshot(self, session_id: str) -> Optional[dict]:
        if not self._redis:
            return None
        try:
            data = await self._redis.get(f"nexusforge:session:{session_id}:latest")
            return json.loads(data) if data else None
        except Exception:
            return None

    # ─── Flush loop ──────────────────────────────────────────────────────────

    async def _flush_loop(self):
        while self._running:
            await asyncio.sleep(self.flush_interval)
            await self._flush_all()

    async def _flush_all(self):
        if self._pool:
            await asyncio.gather(
                self._flush_drone_telemetry(),
                self._flush_hil_packets(),
                self._flush_events(),
                return_exceptions=True,
            )
        else:
            # No DB — just clear buffers to avoid memory leak
            self._drone_buf.clear()
            self._hil_buf.clear()
            self._event_buf.clear()

    async def _flush_drone_telemetry(self):
        if not self._drone_buf:
            return
        batch, self._drone_buf = self._drone_buf[:self.batch_size], self._drone_buf[self.batch_size:]
        try:
            async with self._pool.acquire() as conn:
                await conn.executemany("""
                    INSERT INTO drone_telemetry (
                        time, session_id, drone_id, team,
                        pos_x, pos_y, vel_x, vel_y, heading,
                        health, shield, battery_pct, kills,
                        inference_ms, latency_ms, state
                    ) VALUES (NOW(), $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                """, [
                    (r["session_id"], r["drone_id"], r["team"],
                     r["pos_x"], r["pos_y"], r["vel_x"], r["vel_y"], r["heading"],
                     r["health"], r["shield"], r["battery_pct"], r["kills"],
                     r["inference_ms"], r["latency_ms"], r["state"])
                    for r in batch
                ])
        except Exception as e:
            # Put back if failed
            self._drone_buf = batch + self._drone_buf

    async def _flush_hil_packets(self):
        if not self._hil_buf:
            return
        batch, self._hil_buf = self._hil_buf[:self.batch_size], self._hil_buf[self.batch_size:]
        try:
            async with self._pool.acquire() as conn:
                await conn.executemany("""
                    INSERT INTO hil_packets (
                        time, session_id, drone_id, seq,
                        battery_mv, battery_pct, current_ma, power_mw,
                        cpu_load_pct, heap_free_kb, inference_us,
                        rssi_dbm, snr_db, packet_loss_pct, latency_ms
                    ) VALUES (NOW(), $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                """, [
                    (r["session_id"], r["drone_id"], r["seq"],
                     r["battery_mv"], r["battery_pct"], r["current_ma"], r["power_mw"],
                     r["cpu_load_pct"], r["heap_free_kb"], r["inference_us"],
                     r["rssi_dbm"], r["snr_db"], r["packet_loss_pct"], r["latency_ms"])
                    for r in batch
                ])
        except Exception:
            self._hil_buf = batch + self._hil_buf

    async def _flush_events(self):
        if not self._event_buf:
            return
        batch, self._event_buf = self._event_buf[:500], self._event_buf[500:]
        try:
            async with self._pool.acquire() as conn:
                await conn.executemany("""
                    INSERT INTO session_events (time, session_id, event_type, data)
                    VALUES (NOW(), $1, $2, $3::jsonb)
                """, [(r["session_id"], r["event_type"], r["data"]) for r in batch])
        except Exception:
            self._event_buf = batch + self._event_buf

    # ─── Analytics queries ───────────────────────────────────────────────────

    async def query_session_analytics(self, session_id: str) -> dict:
        """Query TimescaleDB for session-level analytics."""
        if not self._pool:
            return {"error": "TimescaleDB not connected"}
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT team,
                        AVG(health) as avg_health,
                        AVG(battery_pct) as avg_battery,
                        MAX(kills) as max_kills,
                        AVG(latency_ms) as avg_latency,
                        AVG(inference_ms) as avg_inference,
                        COUNT(*) as samples
                    FROM drone_telemetry
                    WHERE session_id = $1
                    GROUP BY team
                """, session_id)
                return {"by_team": [dict(r) for r in rows]}
        except Exception as e:
            return {"error": str(e)}

    async def query_latency_timeseries(self, session_id: str, drone_id: str) -> list:
        """Get latency time-series for a specific drone."""
        if not self._pool:
            return []
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT time_bucket('5 seconds', time) as bucket,
                        AVG(latency_ms) as avg_latency,
                        AVG(inference_ms) as avg_inference,
                        AVG(battery_pct) as avg_battery
                    FROM drone_telemetry
                    WHERE session_id = $1 AND drone_id = $2
                    GROUP BY bucket ORDER BY bucket
                """, session_id, drone_id)
                return [dict(r) for r in rows]
        except Exception as e:
            return []

    async def query_power_profile(self, session_id: str) -> list:
        """Aggregate power usage across the HIL fleet."""
        if not self._pool:
            return []
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT drone_id,
                        AVG(power_mw) as avg_power_mw,
                        AVG(battery_pct) as avg_battery,
                        AVG(cpu_load_pct) as avg_cpu,
                        AVG(inference_us) / 1000.0 as avg_inference_ms,
                        MIN(rssi_dbm) as min_rssi,
                        AVG(packet_loss_pct) as avg_loss
                    FROM hil_packets
                    WHERE session_id = $1
                    GROUP BY drone_id
                    ORDER BY avg_power_mw DESC
                """, session_id)
                return [dict(r) for r in rows]
        except Exception as e:
            return []
