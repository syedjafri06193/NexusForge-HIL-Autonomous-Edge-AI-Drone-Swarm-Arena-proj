-- NexusForge TimescaleDB Schema
-- Stores telemetry as time-series hypertables for fast range queries

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Drone telemetry
CREATE TABLE IF NOT EXISTS drone_telemetry (
  time           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  session_id     TEXT NOT NULL,
  drone_id       TEXT NOT NULL,
  team           TEXT NOT NULL,
  pos_x          DOUBLE PRECISION,
  pos_y          DOUBLE PRECISION,
  vel_x          DOUBLE PRECISION,
  vel_y          DOUBLE PRECISION,
  heading        DOUBLE PRECISION,
  health         DOUBLE PRECISION,
  shield         DOUBLE PRECISION,
  battery_pct    DOUBLE PRECISION,
  kills          INTEGER,
  inference_ms   DOUBLE PRECISION,
  latency_ms     DOUBLE PRECISION,
  state          TEXT
);

SELECT create_hypertable('drone_telemetry', 'time', if_not_exists => TRUE);
CREATE INDEX ON drone_telemetry (session_id, time DESC);
CREATE INDEX ON drone_telemetry (drone_id, time DESC);

-- HIL hardware packets
CREATE TABLE IF NOT EXISTS hil_packets (
  time           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  session_id     TEXT NOT NULL,
  drone_id       TEXT NOT NULL,
  seq            INTEGER,
  battery_mv     DOUBLE PRECISION,
  battery_pct    DOUBLE PRECISION,
  current_ma     DOUBLE PRECISION,
  power_mw       DOUBLE PRECISION,
  cpu_load_pct   DOUBLE PRECISION,
  heap_free_kb   DOUBLE PRECISION,
  inference_us   INTEGER,
  rssi_dbm       INTEGER,
  snr_db         DOUBLE PRECISION,
  packet_loss_pct DOUBLE PRECISION,
  latency_ms     DOUBLE PRECISION
);

SELECT create_hypertable('hil_packets', 'time', if_not_exists => TRUE);
CREATE INDEX ON hil_packets (drone_id, time DESC);

-- Session events (kills, captures, hazards)
CREATE TABLE IF NOT EXISTS session_events (
  time        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  session_id  TEXT NOT NULL,
  event_type  TEXT NOT NULL,
  data        JSONB
);

SELECT create_hypertable('session_events', 'time', if_not_exists => TRUE);

-- Benchmark results
CREATE TABLE IF NOT EXISTS benchmark_results (
  time         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  model        TEXT NOT NULL,
  mcu          TEXT NOT NULL,
  bits         INTEGER,
  latency_p50  DOUBLE PRECISION,
  latency_p99  DOUBLE PRECISION,
  accuracy     DOUBLE PRECISION,
  budget_met   DOUBLE PRECISION,
  energy_uj    DOUBLE PRECISION,
  model_size_kb DOUBLE PRECISION
);

-- Continuous aggregates for real-time analytics
CREATE MATERIALIZED VIEW drone_telemetry_1min
WITH (timescaledb.continuous) AS
SELECT
  time_bucket('1 minute', time) AS bucket,
  session_id,
  team,
  AVG(health) AS avg_health,
  AVG(battery_pct) AS avg_battery,
  SUM(kills) AS total_kills,
  AVG(latency_ms) AS avg_latency,
  AVG(inference_ms) AS avg_inference,
  COUNT(*) AS packet_count
FROM drone_telemetry
GROUP BY bucket, session_id, team
WITH NO DATA;

SELECT add_continuous_aggregate_policy('drone_telemetry_1min',
  start_offset => INTERVAL '10 minutes',
  end_offset   => INTERVAL '1 minute',
  schedule_interval => INTERVAL '1 minute',
  if_not_exists => TRUE
);
