/*
 * NexusForge ESP32 Firmware
 * Hardware-in-the-Loop drone node for real ESP32 boards.
 *
 * Connects to MQTT broker, sends telemetry at 20Hz,
 * receives swarm commands, runs lightweight TinyML inference.
 *
 * Board: ESP32 DevKit / Wemos D1 Mini32
 * Framework: Arduino + FreeRTOS
 * Build: PlatformIO
 */

#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/queue.h>

// ─── Configuration ─────────────────────────────────────────────────────────

#define WIFI_SSID     "YOUR_SSID"
#define WIFI_PASS     "YOUR_PASSWORD"
#define MQTT_BROKER   "192.168.1.100"   // NexusForge server IP
#define MQTT_PORT     1883
#define SESSION_ID    "session01"
#define DRONE_ID      "hw_001"
#define TELEMETRY_HZ  20
#define COMMAND_TOPIC "nexusforge/" SESSION_ID "/" DRONE_ID "/command"
#define TELEM_TOPIC   "nexusforge/" SESSION_ID "/" DRONE_ID "/telemetry"

// ─── Global state ───────────────────────────────────────────────────────────

WiFiClient   wifiClient;
PubSubClient mqtt(wifiClient);

// Simulated sensor state (replace with real IMU/GPS on actual hardware)
struct DroneState {
  float pos_x = 600.0f, pos_y = 450.0f;
  float vel_x = 0.0f,   vel_y = 0.0f;
  float heading = 0.0f;
  float health = 100.0f, shield = 50.0f;
  float battery_pct = 100.0f;
  int   kills = 0;
  char  state[16] = "patrolling";
  uint32_t seq = 0;
};

DroneState drone;
QueueHandle_t cmdQueue;

// ─── TinyML stub ───────────────────────────────────────────────────────────
// Replace with TFLite Micro or ONNX Runtime for real inference

struct InferenceResult {
  float action[4];     // [move_x, move_y, fire, evade]
  uint32_t latency_us;
};

InferenceResult run_inference(float* features, int n_features) {
  InferenceResult result;
  uint32_t t0 = micros();

  // Stub: simple rule-based fallback
  // On real hardware: tflite::MicroInterpreter::Invoke()
  result.action[0] = random(-100, 100) / 100.0f;  // move_x
  result.action[1] = random(-100, 100) / 100.0f;  // move_y
  result.action[2] = random(0, 100) > 70 ? 1.0f : 0.0f;  // fire
  result.action[3] = drone.health < 30 ? 1.0f : 0.0f;    // evade

  result.latency_us = micros() - t0;
  return result;
}

// ─── WiFi + MQTT setup ──────────────────────────────────────────────────────

void connectWiFi() {
  Serial.printf("[WiFi] Connecting to %s", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500); Serial.print(".");
  }
  Serial.printf("\n[WiFi] Connected, IP: %s\n", WiFi.localIP().toString().c_str());
}

void onMqttMessage(char* topic, byte* payload, unsigned int length) {
  // Parse incoming swarm command
  StaticJsonDocument<256> doc;
  DeserializationError err = deserializeJson(doc, payload, length);
  if (err) return;

  const char* cmd = doc["command"] | "";
  if (strcmp(cmd, "evade") == 0)   { strncpy(drone.state, "evading",  15); }
  if (strcmp(cmd, "attack") == 0)  { strncpy(drone.state, "attacking", 15); }
  if (strcmp(cmd, "regroup") == 0) { strncpy(drone.state, "regrouping",15); }

  if (doc.containsKey("waypoint")) {
    drone.pos_x = doc["waypoint"]["x"] | drone.pos_x;
    drone.pos_y = doc["waypoint"]["y"] | drone.pos_y;
  }

  Serial.printf("[CMD] %s\n", cmd);
}

void connectMQTT() {
  mqtt.setServer(MQTT_BROKER, MQTT_PORT);
  mqtt.setCallback(onMqttMessage);
  mqtt.setBufferSize(1024);

  while (!mqtt.connected()) {
    Serial.print("[MQTT] Connecting...");
    if (mqtt.connect(DRONE_ID)) {
      Serial.println("OK");
      mqtt.subscribe(COMMAND_TOPIC);
    } else {
      Serial.printf(" failed, rc=%d, retry in 2s\n", mqtt.state());
      delay(2000);
    }
  }
}

// ─── Telemetry task (20 Hz) ─────────────────────────────────────────────────

void telemetryTask(void* param) {
  const TickType_t period = pdMS_TO_TICKS(1000 / TELEMETRY_HZ);
  TickType_t lastWake = xTaskGetTickCount();

  // Build sensor feature vector for TinyML
  float features[8];

  while (true) {
    vTaskDelayUntil(&lastWake, period);
    if (!mqtt.connected()) continue;

    drone.seq++;

    // Run lightweight inference
    features[0] = drone.pos_x / 1200.0f;
    features[1] = drone.pos_y / 900.0f;
    features[2] = drone.vel_x / 280.0f;
    features[3] = drone.vel_y / 280.0f;
    features[4] = drone.health / 100.0f;
    features[5] = drone.shield / 50.0f;
    features[6] = drone.battery_pct / 100.0f;
    features[7] = drone.heading / (2 * PI);

    InferenceResult infer = run_inference(features, 8);

    // Apply inference output to motion (simplified)
    if (strcmp(drone.state, "patrolling") == 0) {
      drone.vel_x = infer.action[0] * 50.0f;
      drone.vel_y = infer.action[1] * 50.0f;
    }
    drone.pos_x = constrain(drone.pos_x + drone.vel_x * 0.05f, 10, 1190);
    drone.pos_y = constrain(drone.pos_y + drone.vel_y * 0.05f, 10, 890);
    drone.battery_pct = max(0.0f, drone.battery_pct - 0.001f);

    // Serialize telemetry
    StaticJsonDocument<512> doc;
    doc["drone_id"]     = DRONE_ID;
    doc["session_id"]   = SESSION_ID;
    doc["seq"]          = drone.seq;
    doc["ts"]           = (double)esp_timer_get_time() / 1e6;

    JsonObject pos = doc.createNestedObject("pos");
    pos["x"] = roundf(drone.pos_x * 10) / 10;
    pos["y"] = roundf(drone.pos_y * 10) / 10;

    JsonObject vel = doc.createNestedObject("vel");
    vel["x"] = roundf(drone.vel_x * 10) / 10;
    vel["y"] = roundf(drone.vel_y * 10) / 10;

    doc["heading"] = roundf(drone.heading * 1000) / 1000;

    JsonObject power = doc.createNestedObject("power");
    power["battery_pct"] = roundf(drone.battery_pct * 10) / 10;
    power["battery_mv"]  = (int)(3.3f + drone.battery_pct / 100.0f * 0.9f * 1000);

    JsonObject compute = doc.createNestedObject("compute");
    compute["cpu_load_pct"] = 30 + (infer.latency_us / 1000);
    compute["inference_us"] = infer.latency_us;
    compute["heap_free_kb"] = ESP.getFreeHeap() / 1024;

    JsonObject combat = doc.createNestedObject("combat");
    combat["health"] = roundf(drone.health * 10) / 10;
    combat["shield"] = roundf(drone.shield * 10) / 10;
    combat["kills"]  = drone.kills;
    combat["state"]  = drone.state;

    doc["rssi"] = WiFi.RSSI();

    char buf[512];
    size_t len = serializeJson(doc, buf, sizeof(buf));
    mqtt.publish(TELEM_TOPIC, (uint8_t*)buf, len, false);
  }
}

// ─── MQTT keepalive task ────────────────────────────────────────────────────

void mqttTask(void* param) {
  while (true) {
    if (!mqtt.connected()) connectMQTT();
    mqtt.loop();
    vTaskDelay(pdMS_TO_TICKS(10));
  }
}

// ─── Setup & Loop ───────────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);
  Serial.println("\n[NexusForge] ESP32 Firmware booting...");

  connectWiFi();
  connectMQTT();

  cmdQueue = xQueueCreate(8, sizeof(char[64]));

  // Spawn FreeRTOS tasks
  xTaskCreatePinnedToCore(telemetryTask, "telemetry", 8192, NULL, 2, NULL, 1);
  xTaskCreatePinnedToCore(mqttTask,      "mqtt_loop", 4096, NULL, 1, NULL, 0);

  Serial.printf("[NexusForge] Drone %s online | Session: %s\n", DRONE_ID, SESSION_ID);
}

void loop() {
  // Main loop is empty — all work done in FreeRTOS tasks
  vTaskDelay(portMAX_DELAY);
}
