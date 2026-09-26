#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <WiFiClientSecure.h>
#include <esp_camera.h>
#include <esp_timer.h>
#include <ArduinoWebsockets.h>
#include <ESP_I2S.h>
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>
#include <atomic>
#include <ctime>
#include "config_defaults.h"
#include "imu_driver.h"
#include "sensor_math.h"
#include "microphone_session.h"
#include "audio_stream.h"
#include "network_transport.h"
#include "camera_pins.h"

using websockets::WebsocketsClient;
using websockets::WebsocketsMessage;
constexpr size_t kAudioBytes = 640; // 20 ms, 16000 Hz, mono, PCM s16le
constexpr uint32_t kRetryMs = 2000;
static_assert(NAVGUIDE_CAMERA_FPS >= 0 && NAVGUIDE_CAMERA_FPS <= 60, "FPS must be 0..60");
static_assert(NAVGUIDE_JPEG_QUALITY >= 5 && NAVGUIDE_JPEG_QUALITY <= 40, "JPEG quality must be 5..40");
static_assert(NAVGUIDE_SPEAKER_GAIN >= 0.0f && NAVGUIDE_SPEAKER_GAIN <= 1.0f, "Gain must be 0..1");
struct AudioChunk { uint32_t captured; uint8_t data[kAudioBytes]; };
QueueHandle_t audioQueue = nullptr;
I2SClass microphone, speaker;
std::atomic<bool> networkReady{false}, audioReady{false};
std::atomic<uint32_t> audioDropped{0};
framesize_t cameraSize = NAVGUIDE_CAMERA_SIZE, cameraMax = NAVGUIDE_CAMERA_MAX_SIZE;
int cameraQuality = NAVGUIDE_JPEG_QUALITY, cameraFps = NAVGUIDE_CAMERA_FPS;
bool configured = false;

static bool networkAvailable() {
  if (!networkReady.load()) return false;
#if NAVGUIDE_TLS
  // Certificate verification needs a valid wall clock; no insecure fallback.
  return time(nullptr) >= 1704067200;
#else
  return true;
#endif
}
static void configureSocket(WebsocketsClient& ws) {
  if (strlen(NAVGUIDE_DEVICE_TOKEN)) ws.addHeader("Authorization", String("Bearer ") + NAVGUIDE_DEVICE_TOKEN);
}
static bool connectSocket(WebsocketsClient& ws, const char* path) {
  // BoundedSocket already selects a verified TLS transport when configured.
  return ws.connect(NAVGUIDE_SERVER_HOST, NAVGUIDE_SERVER_PORT, path);
}
static void failBoot(const char* message) {
  Serial.println(message);
  while (true) delay(1000);
}
static bool initCamera() {
  if (!psramFound()) { Serial.println("[CAM] PSRAM required"); return false; }
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0; c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM; c.pin_d1 = Y3_GPIO_NUM;
  c.pin_d2 = Y4_GPIO_NUM; c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM; c.pin_d5 = Y7_GPIO_NUM;
  c.pin_d6 = Y8_GPIO_NUM; c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM; c.pin_pclk = PCLK_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM; c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM; c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM; c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000; c.pixel_format = PIXFORMAT_JPEG;
  // Allocate for the largest allowed resolution before switching to streaming size.
  c.frame_size = NAVGUIDE_CAMERA_MAX_SIZE; c.jpeg_quality = cameraQuality;
  c.fb_count = 2; c.fb_location = CAMERA_FB_IN_PSRAM; c.grab_mode = CAMERA_GRAB_LATEST;
  const esp_err_t error = esp_camera_init(&c);
  if (error != ESP_OK) { Serial.printf("[CAM] init error 0x%x\n", error); return false; }
  sensor_t* sensor = esp_camera_sensor_get();
  if (!sensor) return false;
  cameraMax = static_cast<framesize_t>(sensor->status.framesize);
  if (cameraSize > cameraMax) cameraSize = cameraMax;
  if (sensor->set_framesize(sensor, cameraSize) != 0) return false;
  sensor->set_hmirror(sensor, NAVGUIDE_CAMERA_HMIRROR);
  sensor->set_vflip(sensor, NAVGUIDE_CAMERA_VFLIP);
  sensor->set_gain_ctrl(sensor, 1);
  sensor->set_exposure_ctrl(sensor, 1);
  sensor->set_whitebal(sensor, 1);
  return true;
}
static bool parseFramesize(const String& name, framesize_t& out) {
  struct Entry { const char* name; framesize_t size; };
  static const Entry entries[] = {{"QVGA", FRAMESIZE_QVGA}, {"VGA", FRAMESIZE_VGA},
    {"SVGA", FRAMESIZE_SVGA}, {"XGA", FRAMESIZE_XGA}, {"HD", FRAMESIZE_HD},
    {"SXGA", FRAMESIZE_SXGA}, {"UXGA", FRAMESIZE_UXGA}, {"FHD", FRAMESIZE_FHD},
    {"QXGA", FRAMESIZE_QXGA}};
  for (const auto& entry : entries) if (name == entry.name) { out = entry.size; return true; }
  return false;
}
static bool setFramesize(framesize_t size) {
  auto* sensor = esp_camera_sensor_get();
  if (!sensor || size > cameraMax || sensor->set_framesize(sensor, size) != 0) return false;
  cameraSize = size;
  return true;
}
static bool sendFrame(WebsocketsClient& ws) {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) return false;
  bool ok = fb->format == PIXFORMAT_JPEG && ws.streamBinary();
  // Fragment one JPEG message to bound the library's masked-frame copy to 4 KiB.
  for (size_t offset = 0; ok && offset < fb->len; offset += 4096) {
    const size_t bytes = min(static_cast<size_t>(4096), fb->len - offset);
    ok = ws.sendBinary(reinterpret_cast<const char*>(fb->buf + offset), bytes) && ws.available();
    delay(0);
  }
  if (ok) ok = ws.end() && ws.available();
  esp_camera_fb_return(fb); // Exactly one return, including failed sends.
  return ok;
}
static void snapshot(WebsocketsClient& ws) {
  const framesize_t previous = cameraSize;
  const int previousQuality = cameraQuality;
  auto* sensor = esp_camera_sensor_get();
  if (!sensor || !setFramesize(cameraMax)) { ws.send("ERR:SNAP"); return; }
  sensor->set_quality(sensor, 12);
  delay(120);
  // Discard buffered frames captured before the resolution change.
  for (int i = 0; i < 2; ++i) {
    camera_fb_t* stale = esp_camera_fb_get();
    if (stale) esp_camera_fb_return(stale);
  }
  const bool ok = ws.send("SNAP:BEGIN") && sendFrame(ws);
  if (ok) ws.send("SNAP:END");
  else ws.close();
  setFramesize(previous);
  sensor->set_quality(sensor, previousQuality);
}
static bool parseInteger(const String& value, int& out) {
  char* end = nullptr;
  const long parsed = strtol(value.c_str(), &end, 10);
  if (!value.length() || !end || *end || parsed < 0 || parsed > 10000) return false;
  out = static_cast<int>(parsed);
  return true;
}
static void cameraTask(void*) {
  for (;;) {
  // A fresh client resets fragmented-message state after interrupted transmission.
  if (!networkAvailable()) { delay(100); continue; }
  WebsocketsClient ws(std::make_shared<BoundedSocket>());
  configureSocket(ws);
  ws.onMessage([&](WebsocketsMessage msg) {
    if (!msg.isText() || msg.length() > 128) return;
    String command = msg.data(); command.trim();
    bool ok = false;
    if (command.startsWith("SET:FRAMESIZE=")) {
      String name = command.substring(14); name.toUpperCase();
      framesize_t requested;
      ok = parseFramesize(name, requested) && setFramesize(requested);
    } else if (command.startsWith("SET:QUALITY=")) {
      int quality;
      auto* sensor = esp_camera_sensor_get();
      if (parseInteger(command.substring(12), quality) && quality >= 5 && quality <= 40 && sensor) {
        ok = sensor->set_quality(sensor, quality) == 0;
        if (ok) cameraQuality = quality;
      }
    } else if (command.startsWith("SET:FPS=")) {
      int fps;
      ok = parseInteger(command.substring(8), fps) && fps <= 60;
      if (ok) cameraFps = fps; // 0 means unrestricted; actual frame rate is measured below.
    } else if (command == "SNAP:HQ") { snapshot(ws); return; }
    else { ws.send("ERR:UNKNOWN_COMMAND"); return; }
    ws.send(ok ? "OK:SET" : "ERR:SET");
  });
  if (!connectSocket(ws, "/ws/camera")) { delay(kRetryMs); continue; }
  Serial.println("[CAM] connected");
  uint32_t nextFrame = millis(), lastLog = millis(), sent = 0, failed = 0, lastPing = 0;
  while (networkAvailable() && ws.available()) {
    ws.poll();
    const uint32_t now = millis();
    if (ws.available() && static_cast<int32_t>(now - nextFrame) >= 0) {
      if (sendFrame(ws)) ++sent;
      else { ++failed; ws.close(); }
      nextFrame = millis() + (cameraFps > 0 ? 1000 / cameraFps : 1);
    }
    if (ws.available() && now - lastPing >= 15000) { if (!ws.ping()) ws.close(); lastPing = now; }
    if (now - lastLog >= 10000) {
      Serial.printf("[CAM] sent=%lu failed=%lu measured_fps=%.2f target_fps=%d heap=%u\n",
                    (unsigned long)sent, (unsigned long)failed, sent * 1000.0f / (now - lastLog), cameraFps, ESP.getFreeHeap());
      sent = 0; failed = 0; lastLog = now;
    }
    delay(1);
  }
  ws.close();
  delay(kRetryMs);
  }
}
static void microphoneTask(void*) {
  for (;;) {
    AudioChunk chunk;
    size_t read = 0;
    // readBytes preserves signed PCM samples, including the valid value -1.
    while (read < kAudioBytes) {
      const size_t count = microphone.readBytes(reinterpret_cast<char*>(chunk.data) + read, kAudioBytes - read);
      if (!count) { delay(1); break; }
      read += count;
    }
    if (read != kAudioBytes || !audioReady.load()) continue;
    chunk.captured = millis();
    if (xQueueSend(audioQueue, &chunk, 0) != pdPASS) {
      AudioChunk dropped;
      if (xQueueReceive(audioQueue, &dropped, 0) == pdPASS) ++audioDropped;
      if (xQueueSend(audioQueue, &chunk, 0) != pdPASS) ++audioDropped;
    }
  }
}
static void audioTask(void*) {
  WebsocketsClient ws(std::make_shared<BoundedSocket>());
  configureSocket(ws);
  navguide::MicrophoneSession session(millis());
  auto stopUpload = [&]() {
    audioReady.store(false);
    xQueueReset(audioQueue);
  };
  ws.onMessage([&](WebsocketsMessage msg) {
    if (!msg.isText() || msg.length() > 128) return;
    String control = msg.data(); control.trim();
    if (!session.onControl(control.c_str(), millis())) return;
    // Clear any old samples at both the stop and acknowledged-start boundaries.
    stopUpload();
    audioReady.store(session.streaming());
    Serial.printf("[MIC] %s\n", control.c_str());
  });
  uint32_t lastPing = 0;
  for (;;) {
    if (!networkAvailable()) {
      stopUpload();
      if (ws.available()) ws.close();
      session.connectionLost(millis());
      delay(100);
      continue;
    }
    session.tick(millis());
    if (session.retrying()) {
      stopUpload();
      if (ws.available()) ws.close();
      if (!session.canConnect(millis())) { delay(20); continue; }
      if (!connectSocket(ws, "/ws_audio")) {
        session.connectionLost(millis());
        continue;
      }
      Serial.println("[MIC] connected; waiting for START acknowledgement");
      session.startSent(millis());
      if (!ws.send("START") || !ws.available()) {
        ws.close();
        session.connectionLost(millis());
        continue;
      }
      lastPing = millis();
    }
    if (!ws.available()) {
      stopUpload();
      session.connectionLost(millis());
      continue;
    }
    ws.poll();
    session.tick(millis());
    if (session.retrying()) {
      stopUpload();
      ws.close();
      continue;
    }
    if (session.streaming()) {
      AudioChunk chunk;
      if (xQueueReceive(audioQueue, &chunk, pdMS_TO_TICKS(5)) == pdPASS) {
        if (millis() - chunk.captured > 120) ++audioDropped;
        else if (!ws.sendBinary(reinterpret_cast<const char*>(chunk.data), kAudioBytes) || !ws.available()) {
          stopUpload();
          ws.close();
          session.connectionLost(millis());
          continue;
        }
      }
    }
    if (millis() - lastPing >= 15000) {
      if (!ws.ping() || !ws.available()) {
        stopUpload();
        ws.close();
        session.connectionLost(millis());
      }
      lastPing = millis();
    }
    delay(1);
  }
}

class HttpTransport {
public:
  explicit HttpTransport(HttpClient& client) : client_(client) {}
  bool readRaw(uint8_t* out, size_t length) {
    uint32_t progress = millis();
    while (length) {
      const int available = client_.available();
      if (available > 0) {
        const size_t want = length < static_cast<size_t>(available) ? length : available;
        const int count = client_.read(out, want);
        if (count > 0) { out += count; length -= count; progress = millis(); continue; }
      }
      // Drain buffered bytes even when TCP has already received FIN.
      if (!client_.connected() || !networkAvailable() || millis() - progress > 3000) return false;
      delay(1);
    }
    return true;
  }
  bool readLine(char* out, size_t capacity) {
    size_t length = 0;
    uint8_t byte;
    const uint32_t start = millis();
    while (length + 1 < capacity && millis() - start < 3000) {
      if (!readRaw(&byte, 1)) return false;
      if (byte == '\n') {
        if (!length || out[length - 1] != '\r') return false;
        out[length - 1] = 0;
        return true;
      }
      out[length++] = byte;
    }
    return false;
  }
private:
  HttpClient& client_;
};
static void playStream(HttpClient& client) {
  client.print(String("GET /stream.wav HTTP/1.1\r\nHost: ") + NAVGUIDE_SERVER_HOST + ":" + NAVGUIDE_SERVER_PORT + "\r\nConnection: close\r\nAccept: audio/wav\r\n");
  if (strlen(NAVGUIDE_DEVICE_TOKEN)) client.print(String("Authorization: Bearer ") + NAVGUIDE_DEVICE_TOKEN + "\r\n");
  client.print("\r\n");
  HttpTransport transport(client);
  char line[1024];
  if (!transport.readLine(line, sizeof(line))) return;
  if (strncmp(line, "HTTP/1.1 200 ", 13) && strncmp(line, "HTTP/1.0 200 ", 13)) {
    Serial.println("[SPK] HTTP response is not 200"); return;
  }
  bool chunked = false, complete = false;
  size_t headers = 0;
  while (headers < 16384 && transport.readLine(line, sizeof(line))) {
    if (!line[0]) { complete = true; break; }
    headers += strlen(line) + 2;
    String header(line); header.toLowerCase();
    if (header.startsWith("transfer-encoding:")) {
      String encoding = header.substring(18); encoding.trim();
      if (encoding != "chunked") return;
      chunked = true;
    }
    if (header.startsWith("content-encoding:") && !header.endsWith("identity")) return;
  }
  if (!complete) return;
  navguide::HttpBodyReader<HttpTransport> body(transport, chunked);
  navguide::WavInfo wav;
  if (!navguide::readWavHeader(body, wav)) { Serial.println("[SPK] unsupported WAV"); return; }
  if (speaker.txSampleRate() != wav.sampleRate &&
      !speaker.configureTX(wav.sampleRate, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO)) return;
  Serial.printf("[SPK] PCM mono 16bit %lu Hz\n", (unsigned long)wav.sampleRate);
  static uint8_t pcm[1920]; // Maximum 20 ms at 48 kHz.
  static int32_t stereo[1920];
  while (networkAvailable() && wav.dataBytes) {
    size_t bytes = (wav.sampleRate / 50) * 2;
    if (bytes > wav.dataBytes) bytes = wav.dataBytes;
    if (bytes > sizeof(pcm) || !body.read(pcm, bytes)) return;
    wav.dataBytes -= bytes;
    for (size_t i = 0; i < bytes / 2; ++i) {
      const int16_t sample = static_cast<int16_t>(navguide::le16(pcm + 2 * i));
      stereo[2 * i] = stereo[2 * i + 1] = navguide::pcm16ToSlot32(sample, NAVGUIDE_SPEAKER_GAIN);
    }
    size_t offset = 0, outputBytes = bytes * 4;
    const uint32_t start = millis();
    while (offset < outputBytes) {
      const size_t written = speaker.write(reinterpret_cast<uint8_t*>(stereo) + offset, outputBytes - offset);
      offset += written;
      if (millis() - start > 1000 || !networkAvailable()) return;
      if (!written) delay(1);
    }
  }
}
static void speakerTask(void*) {
  for (;;) {
    if (!networkAvailable()) { delay(100); continue; }
    HttpClient client;
#if NAVGUIDE_TLS
    client.setCACert(NAVGUIDE_ROOT_CA);
    client.setHandshakeTimeout(10);
#endif
    client.setTimeout(3000);
    if (client.connect(NAVGUIDE_SERVER_HOST, NAVGUIDE_SERVER_PORT, 3000)) playStream(client);
    client.stop();
    delay(kRetryMs);
  }
}
static void imuTask(void*) {
  SPI.begin(NAVGUIDE_IMU_SCK, NAVGUIDE_IMU_MISO, NAVGUIDE_IMU_MOSI, NAVGUIDE_IMU_CS);
  ImuDriver sensor(SPI, NAVGUIDE_IMU_CS);
  WiFiUDP udp;
  bool initialized = false, filtered = false;
  float gravity[3] = {}, bias[3] = {};
  uint32_t sequence = 0, badReads = 0;
  uint32_t stationarySince = 0;
  TickType_t tick = xTaskGetTickCount();
  for (;;) {
    if (!initialized) {
      initialized = sensor.begin() == 0;
      if (!initialized) { Serial.println("[IMU] ICM42688-P not ready; retry"); delay(2000); tick = xTaskGetTickCount(); continue; }
      filtered = false; badReads = 0; stationarySince = 0;
      Serial.println("[IMU] 50 Hz, accel m/s^2, gyro and yaw deg/s");
    }
    const int result = sensor.readSensor();
    if (result != 0) {
      if (++badReads > 50) initialized = false;
      vTaskDelayUntil(&tick, pdMS_TO_TICKS(20)); continue;
    }
    badReads = 0;
    const float acceleration[] = {sensor.getAccelX_mss(), sensor.getAccelY_mss(), sensor.getAccelZ_mss()};
    const float gyro[] = {sensor.getGyroX_dps(), sensor.getGyroY_dps(), sensor.getGyroZ_dps()};
    float a2 = 0, w2 = 0;
    for (int i = 0; i < 3; ++i) {
      gravity[i] = filtered ? gravity[i] * 0.98f + acceleration[i] * 0.02f : acceleration[i];
      a2 += acceleration[i] * acceleration[i]; w2 += gyro[i] * gyro[i];
    }
    filtered = true;
    const bool stationary = fabsf(sqrtf(a2) - navguide::kGravity) < 0.08f * navguide::kGravity && sqrtf(w2) < 0.4f;
    if (!stationary) stationarySince = 0;
    else if (!stationarySince) stationarySince = millis();
    else if (millis() - stationarySince >= 350) {
      for (int i = 0; i < 3; ++i) bias[i] = bias[i] * 0.998f + gyro[i] * 0.002f;
    }
    const float corrected[] = {gyro[0] - bias[0], gyro[1] - bias[1], gyro[2] - bias[2]};
    const float yawRate = navguide::yawRateDps(corrected, gravity);
    if (networkReady.load()) {
      char packet[640];
      const unsigned long long ts = static_cast<unsigned long long>(esp_timer_get_time() / 1000);
      const int count = snprintf(packet, sizeof(packet),
        "{\"schema_version\":1,\"seq\":%lu,\"ts\":%llu,\"temp_c\":%.2f,"
        "\"accel\":{\"x\":%.4f,\"y\":%.4f,\"z\":%.4f},"
        "\"gyro\":{\"x\":%.4f,\"y\":%.4f,\"z\":%.4f},\"yaw_rate_dps\":%.4f,\"token\":\"%s\"}",
        (unsigned long)sequence++, ts, sensor.getTemperature_C(),
        acceleration[0], acceleration[1], acceleration[2], gyro[0], gyro[1], gyro[2], yawRate, NAVGUIDE_DEVICE_TOKEN);
      if (count > 0 && static_cast<size_t>(count) < sizeof(packet) && udp.beginPacket(NAVGUIDE_UDP_HOST, NAVGUIDE_UDP_PORT)) {
        udp.write(reinterpret_cast<const uint8_t*>(packet), count);
        udp.endPacket();
      }
    }
    vTaskDelayUntil(&tick, pdMS_TO_TICKS(20));
  }
}
static void createTask(TaskFunction_t task, const char* name, uint32_t stack, UBaseType_t priority, BaseType_t core) {
  if (xTaskCreatePinnedToCore(task, name, stack, nullptr, priority, nullptr, core) != pdPASS) failBoot("[BOOT] task allocation failed");
}
void setup() {
  Serial.begin(115200); delay(300);
  Serial.println("NavGuide firmware 1.0");
  if (!strlen(NAVGUIDE_WIFI_SSID) || !strlen(NAVGUIDE_SERVER_HOST)) {
    Serial.println("[CONFIG] Set WiFi and service hostname in firmware/config.h before flashing"); return;
  }
  if (!navguide::validToken(NAVGUIDE_DEVICE_TOKEN)) failBoot("[CONFIG] token must contain 0..128 letters, digits, underscores, dots or hyphens");
#if NAVGUIDE_TLS
  if (!strlen(NAVGUIDE_ROOT_CA)) failBoot("[CONFIG] TLS requires a trusted root CA");
#endif
  if (!initCamera()) failBoot("[CAM] check camera ribbon, PSRAM and board selection");
  microphone.setPinsPdmRx(42, 41);
  if (!microphone.begin(I2S_MODE_PDM_RX, 16000, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) failBoot("[MIC] initialization failed");
  speaker.setPins(NAVGUIDE_SPEAKER_BCLK, NAVGUIDE_SPEAKER_LRCK, NAVGUIDE_SPEAKER_DIN);
  if (!speaker.begin(I2S_MODE_STD, 8000, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO)) failBoot("[SPK] initialization failed");
  audioQueue = xQueueCreate(6, sizeof(AudioChunk));
  if (!audioQueue) failBoot("[BOOT] audio queue allocation failed");
  WiFi.persistent(false); WiFi.mode(WIFI_STA); WiFi.setSleep(false); WiFi.setAutoReconnect(true);
  WiFi.begin(NAVGUIDE_WIFI_SSID, NAVGUIDE_WIFI_PASS);
#if NAVGUIDE_TLS
  configTime(0, 0, "pool.ntp.org", "time.nist.gov");
#endif
  createTask(cameraTask, "camera", 8192, 2, 1);
  createTask(microphoneTask, "microphone", 4096, 3, 0);
  createTask(audioTask, "audio_ws", 8192, 2, 1);
  createTask(speakerTask, "speaker", 8192, 2, 0);
#if NAVGUIDE_IMU_ENABLED
  createTask(imuTask, "imu", 4096, 2, 0);
#endif
  configured = true;
}
void loop() {
  if (!configured) { delay(1000); return; }
  static uint32_t lastRetry = millis(), lastLog = millis();
  const bool connected = WiFi.status() == WL_CONNECTED;
  const bool previous = networkReady.exchange(connected);
  if (connected && !previous) Serial.println(String("[WiFi] connected ") + WiFi.localIP().toString());
  if (!connected && previous) Serial.println("[WiFi] disconnected");
  if (!connected && millis() - lastRetry >= 15000) { WiFi.reconnect(); lastRetry = millis(); }
  if (millis() - lastLog >= 30000) {
    Serial.printf("[HEALTH] heap=%u min_heap=%u psram=%u microphone_drops=%lu\n", ESP.getFreeHeap(), ESP.getMinFreeHeap(), ESP.getFreePsram(), (unsigned long)audioDropped.load());
    lastLog = millis();
  }
  delay(20);
}
