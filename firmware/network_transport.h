#pragma once
#include <ArduinoWebsockets.h>
#include <WiFiClient.h>
#include <WiFiClientSecure.h>
#include "config_defaults.h"
#if NAVGUIDE_TLS
using HttpClient = WiFiClientSecure;
#else
using HttpClient = WiFiClient;
#endif
// ArduinoWebsockets 0.5.4 assumes a TCP write succeeds. Closing on a partial
// write lets the owner detect failure and reconnect instead of corrupting frames.
class BoundedSocket : public websockets::network::GenericEspTcpClient<HttpClient> {
public:
  BoundedSocket() {
    client.setTimeout(3000);
#if NAVGUIDE_TLS
    client.setCACert(NAVGUIDE_ROOT_CA);
    client.setHandshakeTimeout(10);
#endif
  }
  bool connect(const websockets::WSString& host, int port) override {
    const bool ok = client.connect(host.c_str(), port, 3000);
    client.setNoDelay(true);
    return ok;
  }
  void send(const websockets::WSString& data) override { send(reinterpret_cast<const uint8_t*>(data.data()), data.size()); }
  void send(const websockets::WSString&& data) override { send(reinterpret_cast<const uint8_t*>(data.data()), data.size()); }
  void send(const uint8_t* data, uint32_t length) override {
    const uint32_t start = millis();
    while (length && client.connected()) {
      const size_t written = client.write(data, length);
      if (!written || millis() - start > 3000) { client.stop(); return; }
      data += written; length -= written;
      delay(0);
    }
    if (length) client.stop();
  }
  websockets::WSString readLine() override {
    websockets::WSString line;
    const uint32_t start = millis();
    while (client.connected() && millis() - start < 3000 && line.size() < 1024) {
      const int ch = client.read();
      if (ch < 0) { delay(1); continue; }
      line += static_cast<char>(ch);
      if (ch == '\n') return line;
    }
    client.stop();
    return "";
  }
  uint32_t read(uint8_t* data, uint32_t length) override {
    const int count = client.read(data, length);
    return count > 0 ? static_cast<uint32_t>(count) : 0;
  }
};
