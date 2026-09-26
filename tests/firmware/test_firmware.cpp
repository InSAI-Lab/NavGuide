#include "../../firmware/sensor_math.h"
#include "../../firmware/microphone_session.h"
#include "../../firmware/audio_stream.h"
#include <cassert>
#include <climits>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct BufferTransport {
  std::vector<uint8_t> bytes;
  size_t position = 0;
  bool readRaw(uint8_t* out, size_t n) {
    if (n > bytes.size() - position) return false;
    // Model a socket providing individual bytes; caller cannot assume packet boundaries.
    for (size_t i = 0; i < n; ++i) out[i] = bytes[position++];
    return true;
  }
  bool readLine(char* out, size_t capacity) {
    size_t n = 0;
    while (position < bytes.size() && n + 1 < capacity) {
      const char c = bytes[position++];
      if (c == '\n') {
        if (!n || out[n - 1] != '\r') return false;
        out[n - 1] = 0; return true;
      }
      out[n++] = c;
    }
    return false;
  }
};
static void put16(std::vector<uint8_t>& out, uint16_t v) { out.push_back(v & 255); out.push_back(v >> 8); }
static void put32(std::vector<uint8_t>& out, uint32_t v) { put16(out, v & 65535); put16(out, v >> 16); }
static void text(std::vector<uint8_t>& out, const std::string& s) { out.insert(out.end(), s.begin(), s.end()); }
static std::vector<uint8_t> fixture(uint32_t rate = 8000, bool padding = true) {
  std::vector<uint8_t> bytes;
  text(bytes, "RIFF"); put32(bytes, 0x7ffffff0); text(bytes, "WAVE");
  if (padding) { text(bytes, "JUNK"); put32(bytes, 3); text(bytes, "abc"); bytes.push_back(0); }
  text(bytes, "fmt "); put32(bytes, 16); put16(bytes, 1); put16(bytes, 1);
  put32(bytes, rate); put32(bytes, rate * 2); put16(bytes, 2); put16(bytes, 16);
  text(bytes, "data"); put32(bytes, 6); put16(bytes, 0x8000); put16(bytes, 0xffff); put16(bytes, 0x7fff);
  return bytes;
}
static std::vector<uint8_t> chunk(const std::vector<uint8_t>& input, size_t size) {
  std::vector<uint8_t> output;
  for (size_t i = 0; i < input.size(); i += size) {
    const size_t n = std::min(size, input.size() - i);
    std::ostringstream line; line << std::hex << n << ";extension=yes\r\n";
    text(output, line.str()); output.insert(output.end(), input.begin() + i, input.begin() + i + n); text(output, "\r\n");
  }
  text(output, "0\r\n\r\n"); return output;
}
static void testWav(bool chunked, size_t chunkSize, uint32_t rate) {
  auto input = fixture(rate);
  BufferTransport transport{chunked ? chunk(input, chunkSize) : input};
  navguide::HttpBodyReader<BufferTransport> reader(transport, chunked);
  navguide::WavInfo info;
  assert(navguide::readWavHeader(reader, info)); assert(info.sampleRate == rate); assert(info.dataBytes == 6);
  uint8_t samples[6]; assert(reader.read(samples, sizeof(samples)));
  assert(static_cast<int16_t>(navguide::le16(samples)) == -32768);
  assert(static_cast<int16_t>(navguide::le16(samples + 2)) == -1);
  assert(static_cast<int16_t>(navguide::le16(samples + 4)) == 32767);
  uint8_t extra; assert(!reader.read(&extra, 1));
}
static void testMicrophoneSession() {
  using Session = navguide::MicrophoneSession;
  Session session(100);
  assert(session.canConnect(100));
  assert(!session.streaming());
  session.startSent(100);
  assert(!session.streaming()); // Sending START is not its acknowledgement.
  assert(!session.canConnect(101));
  assert(!session.onControl("OK:PROMPT_ACCEPTED", 101));
  assert(session.onControl("OK:STARTED", 102));
  assert(session.streaming());
  assert(!session.onControl("ERR:UNKNOWN_COMMAND", 103));
  assert(session.streaming());
  assert(session.onControl("ERR:ASR_DISABLED", 104));
  assert(!session.streaming());
  assert(!session.canConnect(104 + Session::kDisabledRetryMs - 1));
  session.connectionLost(105);
  session.onControl("ERR:ASR_AUDIO_REJECTED", 106);
  session.onControl("OK:STARTED", 107); // Late acknowledgement cannot resume PCM.
  session.onControl("OK:STOPPED", 108);
  assert(!session.streaming());
  assert(!session.canConnect(104 + Session::kDisabledRetryMs - 1));
  assert(session.canConnect(104 + Session::kDisabledRetryMs));
  for (const char* error : {"ERR:ASR_UNAVAILABLE", "ERR:ASR_AUDIO_REJECTED", "RESTART", "ERROR:SDK_FAILURE"}) {
    session.startSent(70000);
    session.onControl("OK:STARTED", 70001);
    assert(session.streaming());
    assert(session.onControl(error, 130000)); // Includes a 60-second cloud session expiry.
    assert(!session.streaming());
    assert(!session.canConnect(130000 + Session::kRetryMs - 1));
    assert(session.canConnect(130000 + Session::kRetryMs));
  }
  session.startSent(140000);
  session.tick(140000 + Session::kStartTimeoutMs - 1);
  assert(!session.retrying());
  session.tick(140000 + Session::kStartTimeoutMs);
  assert(session.retrying());
  assert(!session.streaming());
  assert(session.canConnect(140000 + Session::kStartTimeoutMs + Session::kRetryMs));
  session.startSent(160000);
  session.onControl("OK:STARTED", 160001);
  session.onControl("OK:STOPPED", 160002);
  session.tick(200000);
  assert(!session.streaming() && !session.retrying());
  session.onControl("OK:STARTED", 200001);
  assert(!session.streaming());
  session.onControl("RESTART", 200002);
  assert(session.canConnect(202002));
  Session rollover(UINT32_MAX - 100);
  rollover.startSent(UINT32_MAX - 100);
  rollover.onControl("ERR:ASR_UNAVAILABLE", UINT32_MAX - 50);
  assert(!rollover.canConnect(100));
  assert(rollover.canConnect(1949));
  rollover.startSent(UINT32_MAX - 100);
  rollover.tick(14900);
  assert(rollover.retrying());
}
int main() {
  testMicrophoneSession();
  assert(std::fabs(navguide::accelMss(2048) - 9.80665f) < 0.0001f);
  assert(navguide::gyroDps(16384) == 1000.0f);
  assert(navguide::temperatureC(0) == 25.0f);
  const float gravityZ[] = {0, 0, 9.80665f}, gyroZ[] = {3, 4, 30};
  const float gravityY[] = {0, 9.80665f, 0}, gyroY[] = {3, 30, 4};
  const float gravityNegative[] = {0, 0, -9.80665f}, zero[] = {0, 0, 0};
  assert(std::fabs(navguide::yawRateDps(gyroZ, gravityZ) - 30) < 0.0001f);
  assert(std::fabs(navguide::yawRateDps(gyroY, gravityY) - 30) < 0.0001f);
  assert(std::fabs(navguide::yawRateDps(gyroZ, gravityNegative) + 30) < 0.0001f);
  assert(navguide::yawRateDps(gyroZ, zero) == 0);
  assert(navguide::pcm16ToSlot32(-32768, 1) == INT32_MIN);
  assert(navguide::pcm16ToSlot32(-1, 1) == -65536);
  assert(navguide::pcm16ToSlot32(32767, 1) == 2147418112);
  assert(navguide::pcm16ToSlot32(1234, -1) == 0);
  assert(navguide::validToken("abc_DEF-123.456"));
  assert(!navguide::validToken("bad\"token")); assert(!navguide::validToken("bad\r\nheader"));
  for (uint32_t rate : {8000u, 16000u, 24000u, 48000u}) {
    testWav(false, 1, rate);
    for (size_t size : {1u, 2u, 3u, 7u, 44u, 128u}) testWav(true, size, rate);
  }
  for (uint32_t badRate : {0u, 7999u, 48001u, 192000u}) {
    BufferTransport t{fixture(badRate)};
    navguide::HttpBodyReader<BufferTransport> reader(t, false); navguide::WavInfo info;
    assert(!navguide::readWavHeader(reader, info));
  }
  auto invalid = fixture(16000, false); invalid[22] = 2; // stereo is unsupported
  BufferTransport stereo{invalid}; navguide::HttpBodyReader<BufferTransport> stereoReader(stereo, false); navguide::WavInfo info;
  assert(!navguide::readWavHeader(stereoReader, info));
  for (size_t n = 0; n < 44; ++n) {
    auto shortWav = fixture(16000, false); shortWav.resize(n);
    BufferTransport t{shortWav}; navguide::HttpBodyReader<BufferTransport> reader(t, false);
    assert(!navguide::readWavHeader(reader, info));
  }
  for (const auto& corrupt : {"xyz\r\n", "1\r\naXX", "1000000000000000000000000000000\r\n"}) {
    std::vector<uint8_t> bytes; text(bytes, corrupt);
    BufferTransport t{bytes}; navguide::HttpBodyReader<BufferTransport> reader(t, true); uint8_t out;
    assert(!reader.read(&out, 1));
  }
  std::cout << "Firmware ASR handshake, signal, PCM, WAV, and HTTP chunk boundary tests passed\n";
}
