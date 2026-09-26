#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>

namespace navguide {
inline uint16_t le16(const uint8_t* p) { return uint16_t(p[0]) | (uint16_t(p[1]) << 8); }
inline uint32_t le32(const uint8_t* p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
// Transport::readRaw must keep partial reads until its deadline; no discarded prefetch.
// Transport::readLine(char*, size_t) returns one CRLF-terminated line without CRLF.
template <class Transport> class HttpBodyReader {
public:
  HttpBodyReader(Transport& transport, bool chunked) : transport_(transport), chunked_(chunked) {}
  bool read(uint8_t* out, size_t length) {
    if (!chunked_) return transport_.readRaw(out, length);
    while (length) {
      if (done_) return false;
      if (!left_) {
        char line[96];
        if (!transport_.readLine(line, sizeof(line))) return false;
        size_t size = 0, digits = 0;
        for (const char* p = line; *p && *p != ';'; ++p) {
          unsigned int nibble;
          if (*p >= '0' && *p <= '9') nibble = *p - '0';
          else if (*p >= 'a' && *p <= 'f') nibble = *p - 'a' + 10;
          else if (*p >= 'A' && *p <= 'F') nibble = *p - 'A' + 10;
          else return false;
          if (size > (SIZE_MAX - nibble) / 16) return false;
          size = size * 16 + nibble;
          ++digits;
        }
        if (!digits) return false;
        left_ = size;
        if (!left_) { done_ = true; return false; }
      }
      const size_t take = length < left_ ? length : left_;
      if (!transport_.readRaw(out, take)) return false;
      out += take; length -= take; left_ -= take;
      if (!left_) {
        uint8_t crlf[2];
        if (!transport_.readRaw(crlf, 2) || crlf[0] != '\r' || crlf[1] != '\n') return false;
      }
    }
    return true;
  }
private:
  Transport& transport_;
  bool chunked_, done_ = false;
  size_t left_ = 0;
};
struct WavInfo { uint32_t sampleRate = 0, dataBytes = 0; };
template <class Reader> bool readWavHeader(Reader& reader, WavInfo& info) {
  uint8_t header[16];
  if (!reader.read(header, 12) || std::memcmp(header, "RIFF", 4) || std::memcmp(header + 8, "WAVE", 4)) return false;
  bool haveFormat = false;
  uint32_t scanned = 12;
  while (scanned < 16384) {
    if (!reader.read(header, 8)) return false;
    scanned += 8;
    const uint32_t bytes = le32(header + 4);
    if (std::memcmp(header, "data", 4) == 0) {
      if (!haveFormat || bytes % 2) return false;
      info.dataBytes = bytes;
      return true;
    }
    const bool isFormat = std::memcmp(header, "fmt ", 4) == 0;
    if (bytes > 16384 - scanned || (isFormat && bytes < 16)) return false;
    uint32_t remaining = bytes;
    if (isFormat) {
      if (!reader.read(header, 16)) return false;
      info.sampleRate = le32(header + 4);
      if (le16(header) != 1 || le16(header + 2) != 1 || le16(header + 14) != 16 ||
          le16(header + 12) != 2 || le32(header + 8) != info.sampleRate * 2 ||
          info.sampleRate < 8000 || info.sampleRate > 48000) return false;
      remaining -= 16;
      haveFormat = true;
    }
    uint8_t discard[64];
    while (remaining) {
      const size_t n = remaining < sizeof(discard) ? remaining : sizeof(discard);
      if (!reader.read(discard, n)) return false;
      remaining -= n;
    }
    if ((bytes & 1) && !reader.read(discard, 1)) return false;
    scanned += bytes + (bytes & 1);
  }
  return false;
}
}
