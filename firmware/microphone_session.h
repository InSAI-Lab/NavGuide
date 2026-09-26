#pragma once
#include <cstdint>
#include <cstring>

namespace navguide {
// Control protocol for the optional ASR service. A successful TCP or WebSocket
// connection alone never authorizes PCM upload: the server must acknowledge START.
class MicrophoneSession {
public:
  static constexpr uint32_t kRetryMs = 2000;
  static constexpr uint32_t kDisabledRetryMs = 60000;
  static constexpr uint32_t kStartTimeoutMs = 15000;
  explicit MicrophoneSession(uint32_t now = 0) : retryAt_(now) {}

  bool streaming() const { return state_ == State::Streaming; }
  bool retrying() const { return state_ == State::Retry; }
  bool canConnect(uint32_t now) const {
    return retrying() && static_cast<int32_t>(now - retryAt_) >= 0;
  }
  void startSent(uint32_t now) { state_ = State::AwaitingStart; startAt_ = now; }
  void connectionLost(uint32_t now) { defer(now, kRetryMs); }
  void tick(uint32_t now) {
    if (state_ == State::AwaitingStart && now - startAt_ >= kStartTimeoutMs) defer(now, kRetryMs);
  }
  bool onControl(const char* message, uint32_t now) {
    if (std::strcmp(message, "OK:STARTED") == 0) {
      // Ignore late acknowledgements from a session already scheduled for retry.
      if (state_ == State::AwaitingStart) state_ = State::Streaming;
    } else if (std::strcmp(message, "OK:STOPPED") == 0) {
      if (!retrying()) state_ = State::Stopped;
    } else if (std::strcmp(message, "ERR:ASR_DISABLED") == 0) {
      defer(now, kDisabledRetryMs);
    } else if (std::strcmp(message, "RESTART") == 0 ||
               std::strncmp(message, "ERR:ASR_", 8) == 0 ||
               std::strcmp(message, "ERROR") == 0 || std::strncmp(message, "ERROR:", 6) == 0) {
      defer(now, kRetryMs);
    } else {
      // ERR:UNKNOWN_COMMAND is a navigation command result, not an ASR failure.
      return false;
    }
    return true;
  }
private:
  enum class State { Retry, AwaitingStart, Streaming, Stopped };
  State state_ = State::Retry;
  uint32_t retryAt_ = 0, startAt_ = 0;
  void defer(uint32_t now, uint32_t interval) {
    // Network failure or a second queued error must not shorten a disabled backoff.
    if (retrying() && static_cast<int32_t>(retryAt_ - now) > 0 && retryAt_ - now > interval) return;
    state_ = State::Retry;
    retryAt_ = now + interval;
  }
};
}
