#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace navguide {
constexpr float kGravity = 9.80665f;
inline float accelMss(int16_t raw) { return raw * (16.0f * kGravity / 32768.0f); }
inline float gyroDps(int16_t raw) { return raw * (2000.0f / 32768.0f); }
inline float temperatureC(int16_t raw) { return raw / 132.48f + 25.0f; }
// Projection is invariant to a rigid rotation of the sensor mounting axes.
// This is a relative angular rate. It is not a magnetometer heading estimate.
inline float yawRateDps(const float gyro[3], const float gravity[3]) {
  const float norm = std::sqrt(gravity[0] * gravity[0] + gravity[1] * gravity[1] + gravity[2] * gravity[2]);
  if (!std::isfinite(norm) || norm < 0.1f) return 0.0f;
  return (gyro[0] * gravity[0] + gyro[1] * gravity[1] + gyro[2] * gravity[2]) / norm;
}
inline int32_t pcm16ToSlot32(int16_t sample, float gain) {
  if (!std::isfinite(gain)) gain = 0.0f;
  if (gain < 0.0f) gain = 0.0f;
  if (gain > 1.0f) gain = 1.0f;
  // Multiplication is defined for negative samples; a signed left shift is not.
  return static_cast<int32_t>(sample * gain) * 65536;
}
inline bool validToken(const char* token) {
  size_t n = 0;
  for (; token[n]; ++n) {
    const char c = token[n];
    if (n >= 128 || !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.')) return false;
  }
  return true;
}
}
