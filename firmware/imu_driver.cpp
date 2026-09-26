#include "imu_driver.h"
#include "sensor_math.h"

ImuDriver::ImuDriver(TwoWire& bus, uint8_t address) : wire_(&bus), address_(address) {}
ImuDriver::ImuDriver(SPIClass& bus, uint8_t cs) : spi_(&bus), cs_(cs) {}

bool ImuDriver::readRegisters(uint8_t reg, uint8_t count, uint8_t* data) {
  if (spi_) {
    spi_->beginTransaction(SPISettings(1000000, MSBFIRST, SPI_MODE0));
    digitalWrite(cs_, LOW);
    spi_->transfer(reg | 0x80);
    for (uint8_t i = 0; i < count; ++i) data[i] = spi_->transfer(0);
    digitalWrite(cs_, HIGH);
    spi_->endTransaction();
    return true;
  }
  wire_->beginTransmission(address_);
  wire_->write(reg);
  if (wire_->endTransmission(false) != 0) return false;
  if (wire_->requestFrom(address_, count) != count) return false;
  for (uint8_t i = 0; i < count; ++i) data[i] = wire_->read();
  return true;
}
bool ImuDriver::writeRegister(uint8_t reg, uint8_t data) {
  if (spi_) {
    spi_->beginTransaction(SPISettings(1000000, MSBFIRST, SPI_MODE0));
    digitalWrite(cs_, LOW);
    spi_->transfer(reg & 0x7f);
    spi_->transfer(data);
    digitalWrite(cs_, HIGH);
    spi_->endTransaction();
    return true;
  }
  wire_->beginTransmission(address_);
  wire_->write(reg);
  wire_->write(data);
  return wire_->endTransmission() == 0;
}
int ImuDriver::begin() {
  if (spi_) { pinMode(cs_, OUTPUT); digitalWrite(cs_, HIGH); }
  uint8_t who = 0;
  if (!writeRegister(0x76, 0) || !readRegisters(0x75, 1, &who) || who != 0x47) return -1;
  if (!writeRegister(0x11, 0x01)) return -1; // DEVICE_CONFIG soft reset
  delay(2);
  if (!writeRegister(0x76, 0) || !writeRegister(0x4e, 0)) return -1;
  // GYRO_CONFIG0=0x4f; ACCEL_CONFIG0=0x50. FS=0, ODR=9 means 50 Hz.
  if (!writeRegister(0x4f, 0x09) || !writeRegister(0x50, 0x09)) return -1;
  if (!writeRegister(0x4e, 0x0f)) return -1; // Both sensors in low noise mode
  delay(50); // Gyroscope startup is 45 ms; no writes during first 200 us.
  uint8_t check[3] = {};
  if (!readRegisters(0x4e, 3, check)) return -1;
  return check[0] == 0x0f && check[1] == 0x09 && check[2] == 0x09 ? 0 : -1;
}
int ImuDriver::readSensor() {
  uint8_t who = 0, status = 0, data[14];
  if (!readRegisters(0x75, 1, &who) || who != 0x47) return -1;
  if (!readRegisters(0x2d, 1, &status)) return -1;
  if (!(status & 0x08)) return 1;
  if (!readRegisters(0x1d, sizeof(data), data)) return -1;
  int16_t fresh[7];
  for (uint8_t i = 0; i < 7; ++i) {
    fresh[i] = static_cast<int16_t>((uint16_t(data[2 * i]) << 8) | data[2 * i + 1]);
    if (fresh[i] == INT16_MIN) return -1; // Invalid sensor-data sentinel
  }
  memcpy(sample_, fresh, sizeof(fresh));
  return 0;
}
float ImuDriver::getAccelX_mss() const { return navguide::accelMss(sample_[1]); }
float ImuDriver::getAccelY_mss() const { return navguide::accelMss(sample_[2]); }
float ImuDriver::getAccelZ_mss() const { return navguide::accelMss(sample_[3]); }
float ImuDriver::getGyroX_dps() const { return navguide::gyroDps(sample_[4]); }
float ImuDriver::getGyroY_dps() const { return navguide::gyroDps(sample_[5]); }
float ImuDriver::getGyroZ_dps() const { return navguide::gyroDps(sample_[6]); }
float ImuDriver::getGyroX_rads() const { return getGyroX_dps() * 0.01745329252f; }
float ImuDriver::getGyroY_rads() const { return getGyroY_dps() * 0.01745329252f; }
float ImuDriver::getGyroZ_rads() const { return getGyroZ_dps() * 0.01745329252f; }
float ImuDriver::getTemperature_C() const { return navguide::temperatureC(sample_[0]); }
