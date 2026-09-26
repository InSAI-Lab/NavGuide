#pragma once
#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>

// ICM-42688-P register driver, 50 Hz, ±16 g and ±2000 deg/s.
// Only the task that owns this instance may access its bus.
class ImuDriver {
public:
  ImuDriver(TwoWire& bus, uint8_t address);
  ImuDriver(SPIClass& bus, uint8_t cs);
  int begin();
  int readSensor(); // 0: fresh sample, 1: not ready, -1: transfer/sample error
  float getAccelX_mss() const;
  float getAccelY_mss() const;
  float getAccelZ_mss() const;
  float getGyroX_dps() const;
  float getGyroY_dps() const;
  float getGyroZ_dps() const;
  float getGyroX_rads() const;
  float getGyroY_rads() const;
  float getGyroZ_rads() const;
  float getTemperature_C() const;
private:
  TwoWire* wire_ = nullptr;
  SPIClass* spi_ = nullptr;
  uint8_t address_ = 0;
  uint8_t cs_ = 0;
  int16_t sample_[7] = {};
  bool writeRegister(uint8_t reg, uint8_t data);
  bool readRegisters(uint8_t reg, uint8_t count, uint8_t* data);
};
