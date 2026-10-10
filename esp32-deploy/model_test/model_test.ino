#include "esp32_transformer.h"

struct TestCase {
  const char* label;
  float hr, rr, spo2, tempC;
};

TestCase cases[] = {
  { "Normal",   72,  16, 98, 36.8 },
  { "Elevated", 95,  22, 95, 38.3 },
  { "High",     118, 23, 92, 39.2 },
  { "Critical", 145, 30, 85, 39.6 },
};
const int NUM_CASES = 4;

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("=== Transformer model test running on ESP32 ===");
  Serial.println("(Using hardcoded sample vitals -- real sensors come next)");
  Serial.println();
}

void loop() {
  for (int i = 0; i < NUM_CASES; i++) {
    TestCase c = cases[i];
    float logits[TFM_N_CLASSES];

    unsigned long startMicros = micros();
    int predictedClass = tfmPredict(c.hr, c.rr, c.spo2, c.tempC, logits);
    unsigned long elapsedMicros = micros() - startMicros;

    Serial.print("Test case: ");
    Serial.println(c.label);
    Serial.print("  Input -> HR=");
    Serial.print(c.hr);
    Serial.print(" RR=");
    Serial.print(c.rr);
    Serial.print(" SpO2=");
    Serial.print(c.spo2);
    Serial.print(" Temp=");
    Serial.println(c.tempC);

    Serial.print("  Model says: ");
    Serial.print(TFM_CLASS_NAMES[predictedClass]);
    Serial.print("  (inference took ");
    Serial.print(elapsedMicros);
    Serial.println(" microseconds)");
    Serial.println();

    delay(2000);
  }
  Serial.println("--- cycle complete, repeating in 5 seconds ---");
  Serial.println();
  delay(5000);
}
