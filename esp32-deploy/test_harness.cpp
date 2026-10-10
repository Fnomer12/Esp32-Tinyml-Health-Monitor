#include <cstdio>
#include "esp32_transformer.h"

void runCase(const char* label, float hr, float rr, float spo2, float t) {
  float logits[TFM_N_CLASSES];
  int cls = tfmPredict(hr, rr, spo2, t, logits);
  printf("%-10s HR=%.1f RR=%.1f SpO2=%.1f T=%.1f -> class=%d (%s) logits=[%.5f, %.5f, %.5f, %.5f]\n",
         label, hr, rr, spo2, t, cls, TFM_CLASS_NAMES[cls],
         logits[0], logits[1], logits[2], logits[3]);
}

int main() {
  runCase("normal",   72,  16, 98, 36.8f);
  runCase("warning",  95,  22, 95, 38.3f);
  runCase("high",     118, 23, 92, 39.2f);
  runCase("critical", 145, 30, 85, 39.6f);
  return 0;
}
