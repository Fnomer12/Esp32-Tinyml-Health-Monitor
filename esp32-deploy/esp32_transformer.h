
#pragma once
#include <math.h>
#include "model_weights.h"

// Internal scratch buffers (fixed size, no heap allocation -- safe on ESP32)
static float _tfm_tok[TFM_N_TOKENS][TFM_D_MODEL];
static float _tfm_Q[TFM_N_TOKENS][TFM_D_MODEL];
static float _tfm_K[TFM_N_TOKENS][TFM_D_MODEL];
static float _tfm_V[TFM_N_TOKENS][TFM_D_MODEL];
static float _tfm_attnOut[TFM_N_TOKENS][TFM_D_MODEL];
static float _tfm_x1[TFM_N_TOKENS][TFM_D_MODEL];
static float _tfm_ffnHidden[TFM_N_TOKENS][TFM_FFN_HIDDEN];
static float _tfm_x2[TFM_N_TOKENS][TFM_D_MODEL];
static float _tfm_pooled[TFM_D_MODEL];
static float _tfm_scores[TFM_N_TOKENS][TFM_N_TOKENS];

// out[t][d'] = sum_d in[t][d] * W[d][d'] + (b ? b[d'] : 0)
static void _tfmLinear(const float in_[TFM_N_TOKENS][TFM_D_MODEL],
                        const float* W, const float* b,
                        float out[TFM_N_TOKENS][TFM_D_MODEL]) {
  for (int t = 0; t < TFM_N_TOKENS; t++) {
    for (int dp = 0; dp < TFM_D_MODEL; dp++) {
      float sum = b ? b[dp] : 0.0f;
      for (int d = 0; d < TFM_D_MODEL; d++) {
        sum += in_[t][d] * W[d * TFM_D_MODEL + dp];
      }
      out[t][dp] = sum;
    }
  }
}

static void _tfmLayerNorm(float x[TFM_N_TOKENS][TFM_D_MODEL],
                           const float* gamma, const float* beta) {
  for (int t = 0; t < TFM_N_TOKENS; t++) {
    float mean = 0.0f;
    for (int d = 0; d < TFM_D_MODEL; d++) mean += x[t][d];
    mean /= TFM_D_MODEL;
    float var = 0.0f;
    for (int d = 0; d < TFM_D_MODEL; d++) {
      float diff = x[t][d] - mean;
      var += diff * diff;
    }
    var /= TFM_D_MODEL;
    float invStd = 1.0f / sqrtf(var + 1e-5f);
    for (int d = 0; d < TFM_D_MODEL; d++) {
      x[t][d] = (x[t][d] - mean) * invStd * gamma[d] + beta[d];
    }
  }
}

// Runs the full forward pass. vitals[] must be in order: HR, RR, SpO2, Temperature_C
// Returns the predicted class index (0=Normal,1=Warning,2=High,3=Critical).
// If probsOut is non-NULL, writes the 4 raw logits there (not full softmax,
// but argmax is identical and this avoids an extra exp() pass on the ESP32).
static int tfmPredict(float hr, float rr, float spo2, float tempC, float* probsOut) {
  float rawVitals[TFM_N_TOKENS] = { hr, rr, spo2, tempC };
  float x[TFM_N_TOKENS];

  // 1. Normalize exactly like the Python StandardScaler did at training time
  for (int i = 0; i < TFM_N_TOKENS; i++) {
    x[i] = (rawVitals[i] - TFM_SCALER_MEAN[i]) / TFM_SCALER_SCALE[i];
  }

  // 2. Per-token embedding: tok[t][d] = x[t] * Wemb[t][d] + bemb[t][d]
  for (int t = 0; t < TFM_N_TOKENS; t++) {
    for (int d = 0; d < TFM_D_MODEL; d++) {
      _tfm_tok[t][d] = x[t] * TFM_Wemb[t * TFM_D_MODEL + d] + TFM_bemb[t * TFM_D_MODEL + d];
    }
  }

  // 3. Q, K, V projections
  _tfmLinear(_tfm_tok, TFM_Wq, nullptr, _tfm_Q);
  _tfmLinear(_tfm_tok, TFM_Wk, nullptr, _tfm_K);
  _tfmLinear(_tfm_tok, TFM_Wv, nullptr, _tfm_V);

  // 4. Multi-head self-attention (2 heads x 8 dims), head h owns columns
  //    [h*HEAD_DIM, (h+1)*HEAD_DIM) of the D_MODEL=16 axis.
  for (int h = 0; h < TFM_N_HEADS; h++) {
    int base = h * TFM_HEAD_DIM;

    // scores[t][s] = (Q_h[t] . K_h[s]) / sqrt(HEAD_DIM)
    for (int t = 0; t < TFM_N_TOKENS; t++) {
      for (int s = 0; s < TFM_N_TOKENS; s++) {
        float dot = 0.0f;
        for (int hd = 0; hd < TFM_HEAD_DIM; hd++) {
          dot += _tfm_Q[t][base + hd] * _tfm_K[s][base + hd];
        }
        _tfm_scores[t][s] = dot / sqrtf((float)TFM_HEAD_DIM);
      }
    }

    // softmax each row, then weighted sum of V_h -> attnOut columns [base, base+HEAD_DIM)
    for (int t = 0; t < TFM_N_TOKENS; t++) {
      float maxVal = _tfm_scores[t][0];
      for (int s = 1; s < TFM_N_TOKENS; s++) if (_tfm_scores[t][s] > maxVal) maxVal = _tfm_scores[t][s];
      float sumExp = 0.0f;
      float weights[TFM_N_TOKENS];
      for (int s = 0; s < TFM_N_TOKENS; s++) {
        weights[s] = expf(_tfm_scores[t][s] - maxVal);
        sumExp += weights[s];
      }
      for (int s = 0; s < TFM_N_TOKENS; s++) weights[s] /= sumExp;

      for (int hd = 0; hd < TFM_HEAD_DIM; hd++) {
        float sum = 0.0f;
        for (int s = 0; s < TFM_N_TOKENS; s++) {
          sum += weights[s] * _tfm_V[s][base + hd];
        }
        _tfm_attnOut[t][base + hd] = sum;
      }
    }
  }

  // 5. Output projection Wo, residual add, layer norm 1
  float attnProjected[TFM_N_TOKENS][TFM_D_MODEL];
  _tfmLinear(_tfm_attnOut, TFM_Wo, nullptr, attnProjected);
  for (int t = 0; t < TFM_N_TOKENS; t++)
    for (int d = 0; d < TFM_D_MODEL; d++)
      _tfm_x1[t][d] = _tfm_tok[t][d] + attnProjected[t][d];
  _tfmLayerNorm(_tfm_x1, TFM_gamma1, TFM_beta1);

  // 6. Feed-forward: relu(x1 @ W1 + b1) @ W2 + b2
  for (int t = 0; t < TFM_N_TOKENS; t++) {
    for (int f = 0; f < TFM_FFN_HIDDEN; f++) {
      float sum = TFM_b1[f];
      for (int d = 0; d < TFM_D_MODEL; d++) {
        sum += _tfm_x1[t][d] * TFM_W1[d * TFM_FFN_HIDDEN + f];
      }
      _tfm_ffnHidden[t][f] = sum > 0.0f ? sum : 0.0f;  // ReLU
    }
  }
  float ffnOut[TFM_N_TOKENS][TFM_D_MODEL];
  for (int t = 0; t < TFM_N_TOKENS; t++) {
    for (int d = 0; d < TFM_D_MODEL; d++) {
      float sum = TFM_b2[d];
      for (int f = 0; f < TFM_FFN_HIDDEN; f++) {
        sum += _tfm_ffnHidden[t][f] * TFM_W2[f * TFM_D_MODEL + d];
      }
      ffnOut[t][d] = sum;
    }
  }

  // 7. Residual add, layer norm 2
  for (int t = 0; t < TFM_N_TOKENS; t++)
    for (int d = 0; d < TFM_D_MODEL; d++)
      _tfm_x2[t][d] = _tfm_x1[t][d] + ffnOut[t][d];
  _tfmLayerNorm(_tfm_x2, TFM_gamma2, TFM_beta2);

  // 8. Mean-pool across the 4 tokens -> single D_MODEL vector
  for (int d = 0; d < TFM_D_MODEL; d++) {
    float sum = 0.0f;
    for (int t = 0; t < TFM_N_TOKENS; t++) sum += _tfm_x2[t][d];
    _tfm_pooled[d] = sum / TFM_N_TOKENS;
  }

  // 9. Classifier head -> 4 logits, argmax = predicted risk class
  float logits[TFM_N_CLASSES];
  for (int c = 0; c < TFM_N_CLASSES; c++) {
    float sum = TFM_bc[c];
    for (int d = 0; d < TFM_D_MODEL; d++) {
      sum += _tfm_pooled[d] * TFM_Wc[d * TFM_N_CLASSES + c];
    }
    logits[c] = sum;
  }

  if (probsOut) {
    for (int c = 0; c < TFM_N_CLASSES; c++) probsOut[c] = logits[c];
  }

  int best = 0;
  for (int c = 1; c < TFM_N_CLASSES; c++) {
    if (logits[c] > logits[best]) best = c;
  }
  return best;
}
