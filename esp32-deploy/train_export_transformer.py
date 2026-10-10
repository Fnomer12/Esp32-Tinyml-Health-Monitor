"""
train_export_transformer.py
============================
Trains the SAME small hand-built Transformer architecture that already
exists in generate_news2_datasets.py (JAX, D_MODEL=16, 2 heads,
FFN_HIDDEN=64, 4 vital-sign "tokens"), on the SAME three real, cleaned
datasets (Cardiac Triage, NHAMCS 2021, NHAMCS 2022) -- but this time it
actually SAVES the trained weights and exports them as a plain C header
file so they can be hardcoded into the ESP32 firmware directly.

No TensorFlow / TFLite Micro needed: the network is tiny (~3,400
parameters, ~14KB as float32), so a hand-written C forward pass on the
ESP32 is simpler and more reliable than fighting a model-conversion
pipeline for something this small.

Run with: python3 train_export_transformer.py
Outputs (next to this script):
  - model_weights.h   <- C header with all trained weights + scaler stats
  - training_report.json <- accuracy + param counts, for the docs
"""

import os
import json
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import optax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Dataset", "Filtered_dataset"))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURES = ["HR", "RR", "SpO2", "Temperature_C"]
CLASSES = ["Normal", "Warning", "High", "Critical"]
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# EXACT official NEWS2 bands (Dec 2022 chart, SpO2 Scale 1) -- copied as-is
# from generate_news2_datasets.py so labeling is identical.
# ---------------------------------------------------------------------------
def hr_severity(hr):
    if 51 <= hr <= 90: return 0
    if (41 <= hr <= 50) or (91 <= hr <= 110): return 1
    if 111 <= hr <= 130: return 2
    return 3

def temp_severity(t):
    if 36.1 <= t <= 38.0: return 0
    if (35.1 <= t <= 36.0) or (38.1 <= t <= 39.0): return 1
    if t >= 39.1: return 2
    return 3

def spo2_severity(s):
    if s >= 96: return 0
    if 94 <= s <= 95: return 1
    if 92 <= s <= 93: return 2
    return 3

def rr_severity(rr):
    if 12 <= rr <= 20: return 0
    if 9 <= rr <= 11: return 1
    if 21 <= rr <= 24: return 2
    return 3

def label_df(df):
    df = df.copy()
    df["hr_sev"] = df["HR"].apply(hr_severity)
    df["temp_sev"] = df["Temperature_C"].apply(temp_severity)
    df["spo2_sev"] = df["SpO2"].apply(spo2_severity)
    df["rr_sev"] = df["RR"].apply(rr_severity)
    df["severity"] = df[["hr_sev", "temp_sev", "spo2_sev", "rr_sev"]].max(axis=1)
    df["label"] = df["severity"].map(dict(enumerate(CLASSES)))
    return df

# ---------------------------------------------------------------------------
# Load + combine all three real datasets (same files generate_news2_datasets.py uses)
# ---------------------------------------------------------------------------
print("Loading datasets...")
cardiac = pd.read_csv(os.path.join(DATASET_DIR, "CardiacPatientData_filtered.csv"))[["HR", "RR", "SpO2", "BT_C"]].rename(columns={"BT_C": "Temperature_C"})
nhamcs2021 = pd.read_csv(os.path.join(DATASET_DIR, "NHAMCS_2021_CLEAN.csv"))[["HR", "RR", "SpO2", "Temperature_C"]]
nhamcs2022 = pd.read_csv(os.path.join(DATASET_DIR, "NHAMCS_2022_CLEAN.csv"))[["HR", "RR", "SpO2", "Temperature_C"]]

combined = pd.concat([cardiac, nhamcs2021, nhamcs2022], ignore_index=True).dropna()
combined = label_df(combined)
print(f"  Cardiac Triage: {len(cardiac)} rows")
print(f"  NHAMCS 2021:    {len(nhamcs2021)} rows")
print(f"  NHAMCS 2022:    {len(nhamcs2022)} rows")
print(f"  Combined (clean): {len(combined)} rows")
print(combined["label"].value_counts())

X_all = combined[FEATURES].values.astype(np.float32)
y_all = np.array([CLASSES.index(v) for v in combined["label"].values])

# ---------------------------------------------------------------------------
# Scale features (ESP32 firmware must apply the SAME mean/scale to live
# sensor readings before calling the model -- that's why we export these)
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all).astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y_all
)

# ---------------------------------------------------------------------------
# The small hand-built Transformer -- identical architecture to
# generate_news2_datasets.py's _tfm_* functions.
# ---------------------------------------------------------------------------
D_MODEL, FFN_HIDDEN, N_HEADS = 16, 64, 2
HEAD_DIM = D_MODEL // N_HEADS
N_TOKENS = len(FEATURES)
N_CLASSES = len(CLASSES)
EPOCHS, BATCH_SIZE, LEARNING_RATE = 90, 1024, 3e-3
_tfm_optimizer = optax.adam(LEARNING_RATE)

def _tfm_init_params(key):
    keys = jax.random.split(key, 8)
    init = lambda k, shape, scale=0.15: jax.random.normal(k, shape) * scale
    return {
        "Wemb": init(keys[0], (N_TOKENS, D_MODEL)), "bemb": jnp.zeros((N_TOKENS, D_MODEL)),
        "Wq": init(keys[1], (D_MODEL, D_MODEL)), "Wk": init(keys[2], (D_MODEL, D_MODEL)),
        "Wv": init(keys[3], (D_MODEL, D_MODEL)), "Wo": init(keys[4], (D_MODEL, D_MODEL)),
        "gamma1": jnp.ones((D_MODEL,)), "beta1": jnp.zeros((D_MODEL,)),
        "W1": init(keys[5], (D_MODEL, FFN_HIDDEN)), "b1": jnp.zeros((FFN_HIDDEN,)),
        "W2": init(keys[6], (FFN_HIDDEN, D_MODEL)), "b2": jnp.zeros((D_MODEL,)),
        "gamma2": jnp.ones((D_MODEL,)), "beta2": jnp.zeros((D_MODEL,)),
        "Wc": init(keys[7], (D_MODEL, N_CLASSES)), "bc": jnp.zeros((N_CLASSES,)),
    }

def _tfm_count_params(params):
    return int(sum(np.prod(v.shape) for v in jax.tree_util.tree_leaves(params)))

def _tfm_layernorm(x, gamma, beta, eps=1e-5):
    mu = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + eps) * gamma + beta

def _tfm_forward(params, x):
    tok = x[:, :, None] * params["Wemb"][None, :, :] + params["bemb"][None, :, :]
    Q, K, V = tok @ params["Wq"], tok @ params["Wk"], tok @ params["Wv"]
    B, T, D = Q.shape
    split = lambda t: t.reshape(B, T, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    Qh, Kh, Vh = split(Q), split(K), split(V)
    scores = jnp.einsum("bhtd,bhsd->bhts", Qh, Kh) / jnp.sqrt(HEAD_DIM)
    attn = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bhts,bhsd->bhtd", attn, Vh).transpose(0, 2, 1, 3).reshape(B, T, D)
    out = out @ params["Wo"]
    x1 = _tfm_layernorm(tok + out, params["gamma1"], params["beta1"])
    ffn = jax.nn.relu(x1 @ params["W1"] + params["b1"]) @ params["W2"] + params["b2"]
    x2 = _tfm_layernorm(x1 + ffn, params["gamma2"], params["beta2"])
    pooled = jnp.mean(x2, axis=1)
    return pooled @ params["Wc"] + params["bc"]

def _tfm_loss(params, xb, yb):
    logits = _tfm_forward(params, xb)
    one_hot = jax.nn.one_hot(yb, N_CLASSES)
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(one_hot * logp, axis=-1))

def _tfm_train_step(params, opt_state, xb, yb):
    loss, grads = jax.value_and_grad(_tfm_loss)(params, xb, yb)
    updates, opt_state = _tfm_optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss

def _tfm_make_batches(X, y, batch_size, seed):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    Xs, ys = np.asarray(X)[perm], np.asarray(y)[perm]
    n_batches = max(1, n // batch_size)
    Xb = Xs[:n_batches * batch_size].reshape(n_batches, batch_size, -1)
    yb = ys[:n_batches * batch_size].reshape(n_batches, batch_size)
    return Xb, yb, n_batches

@jax.jit
def _tfm_run_scan(params, opt_state, Xb_all, yb_all):
    def step(carry, batch):
        params, opt_state = carry
        xb, yb = batch
        params, opt_state, loss = _tfm_train_step(params, opt_state, xb, yb)
        return (params, opt_state), loss
    (params, opt_state), losses = jax.lax.scan(step, (params, opt_state), (Xb_all, yb_all))
    return params, losses

def train_transformer(Xtr, ytr, seed=RANDOM_STATE):
    Xb, yb, n_batches = _tfm_make_batches(Xtr, ytr, min(BATCH_SIZE, len(Xtr)), seed)
    Xb_all = jnp.asarray(np.tile(Xb, (EPOCHS, 1, 1)), dtype=jnp.float32)
    yb_all = jnp.asarray(np.tile(yb, (EPOCHS, 1)), dtype=jnp.int32)
    key = jax.random.PRNGKey(seed)
    params = _tfm_init_params(key)
    opt_state = _tfm_optimizer.init(params)
    params, losses = _tfm_run_scan(params, opt_state, Xb_all, yb_all)
    return params

def predict_transformer(params, X):
    logits = _tfm_forward(params, jnp.asarray(X, dtype=jnp.float32))
    return np.array(jnp.argmax(logits, axis=-1))

# ---------------------------------------------------------------------------
# Train on the train split, report accuracy, then retrain on ALL data for
# the final deployed model (standard practice once you've validated it).
# ---------------------------------------------------------------------------
print("\nTraining transformer on train split (for accuracy check)...")
params_eval = train_transformer(X_train, y_train, seed=RANDOM_STATE)
y_pred = predict_transformer(params_eval, X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred).tolist()
print(f"Test accuracy: {acc*100:.2f}%")
print("Confusion matrix (rows=true, cols=pred):")
for row in cm:
    print(" ", row)

print("\nTraining final model on ALL data for deployment...")
final_params = train_transformer(X_scaled, y_all, seed=RANDOM_STATE)
param_count = _tfm_count_params(final_params)
print(f"Total parameters: {param_count}  (~{param_count*4/1024:.1f} KB as float32)")

# ---------------------------------------------------------------------------
# Export: C header with every weight matrix as a const float array, plus
# the StandardScaler mean/scale so the ESP32 normalizes sensor readings
# the exact same way the training data was normalized.
# ---------------------------------------------------------------------------
def c_array(name, arr):
    flat = np.asarray(arr).flatten()
    vals = ", ".join(f"{v:.8f}f" for v in flat)
    return f"const float {name}[{flat.size}] = {{{vals}}};"

lines = []
lines.append("// AUTO-GENERATED by train_export_transformer.py -- do not hand-edit.")
lines.append("// Tiny hand-built Transformer: 4 vitals -> 4-class partial-NEWS2 risk band.")
lines.append("// Trained on: CardiacPatientData_filtered.csv + NHAMCS_2021_CLEAN.csv + NHAMCS_2022_CLEAN.csv")
lines.append(f"// Test accuracy on held-out 20%% split: {acc*100:.2f}%%")
lines.append("#pragma once")
lines.append("")
lines.append(f"#define TFM_N_TOKENS {N_TOKENS}")
lines.append(f"#define TFM_D_MODEL {D_MODEL}")
lines.append(f"#define TFM_N_HEADS {N_HEADS}")
lines.append(f"#define TFM_HEAD_DIM {HEAD_DIM}")
lines.append(f"#define TFM_FFN_HIDDEN {FFN_HIDDEN}")
lines.append(f"#define TFM_N_CLASSES {N_CLASSES}")
lines.append("")
lines.append("// Feature order the model expects: HR, RR, SpO2, Temperature_C")
lines.append(c_array("TFM_SCALER_MEAN", scaler.mean_))
lines.append(c_array("TFM_SCALER_SCALE", scaler.scale_))
lines.append("")
for key in ["Wemb", "bemb", "Wq", "Wk", "Wv", "Wo", "gamma1", "beta1",
            "W1", "b1", "W2", "b2", "gamma2", "beta2", "Wc", "bc"]:
    lines.append(c_array(f"TFM_{key}", np.asarray(final_params[key])))
lines.append("")
lines.append('static const char* TFM_CLASS_NAMES[4] = {"Normal", "Warning", "High", "Critical"};')

header_path = os.path.join(OUT_DIR, "model_weights.h")
with open(header_path, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nWrote {header_path}")

report = {
    "test_accuracy_pct": round(acc * 100, 2),
    "confusion_matrix": cm,
    "classes": CLASSES,
    "param_count": param_count,
    "size_kb_float32": round(param_count * 4 / 1024, 2),
    "rows_combined": len(combined),
    "rows_by_source": {
        "CardiacPatientData_filtered.csv": len(cardiac),
        "NHAMCS_2021_CLEAN.csv": len(nhamcs2021),
        "NHAMCS_2022_CLEAN.csv": len(nhamcs2022),
    },
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
}
report_path = os.path.join(OUT_DIR, "training_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"Wrote {report_path}")
