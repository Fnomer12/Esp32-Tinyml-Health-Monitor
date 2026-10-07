"""
generate_category1_full.py
---------------------------
Rebuilds Category 1 from scratch using the REAL NEWS2 (National Early
Warning Score 2, Royal College of Physicians) clinical bands for heart
rate, temperature, and SpO2 -- the same chart Sampson supplied. Each
reading gets whichever of the three vitals scores worst (highest NEWS2
point value), mapped onto this site's 4-level scale:
    NEWS2 points 0 -> Normal, 1 -> Warning, 2 -> High, 3 -> Critical

This REPLACES generate_extra_models_cat1.py for Category 1: it trains
Decision Tree, Random Forest, Neural Network, XGBoost, and the hand-built
Transformer, all on the correct 4-class target, and writes a complete,
internally-consistent chart-data.json (not a merge -- a full rebuild).
It also adds "featureHistograms" (binned per-class distributions) so the
feature-distribution chart can be built from real data instead of a
static image.

Run this once in Spyder; it replaces chart-data.json in full.
"""

import json
import os
import pickle
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report, precision_recall_fscore_support,
                              accuracy_score, matthews_corrcoef, make_scorer)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import jax
import jax.numpy as jnp
import optax

# ---- paths (same resolution logic as generate_extra_models_cat1.py) ----
SCRATCH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH = os.path.join(os.path.dirname(SCRATCH), "Datasets", "health_data_clean_1.csv")
JSON_PATH = f"{SCRATCH}/model-explainer-web/data/chart-data.json"

CLASS_NAMES = ["Normal", "Warning", "High", "Critical"]
N_CLASSES = len(CLASS_NAMES)
RANDOM_STATE = 42
N_FOLDS = 5
HR_NOISE_STD, TEMP_NOISE_STD, SPO2_NOISE_STD = 5.0, 0.3, 2.0
RAW_FEATURES = ["heart_rate_bpm", "temperature_c", "spo2_percent"]
FEATURE_LABELS = {"heart_rate_bpm": "Heart rate", "temperature_c": "Temperature", "spo2_percent": "SpO2"}
MODEL_COLORS = {"Decision Tree": "#eb6834", "Random Forest": "#2a78d6", "Neural Network": "#1baf7a",
                "XGBoost": "#eda100", "Transformer": "#e87ba4"}

t_start = time.time()

# ================= NEWS2-derived 4-class relabeling =================
# Bands taken directly from the Royal College of Physicians NEWS2 chart
# (Pulse, SpO2 Scale 1, Temperature rows). "Worst of three" = highest
# NEWS2 point score across the 3 vitals this project measures.

def hr_score(hr):
    if hr <= 30 or hr >= 131:
        return 3
    if 111 <= hr <= 130:
        return 2
    if (91 <= hr <= 110) or (41 <= hr <= 50):
        return 1
    return 0

def spo2_score(spo2):
    if spo2 <= 91:
        return 3
    if 92 <= spo2 <= 93:
        return 2
    if 94 <= spo2 <= 95:
        return 1
    return 0

def temp_score(t):
    if t <= 35.0:
        return 3
    if t >= 39.1:
        return 2
    if (38.1 <= t <= 39.0) or (35.1 <= t <= 36.0):
        return 1
    return 0

def news2_label(row):
    s = max(hr_score(row["heart_rate_bpm"]), spo2_score(row["spo2_percent"]), temp_score(row["temperature_c"]))
    return s

df = pd.read_csv(CSV_PATH)
df["severity"] = df.apply(news2_label, axis=1)
y = df["severity"]
X_clean = df[RAW_FEATURES].copy()

print(f"[{time.time()-t_start:.0f}s] NEWS2 relabel done. Class counts:")
print(y.value_counts().sort_index())

rng = np.random.default_rng(RANDOM_STATE)
X_noisy = X_clean.copy()
X_noisy["heart_rate_bpm"] += rng.normal(0, HR_NOISE_STD, len(df))
X_noisy["temperature_c"] += rng.normal(0, TEMP_NOISE_STD, len(df))
X_noisy["spo2_percent"] += rng.normal(0, SPO2_NOISE_STD, len(df))
X_noisy["spo2_percent"] = X_noisy["spo2_percent"].clip(0, 100)

idx_train, idx_test = train_test_split(df.index, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

out = {}
out["classNames"] = CLASS_NAMES
out["features"] = RAW_FEATURES

counts = y.value_counts().sort_index()
out["classCounts"] = [{"name": CLASS_NAMES[i], "count": int(counts.get(i, 0))} for i in range(N_CLASSES)]

# ================= feature histograms (real binned data, for the g2 chart) =================
N_BINS = 22
feature_histograms = {}
for raw_f in RAW_FEATURES:
    label = FEATURE_LABELS[raw_f]
    vals = X_clean[raw_f].to_numpy()
    lo, hi = float(vals.min()), float(vals.max())
    edges = np.linspace(lo, hi, N_BINS + 1)
    counts_by_class = {}
    for ci, cname in enumerate(CLASS_NAMES):
        mask = (y == ci).to_numpy()
        hist, _ = np.histogram(vals[mask], bins=edges)
        counts_by_class[cname] = [int(v) for v in hist]
    feature_histograms[label] = {
        "binEdges": [round(float(e), 2) for e in edges],
        "counts": counts_by_class,
    }
out["featureHistograms"] = feature_histograms
print(f"[{time.time()-t_start:.0f}s] feature histograms done")

# ================= models =================

def make_pipeline(model, scale=False):
    steps = []
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", model))
    return Pipeline(steps=steps)

def make_dt():
    return DecisionTreeClassifier(random_state=RANDOM_STATE)

def make_rf():
    return RandomForestClassifier(n_estimators=60, random_state=RANDOM_STATE, n_jobs=-1)

def make_mlp():
    return MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", alpha=1e-4,
                          max_iter=400, random_state=RANDOM_STATE)

def make_xgb():
    return XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.15,
                          objective="multi:softprob", num_class=N_CLASSES,
                          eval_metric="mlogloss", random_state=RANDOM_STATE,
                          n_jobs=-1, verbosity=0)

# ---- hand-built Transformer (JAX), identical architecture to the other pages' ----
D_MODEL, FFN_HIDDEN, N_HEADS = 16, 64, 2
HEAD_DIM = D_MODEL // N_HEADS
N_TOKENS = len(RAW_FEATURES)
EPOCHS, BATCH_SIZE, LEARNING_RATE = 90, 256, 3e-3
optimizer = optax.adam(LEARNING_RATE)

def init_transformer_params(key):
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

def count_params(params):
    return int(sum(np.prod(v.shape) for v in jax.tree_util.tree_leaves(params)))

def layernorm(x, gamma, beta, eps=1e-5):
    mu = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + eps) * gamma + beta

def transformer_forward(params, x):
    tok = x[:, :, None] * params["Wemb"][None, :, :] + params["bemb"][None, :, :]
    Q, K, V = tok @ params["Wq"], tok @ params["Wk"], tok @ params["Wv"]
    B, T, D = Q.shape
    split = lambda t: t.reshape(B, T, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    Qh, Kh, Vh = split(Q), split(K), split(V)
    scores = jnp.einsum("bhtd,bhsd->bhts", Qh, Kh) / jnp.sqrt(HEAD_DIM)
    attn = jax.nn.softmax(scores, axis=-1)
    out_ = jnp.einsum("bhts,bhsd->bhtd", attn, Vh).transpose(0, 2, 1, 3).reshape(B, T, D)
    out_ = out_ @ params["Wo"]
    x1 = layernorm(tok + out_, params["gamma1"], params["beta1"])
    ffn = jax.nn.relu(x1 @ params["W1"] + params["b1"]) @ params["W2"] + params["b2"]
    x2 = layernorm(x1 + ffn, params["gamma2"], params["beta2"])
    pooled = jnp.mean(x2, axis=1)
    return pooled @ params["Wc"] + params["bc"]

def loss_fn(params, xb, yb):
    logits = transformer_forward(params, xb)
    one_hot = jax.nn.one_hot(yb, N_CLASSES)
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(one_hot * logp, axis=-1))

def train_step(params, opt_state, xb, yb):
    loss, grads = jax.value_and_grad(loss_fn)(params, xb, yb)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss

def _make_batches(X, y_, batch_size, seed):
    rng_ = np.random.default_rng(seed)
    n = X.shape[0]
    perm = rng_.permutation(n)
    Xs, ys = np.asarray(X)[perm], np.asarray(y_)[perm]
    n_batches = max(1, n // batch_size)
    Xb = Xs[:n_batches * batch_size].reshape(n_batches, batch_size, -1)
    yb = ys[:n_batches * batch_size].reshape(n_batches, batch_size)
    return Xb, yb, n_batches

@jax.jit
def _run_scan(params, opt_state, Xb_all, yb_all):
    def step(carry, batch):
        params, opt_state = carry
        xb, yb = batch
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        return (params, opt_state), loss
    (params, opt_state), losses = jax.lax.scan(step, (params, opt_state), (Xb_all, yb_all))
    return params, losses

def train_transformer(Xtr, ytr, seed=RANDOM_STATE):
    Xb, yb, n_batches = _make_batches(Xtr, ytr, min(BATCH_SIZE, len(Xtr)), seed)
    Xb_all = jnp.asarray(np.tile(Xb, (EPOCHS, 1, 1)), dtype=jnp.float32)
    yb_all = jnp.asarray(np.tile(yb, (EPOCHS, 1)), dtype=jnp.int32)
    key = jax.random.PRNGKey(seed)
    params = init_transformer_params(key)
    opt_state = optimizer.init(params)
    params, losses = _run_scan(params, opt_state, Xb_all, yb_all)
    return params

def predict_transformer(params, X):
    logits = transformer_forward(params, jnp.asarray(X, dtype=jnp.float32))
    return np.array(jnp.argmax(logits, axis=-1))

def transformer_scale_fit(X_train_raw, y_train_raw):
    scaler = StandardScaler().fit(X_train_raw)
    y_arr = y_train_raw.to_numpy() if hasattr(y_train_raw, "to_numpy") else y_train_raw
    params = train_transformer(scaler.transform(X_train_raw), y_arr)
    return params, scaler

# ================= 5-fold CV scorecards (clean + noisy), all 5 models =================
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
scoring = {"accuracy": "accuracy", "precision": "precision_macro", "recall": "recall_macro",
           "f1": "f1_macro", "mcc": make_scorer(matthews_corrcoef)}
datasets = {"clean": X_clean, "noisy": X_noisy}

scorecards = []
for ds_name, X in datasets.items():
    for model_name, model, scale in [("Decision Tree", make_dt(), False), ("Random Forest", make_rf(), False),
                                      ("Neural Network", make_mlp(), True), ("XGBoost", make_xgb(), False)]:
        pipe = make_pipeline(model, scale=scale)
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
        scorecards.append({
            "dataset": ds_name, "model": model_name,
            "accuracy": round(float(scores["test_accuracy"].mean()) * 100, 1),
            "precision": round(float(scores["test_precision"].mean()) * 100, 1),
            "recall": round(float(scores["test_recall"].mean()) * 100, 1),
            "f1": round(float(scores["test_f1"].mean()) * 100, 1),
            "mcc": round(float(scores["test_mcc"].mean()), 3),
        })
    print(f"[{time.time()-t_start:.0f}s] DT + RF + NN + XGBoost done for {ds_name} data")

for ds_name, X in datasets.items():
    fold_metrics = []
    for fold_i, (tr_idx, te_idx) in enumerate(cv.split(X, y)):
        Xtr_raw, ytr_raw = X.iloc[tr_idx], y.iloc[tr_idx]
        Xte_raw, yte_raw = X.iloc[te_idx], y.iloc[te_idx]
        params, scaler = transformer_scale_fit(Xtr_raw, ytr_raw)
        preds = predict_transformer(params, scaler.transform(Xte_raw))
        acc = accuracy_score(yte_raw, preds)
        p, r, f1, _ = precision_recall_fscore_support(yte_raw, preds, average="macro", zero_division=0)
        mcc = matthews_corrcoef(yte_raw, preds)
        fold_metrics.append((acc, p, r, f1, mcc))
    arr = np.array(fold_metrics)
    scorecards.append({
        "dataset": ds_name, "model": "Transformer",
        "accuracy": round(float(arr[:, 0].mean()) * 100, 1),
        "precision": round(float(arr[:, 1].mean()) * 100, 1),
        "recall": round(float(arr[:, 2].mean()) * 100, 1),
        "f1": round(float(arr[:, 3].mean()) * 100, 1),
        "mcc": round(float(arr[:, 4].mean()), 3),
    })
    print(f"[{time.time()-t_start:.0f}s] Transformer done for {ds_name} data")

out["scorecards"] = scorecards

# ================= single 80/20 split: confusion matrices, feature importance, tree =================
dt_clean = make_pipeline(make_dt())
dt_clean.fit(X_clean.loc[idx_train], y.loc[idx_train])
cm_clean = confusion_matrix(y.loc[idx_test], dt_clean.predict(X_clean.loc[idx_test]), labels=list(range(N_CLASSES)))
out["confusionMatrixClean"] = cm_clean.tolist()

dt_noisy = make_pipeline(make_dt())
dt_noisy.fit(X_noisy.loc[idx_train], y.loc[idx_train])
cm_noisy = confusion_matrix(y.loc[idx_test], dt_noisy.predict(X_noisy.loc[idx_test]), labels=list(range(N_CLASSES)))
out["confusionMatrixNoisy"] = cm_noisy.tolist()

importances = dt_noisy.named_steps["clf"].feature_importances_
out["featureImportance"] = [
    {"feature": FEATURE_LABELS[f], "value": round(float(v), 4)} for f, v in zip(RAW_FEATURES, importances)
]

# ---- full decision tree, serialized to JSON (frontend truncates for display) ----
def _tree_to_json(tree_, node_id, feature_names, class_names):
    left = tree_.children_left[node_id]
    right = tree_.children_right[node_id]
    if left == right:
        counts = tree_.value[node_id][0].tolist()
        pred_idx = int(max(range(len(counts)), key=lambda i: counts[i]))
        return {
            "leaf": True,
            "prediction": class_names[pred_idx],
            "counts": [int(round(c)) for c in counts],
            "totalSamples": int(sum(round(c) for c in counts)),
        }
    return {
        "leaf": False,
        "feature": feature_names[tree_.feature[node_id]],
        "threshold": round(float(tree_.threshold[node_id]), 2),
        "left": _tree_to_json(tree_, left, feature_names, class_names),
        "right": _tree_to_json(tree_, right, feature_names, class_names),
    }

out["decisionTree"] = _tree_to_json(dt_noisy.named_steps["clf"].tree_, 0, [FEATURE_LABELS[f] for f in RAW_FEATURES], CLASS_NAMES)
print(f"[{time.time()-t_start:.0f}s] confusion matrices, feature importance, decision tree done")

# ================= extra models detail: confusion + per-class recall, noisy, 80/20 split =================
mlp_pipe = make_pipeline(make_mlp(), scale=True)
mlp_pipe.fit(X_noisy.loc[idx_train], y.loc[idx_train])
mlp_preds = mlp_pipe.predict(X_noisy.loc[idx_test])

xgb_pipe = make_pipeline(make_xgb(), scale=False)
xgb_pipe.fit(X_noisy.loc[idx_train], y.loc[idx_train])
xgb_preds = xgb_pipe.predict(X_noisy.loc[idx_test])

tfm_params, tfm_scaler = transformer_scale_fit(X_noisy.loc[idx_train], y.loc[idx_train])
tfm_preds = predict_transformer(tfm_params, tfm_scaler.transform(X_noisy.loc[idx_test]))

extra_models_detail = {}
for name, preds in [("Neural Network", mlp_preds), ("XGBoost", xgb_preds), ("Transformer", tfm_preds)]:
    cm = confusion_matrix(y.loc[idx_test], preds, labels=list(range(N_CLASSES)))
    report = classification_report(y.loc[idx_test], preds, labels=list(range(N_CLASSES)),
                                    target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    extra_models_detail[name] = {
        "confusionMatrixNoisy": cm.tolist(),
        "perClassRecallNoisy": [
            {"name": n, "recall": round(report[n]["recall"] * 100, 1), "support": int(report[n]["support"])}
            for n in CLASS_NAMES
        ],
    }
out["extraModels"] = extra_models_detail
out["extraModelsNote"] = ("Neural Network, XGBoost and Transformer, trained on Category 1's own readings alone "
                           "(no combination with Category 2's data), using the official NEWS2 clinical bands for "
                           "heart rate, temperature, and SpO2 (worst of the three). Evaluated the same way as the "
                           "Decision Tree and Random Forest: same data, same 5-fold cross-validation, same "
                           "realistic sensor noise. None of them is the model deployed to the ESP32 -- only the "
                           "Decision Tree is small and simple enough to run on the chip without a phone or cloud.")

# ================= model sizes (KB), fit once on full noisy data =================
dt_full = make_dt()
dt_full.fit(X_noisy, y)
dt_kb = round(len(pickle.dumps(dt_full)) / 1024, 1)

rf_full = make_rf()
rf_full.fit(X_noisy, y)
rf_kb = round(len(pickle.dumps(rf_full)) / 1024, 1)

mlp_full = make_mlp()
mlp_full.fit(StandardScaler().fit_transform(X_noisy), y)
mlp_kb = round(len(pickle.dumps(mlp_full)) / 1024, 1)

xgb_full = make_xgb()
xgb_full.fit(X_noisy, y)
xgb_kb = round(len(pickle.dumps(xgb_full)) / 1024, 1)

tfm_full_params = train_transformer(StandardScaler().fit_transform(X_noisy), y.to_numpy())
tfm_param_count = count_params(tfm_full_params)
tfm_kb = round(tfm_param_count * 4 / 1024, 1)

out["modelSize"] = [
    {"model": "Decision Tree", "kb": dt_kb},
    {"model": "Random Forest", "kb": rf_kb},
    {"model": "Neural Network", "kb": mlp_kb},
    {"model": "XGBoost", "kb": xgb_kb},
    {"model": "Transformer", "kb": tfm_kb},
]
out["modelSizeRatio"] = round(rf_kb / dt_kb, 1)
out["transformerParamCount"] = tfm_param_count
print(f"[{time.time()-t_start:.0f}s] model sizes: DT={dt_kb}KB RF={rf_kb}KB NN={mlp_kb}KB XGB={xgb_kb}KB TFM={tfm_kb}KB ({tfm_param_count} params)")

with open(JSON_PATH, "w") as f:
    json.dump(out, f, indent=2)

print(f"[{time.time()-t_start:.0f}s] DONE. Wrote {JSON_PATH} (full rebuild, NEWS2-based labels)")

def get(ds, model):
    return next(s for s in scorecards if s["dataset"] == ds and s["model"] == model)

for m in ["Decision Tree", "Random Forest", "Neural Network", "XGBoost", "Transformer"]:
    s = get("noisy", m)
    print(f"{m:16s} noisy acc={s['accuracy']}%  f1={s['f1']}%  mcc={s['mcc']}")
