

import json
import os
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score, matthews_corrcoef, make_scorer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import jax
import jax.numpy as jnp
import optax

# ---- must match generate_all.py exactly ----
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

OUT_CHARTS_DIR = f"{SCRATCH}/model-explainer-web/public/charts"
os.makedirs(OUT_CHARTS_DIR, exist_ok=True)

t_start = time.time()

df = pd.read_csv(CSV_PATH)
y = df["status"]
X_clean = df[RAW_FEATURES].copy()

rng = np.random.default_rng(RANDOM_STATE)
X_noisy = X_clean.copy()
X_noisy["heart_rate_bpm"] += rng.normal(0, HR_NOISE_STD, len(df))
X_noisy["temperature_c"] += rng.normal(0, TEMP_NOISE_STD, len(df))
X_noisy["spo2_percent"] += rng.normal(0, SPO2_NOISE_STD, len(df))
X_noisy["spo2_percent"] = X_noisy["spo2_percent"].clip(0, 100)

idx_train, idx_test = train_test_split(df.index, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_CHARTS_DIR, name), dpi=150)
    plt.close()

def make_pipeline(model, scale=False):
    steps = []
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", model))
    return Pipeline(steps=steps)

def make_mlp():
    return MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", alpha=1e-4,
                          max_iter=400, random_state=RANDOM_STATE)

def make_xgb():
    return XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.15,
                          objective="multi:softprob", num_class=N_CLASSES,
                          eval_metric="mlogloss", random_state=RANDOM_STATE,
                          n_jobs=-1, verbosity=0)

# ================= Transformer (hand-built in JAX) — identical architecture to the Comparison page's =================
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
    out = jnp.einsum("bhts,bhsd->bhtd", attn, Vh).transpose(0, 2, 1, 3).reshape(B, T, D)
    out = out @ params["Wo"]
    x1 = layernorm(tok + out, params["gamma1"], params["beta1"])
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

def _make_batches(X, y, batch_size, seed):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    Xs, ys = np.asarray(X)[perm], np.asarray(y)[perm]
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

print(f"[{time.time()-t_start:.0f}s] Category 1 data ready ({len(df)} rows), training extra models...")

# ================= 5-fold CV scorecards (clean + noisy) =================
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
scoring = {"accuracy": "accuracy", "precision": "precision_macro", "recall": "recall_macro", "f1": "f1_macro", "mcc": make_scorer(matthews_corrcoef)}
datasets = {"clean": X_clean, "noisy": X_noisy}

new_scorecards = []

for ds_name, X in datasets.items():
    for model_name, model, scale in [("Neural Network", make_mlp(), True), ("XGBoost", make_xgb(), False)]:
        pipe = make_pipeline(model, scale=scale)
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
        new_scorecards.append({
            "dataset": ds_name, "model": model_name,
            "accuracy": round(float(scores["test_accuracy"].mean()) * 100, 1),
            "precision": round(float(scores["test_precision"].mean()) * 100, 1),
            "recall": round(float(scores["test_recall"].mean()) * 100, 1),
            "f1": round(float(scores["test_f1"].mean()) * 100, 1),
            "mcc": round(float(scores["test_mcc"].mean()), 3),
        })
    print(f"[{time.time()-t_start:.0f}s] Neural Network + XGBoost done for {ds_name} data")

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
    new_scorecards.append({
        "dataset": ds_name, "model": "Transformer",
        "accuracy": round(float(arr[:, 0].mean()) * 100, 1),
        "precision": round(float(arr[:, 1].mean()) * 100, 1),
        "recall": round(float(arr[:, 2].mean()) * 100, 1),
        "f1": round(float(arr[:, 3].mean()) * 100, 1),
        "mcc": round(float(arr[:, 4].mean()), 3),
    })
    print(f"[{time.time()-t_start:.0f}s] Transformer done for {ds_name} data")

# ================= single 80/20 split: confusion matrix + per-class recall on noisy data =================
extra_models_detail = {}

mlp_pipe = make_pipeline(make_mlp(), scale=True)
mlp_pipe.fit(X_noisy.loc[idx_train], y.loc[idx_train])
mlp_preds = mlp_pipe.predict(X_noisy.loc[idx_test])

xgb_pipe = make_pipeline(make_xgb(), scale=False)
xgb_pipe.fit(X_noisy.loc[idx_train], y.loc[idx_train])
xgb_preds = xgb_pipe.predict(X_noisy.loc[idx_test])

tfm_params, tfm_scaler = transformer_scale_fit(X_noisy.loc[idx_train], y.loc[idx_train])
tfm_preds = predict_transformer(tfm_params, tfm_scaler.transform(X_noisy.loc[idx_test]))

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
print(f"[{time.time()-t_start:.0f}s] confusion matrices + per-class recall done")

# ================= model size (KB), fit once on the full noisy data =================
mlp_full = make_mlp()
mlp_full.fit(StandardScaler().fit_transform(X_noisy), y)
mlp_kb = round(len(pickle.dumps(mlp_full)) / 1024, 1)

xgb_full = make_xgb()
xgb_full.fit(X_noisy, y)
xgb_kb = round(len(pickle.dumps(xgb_full)) / 1024, 1)

tfm_full_params = train_transformer(StandardScaler().fit_transform(X_noisy), y.to_numpy())
tfm_param_count = count_params(tfm_full_params)
tfm_kb = round(tfm_param_count * 4 / 1024, 1)

new_model_sizes = [
    {"model": "Neural Network", "kb": mlp_kb},
    {"model": "XGBoost", "kb": xgb_kb},
    {"model": "Transformer", "kb": tfm_kb},
]
print(f"[{time.time()-t_start:.0f}s] model sizes: {new_model_sizes}  (transformer has {tfm_param_count} parameters)")

# ================= merge into chart-data.json =================
with open(JSON_PATH) as f:
    out = json.load(f)

existing_pairs = {(s["dataset"], s["model"]) for s in out["scorecards"]}
for s in new_scorecards:
    key = (s["dataset"], s["model"])
    out["scorecards"] = [x for x in out["scorecards"] if (x["dataset"], x["model"]) != key] + [s]

out["modelSize"] = [m for m in out["modelSize"] if m["model"] not in {s["model"] for s in new_model_sizes}] + new_model_sizes

out["extraModels"] = extra_models_detail
out["extraModelsNote"] = ("Neural Network, XGBoost and Transformer, trained on Category 1's own 5,909 readings "
                           "alone (no combination with Category 2's data). Evaluated the same way as the "
                           "Decision Tree and Random Forest: same data, same 5-fold cross-validation, same "
                           "realistic sensor noise. None of them is the model deployed to the ESP32 -- only the "
                           "Decision Tree is small and simple enough to run on the chip without a phone or cloud.")
out["transformerParamCount"] = tfm_param_count

with open(JSON_PATH, "w") as f:
    json.dump(out, f, indent=2)

# ================= charts: all 5 models side by side =================
ALL_MODELS = ["Decision Tree", "Random Forest", "Neural Network", "XGBoost", "Transformer"]
MODEL_COLORS = {"Decision Tree": "#b5652a", "Random Forest": "#2A6F9A", "Neural Network": "#7d3c98",
                "XGBoost": "#1e8449", "Transformer": "#c0392b"}

def get(ds, model):
    return next(s for s in out["scorecards"] if s["dataset"] == ds and s["model"] == model)

noisy_acc = [get("noisy", m)["accuracy"] for m in ALL_MODELS]
noisy_f1 = [get("noisy", m)["f1"] for m in ALL_MODELS]

x = np.arange(len(ALL_MODELS))
width = 0.35
plt.figure(figsize=(9, 5))
plt.bar(x - width / 2, noisy_acc, width, label="Accuracy", color="#2A6F9A")
plt.bar(x + width / 2, noisy_f1, width, label="Macro F1", color="#c0392b")
plt.xticks(x, ALL_MODELS, rotation=15)
plt.ylabel("Score on realistic noisy data (%)")
plt.title("All 5 models compared, on the same noisy data (Category 1)")
plt.ylim(0, 100)
plt.legend()
for i, (a, f1) in enumerate(zip(noisy_acc, noisy_f1)):
    plt.text(i - width / 2, a + 1.5, f"{a:.1f}", ha="center", fontsize=8)
    plt.text(i + width / 2, f1 + 1.5, f"{f1:.1f}", ha="center", fontsize=8)
savefig("all_models_accuracy.png")

sizes = [next(m["kb"] for m in out["modelSize"] if m["model"] == name) for name in ALL_MODELS]
plt.figure(figsize=(9, 5))
plt.bar(ALL_MODELS, sizes, color=[MODEL_COLORS[m] for m in ALL_MODELS])
plt.yscale("log")
plt.ylabel("Model file size in KB (log scale)")
plt.title("Model footprint: all 5 models (Category 1)")
plt.xticks(rotation=15)
for i, v in enumerate(sizes):
    plt.text(i, v * 1.15, f"{v:,.1f} KB", ha="center", fontsize=8)
savefig("all_models_size.png")

print(f"[{time.time()-t_start:.0f}s] DONE. Wrote {JSON_PATH} and 2 new charts to {OUT_CHARTS_DIR}")
print("Noisy accuracy:", dict(zip(ALL_MODELS, noisy_acc)))
print("Noisy macro F1:", dict(zip(ALL_MODELS, noisy_f1)))
print("Model sizes (KB):", dict(zip(ALL_MODELS, sizes)))
