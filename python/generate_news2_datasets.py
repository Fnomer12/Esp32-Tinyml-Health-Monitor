"""
generate_news2_datasets.py
============================
Trains Category 1, Category 2, and Category 3 INDIVIDUALLY -- one real
dataset each, not combined. This replaces the old Category 1 (a
synthetic 5,909-row Kaggle set) and Category 2 (a synthetic 200,020-row
Kaggle set) with the three real datasets used everywhere else on this
site:

  Category 1 -> Cardiac Triage   (data/chart-data.json)
  Category 2 -> NHAMCS 2021      (data/chart-data-category2.json)
  Category 3 -> NHAMCS 2022      (data/chart-data-category3.json)   [NEW]

Each dataset has a real, measured HR, RR, SpO2, and Temperature --
nothing synthetic, nothing guessed. All four vitals are banded against
the exact official NHS/Royal College of Physicians NEWS2 chart (Dec
2022 edition, SpO2 Scale 1), same as the combined Comparison page
(see generate_rr_comparison.py, which produces data/chart-data-rr.json
for the Comparison page).

Five models are trained and scored on each dataset: Decision Tree,
Random Forest, Neural Network, XGBoost, and a small hand-built
Transformer (adapted from the old generate_extra_models.py -- same
architecture, re-trained here on the real 4-vital NEWS2 data with no
resampling, since these real datasets don't need SMOTE/undersampling
the way the old synthetic ones did).

Each output JSON has the same shape chart-data-rr.json uses, so
Category1/2/3Explainer.tsx can share the same VizCharts components the
rest of the site already uses: BarChart, ConfusionMatrix, Histogram,
DecisionTreeDiagram, GroupedBarChart.

HOW TO RUN
----------
Run from the model-explainer-web/python/ directory (F5 in Spyder, or
`python3 generate_news2_datasets.py`). Needs jax and optax installed
(`pip install jax optax`) in addition to the usual sklearn/xgboost
stack. Trains 3 datasets x 5 models x 5 folds, so it takes a few
minutes. Restart `npm run dev` afterwards.
"""

import pandas as pd
import numpy as np
import json
import os
import pickle

from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix
import xgboost as xgb

import jax
import jax.numpy as jnp
import optax

np.random.seed(42)

DATASET_DIR = "/Users/macbookpro/Desktop/Valley View University/Final Year Project/Dataset/Filtered_dataset"
DATA_OUT = "../data"
os.makedirs(DATA_OUT, exist_ok=True)

FEATURES = ["HR", "RR", "SpO2", "Temperature_C"]
DISPLAY_NAMES = {"HR": "Heart rate", "RR": "Respiratory rate", "SpO2": "SpO2", "Temperature_C": "Temperature"}
CLASSES = ["Normal", "Warning", "High", "Critical"]
RANDOM_STATE = 42
N_FOLDS = 5

# ---- EXACT official NEWS2 bands (Dec 2022 chart, SpO2 Scale 1) -- same as generate_rr_comparison.py ----
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

def add_noise(X):
    noisy = X.copy().astype(float)
    noisy[:, 0] += np.random.normal(0, 5, size=len(X))
    noisy[:, 1] += np.random.normal(0, 2, size=len(X))
    noisy[:, 2] += np.random.normal(0, 2, size=len(X))
    noisy[:, 3] += np.random.normal(0, 0.3, size=len(X))
    return noisy

def build_models():
    return {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=400, random_state=RANDOM_STATE),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE, eval_metric="mlogloss", n_jobs=-1),
    }

# ---------------------------------------------------------------------------
# Transformer -- a small hand-built attention model (adapted from the old
# generate_extra_models.py). Each of the 4 vitals is treated as one "token".
# No resampling here: unlike the old synthetic data, these real NEWS2
# datasets have all four classes represented well enough for plain
# StandardScaler + stratified CV, same as the other four models.
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


def run_dataset(label, df, out_filename, source_note):
    df = label_df(df)
    X_all = df[FEATURES].values
    y_all = np.array([CLASSES.index(v) for v in df["label"].values])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scorecards = []
    dt_cm_clean = np.zeros((4, 4), dtype=int)
    dt_cm_noisy = np.zeros((4, 4), dtype=int)

    for model_name in build_models():
        acc_c, prec_c, rec_c, f1_c, mcc_c = [], [], [], [], []
        acc_n, prec_n, rec_n, f1_n, mcc_n = [], [], [], [], []
        for train_idx, test_idx in skf.split(X_all, y_all):
            X_train, X_test = X_all[train_idx], X_all[test_idx]
            y_train, y_test = y_all[train_idx], y_all[test_idx]
            scaler = StandardScaler().fit(X_train)
            Xtr_s, Xte_s = scaler.transform(X_train), scaler.transform(X_test)
            model = build_models()[model_name]
            model.fit(Xtr_s, y_train)

            pred = model.predict(Xte_s)
            acc_c.append(accuracy_score(y_test, pred))
            prec_c.append(precision_score(y_test, pred, average="macro", zero_division=0))
            rec_c.append(recall_score(y_test, pred, average="macro", zero_division=0))
            f1_c.append(f1_score(y_test, pred, average="macro"))
            mcc_c.append(matthews_corrcoef(y_test, pred))

            Xte_noisy_s = scaler.transform(add_noise(X_test))
            pred_n = model.predict(Xte_noisy_s)
            acc_n.append(accuracy_score(y_test, pred_n))
            prec_n.append(precision_score(y_test, pred_n, average="macro", zero_division=0))
            rec_n.append(recall_score(y_test, pred_n, average="macro", zero_division=0))
            f1_n.append(f1_score(y_test, pred_n, average="macro"))
            mcc_n.append(matthews_corrcoef(y_test, pred_n))

            if model_name == "Decision Tree":
                dt_cm_clean += confusion_matrix(y_test, pred, labels=[0, 1, 2, 3])
                dt_cm_noisy += confusion_matrix(y_test, pred_n, labels=[0, 1, 2, 3])

        scorecards.append({"dataset": "clean", "model": model_name,
            "accuracy": round(np.mean(acc_c) * 100, 1), "precision": round(np.mean(prec_c) * 100, 1),
            "recall": round(np.mean(rec_c) * 100, 1), "f1": round(np.mean(f1_c) * 100, 1), "mcc": round(np.mean(mcc_c), 3)})
        scorecards.append({"dataset": "noisy", "model": model_name,
            "accuracy": round(np.mean(acc_n) * 100, 1), "precision": round(np.mean(prec_n) * 100, 1),
            "recall": round(np.mean(rec_n) * 100, 1), "f1": round(np.mean(f1_n) * 100, 1), "mcc": round(np.mean(mcc_n), 3)})
        print(f"  {model_name:15s} clean {scorecards[-2]['accuracy']:5.1f}%  noisy {scorecards[-1]['accuracy']:5.1f}%")

    # ---- Transformer: same folds (same skf + random_state => identical splits), manual loop ----
    acc_c, prec_c, rec_c, f1_c, mcc_c = [], [], [], [], []
    acc_n, prec_n, rec_n, f1_n, mcc_n = [], [], [], [], []
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all)):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        scaler = StandardScaler().fit(X_train)
        Xtr_s, Xte_s = scaler.transform(X_train), scaler.transform(X_test)
        params = train_transformer(Xtr_s, y_train, seed=RANDOM_STATE + fold_i)

        pred = predict_transformer(params, Xte_s)
        acc_c.append(accuracy_score(y_test, pred))
        prec_c.append(precision_score(y_test, pred, average="macro", zero_division=0))
        rec_c.append(recall_score(y_test, pred, average="macro", zero_division=0))
        f1_c.append(f1_score(y_test, pred, average="macro"))
        mcc_c.append(matthews_corrcoef(y_test, pred))

        Xte_noisy_s = scaler.transform(add_noise(X_test))
        pred_n = predict_transformer(params, Xte_noisy_s)
        acc_n.append(accuracy_score(y_test, pred_n))
        prec_n.append(precision_score(y_test, pred_n, average="macro", zero_division=0))
        rec_n.append(recall_score(y_test, pred_n, average="macro", zero_division=0))
        f1_n.append(f1_score(y_test, pred_n, average="macro"))
        mcc_n.append(matthews_corrcoef(y_test, pred_n))

    scorecards.append({"dataset": "clean", "model": "Transformer",
        "accuracy": round(np.mean(acc_c) * 100, 1), "precision": round(np.mean(prec_c) * 100, 1),
        "recall": round(np.mean(rec_c) * 100, 1), "f1": round(np.mean(f1_c) * 100, 1), "mcc": round(np.mean(mcc_c), 3)})
    scorecards.append({"dataset": "noisy", "model": "Transformer",
        "accuracy": round(np.mean(acc_n) * 100, 1), "precision": round(np.mean(prec_n) * 100, 1),
        "recall": round(np.mean(rec_n) * 100, 1), "f1": round(np.mean(f1_n) * 100, 1), "mcc": round(np.mean(mcc_n), 3)})
    print(f"  {'Transformer':15s} clean {scorecards[-2]['accuracy']:5.1f}%  noisy {scorecards[-1]['accuracy']:5.1f}%")

    per_class_recall = []
    for i, c in enumerate(CLASSES):
        support = dt_cm_noisy[i].sum()
        recall = dt_cm_noisy[i, i] / support * 100 if support else 0
        per_class_recall.append({"name": c, "recall": round(float(recall), 1), "support": int(support)})

    scaler = StandardScaler().fit(X_all)
    Xs = scaler.transform(X_all)
    dt_full = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE).fit(Xs, y_all)
    rf_full = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1).fit(Xs, y_all)
    nn_full = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=400, random_state=RANDOM_STATE).fit(Xs, y_all)
    xgb_full = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE, eval_metric="mlogloss", n_jobs=-1).fit(Xs, y_all)
    tfm_full_params = train_transformer(Xs, y_all, seed=RANDOM_STATE)
    tfm_param_count = _tfm_count_params(tfm_full_params)

    feature_importance = [{"feature": DISPLAY_NAMES[f], "value": round(float(v), 4)} for f, v in zip(FEATURES, dt_full.feature_importances_)]

    sizes_kb = {}
    for name, model in [("Decision Tree", dt_full), ("Random Forest", rf_full), ("Neural Network", nn_full), ("XGBoost", xgb_full)]:
        sizes_kb[name] = round(len(pickle.dumps(model)) / 1024, 1)
    sizes_kb["Transformer"] = round(tfm_param_count * 4 / 1024, 1)  # float32 weights
    model_size_ratio = round(sizes_kb["Random Forest"] / sizes_kb["Decision Tree"], 1)

    tree_ = dt_full.tree_
    feat_names_for_tree = [DISPLAY_NAMES[f] for f in FEATURES]
    means, scales = scaler.mean_, scaler.scale_

    def export_node(node_id):
        if tree_.children_left[node_id] == tree_.children_right[node_id]:
            counts = tree_.value[node_id][0]
            pred_idx = int(np.argmax(counts))
            return {"leaf": True, "prediction": CLASSES[pred_idx], "counts": [int(c) for c in counts], "totalSamples": int(counts.sum())}
        feat_idx = tree_.feature[node_id]
        raw_threshold = tree_.threshold[node_id] * scales[feat_idx] + means[feat_idx]
        return {"leaf": False, "feature": feat_names_for_tree[feat_idx], "threshold": round(float(raw_threshold), 1),
                "left": export_node(tree_.children_left[node_id]), "right": export_node(tree_.children_right[node_id])}

    tree_json = export_node(0)

    histograms = {}
    bin_counts = {"HR": 15, "RR": 15, "SpO2": 12, "Temperature_C": 12}
    for feat in FEATURES:
        vals = df[feat].values
        lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
        edges = np.linspace(lo, hi, bin_counts[feat] + 1)
        counts_by_class = {}
        for c in CLASSES:
            sub = df[df["label"] == c][feat].values
            counts, _ = np.histogram(sub, bins=edges)
            counts_by_class[c] = counts.tolist()
        histograms[feat] = {"binEdges": edges.round(2).tolist(), "counts": counts_by_class}

    final = {
        "classNames": CLASSES,
        "totalRows": int(len(df)),
        "sourceNote": source_note,
        "features": FEATURES,
        "featureDisplayNames": DISPLAY_NAMES,
        "classCounts": [{"name": c, "count": int((df["label"] == c).sum())} for c in CLASSES],
        "confusionMatrixClean": dt_cm_clean.tolist(),
        "confusionMatrixNoisy": dt_cm_noisy.tolist(),
        "perClassRecallNoisy": per_class_recall,
        "featureImportance": feature_importance,
        "modelSizeRatio": model_size_ratio,
        "scorecards": scorecards,
        "modelSize": [{"model": k, "kb": v} for k, v in sizes_kb.items()],
        "decisionTree": tree_json,
        "histograms": histograms,
        "rrBandingNote": "All four vitals (heart rate, respiratory rate, SpO2, temperature) are banded using the exact official NHS/Royal College of Physicians NEWS2 chart (Dec 2022 edition, SpO2 Scale 1) -- not an ad-hoc rule.",
        "noiseModel": {"HR": "+-5 bpm", "RR": "+-2 breaths/min", "SpO2": "+-2%", "Temperature_C": "+-0.3C"},
        "transformerParamCount": tfm_param_count,
    }

    out_path = os.path.join(DATA_OUT, out_filename)
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"  -> wrote {out_path} ({final['totalRows']} rows)")
    print("  class counts:", {c['name']: c['count'] for c in final['classCounts']})
    return final


if __name__ == "__main__":
    cardiac = pd.read_csv(os.path.join(DATASET_DIR, "CardiacPatientData_filtered.csv"))[["HR", "RR", "SpO2", "BT_C"]].rename(columns={"BT_C": "Temperature_C"})
    nhamcs2021 = pd.read_csv(os.path.join(DATASET_DIR, "NHAMCS_2021_CLEAN.csv"))[["HR", "RR", "SpO2", "Temperature_C"]]
    nhamcs2022 = pd.read_csv(os.path.join(DATASET_DIR, "NHAMCS_2022_CLEAN.csv"))[["HR", "RR", "SpO2", "Temperature_C"]]

    print("Category 1 -- Cardiac Triage")
    run_dataset("Category 1", cardiac, "chart-data.json", "Real cardiac/ED triage dataset -- clinician-recorded vitals, including respiratory rate.")

    print("\nCategory 2 -- NHAMCS 2021")
    run_dataset("Category 2", nhamcs2021, "chart-data-category2.json", "NHAMCS 2021 -- a real year of U.S. emergency department visits (National Hospital Ambulatory Medical Care Survey).")

    print("\nCategory 3 -- NHAMCS 2022")
    run_dataset("Category 3", nhamcs2022, "chart-data-category3.json", "NHAMCS 2022 -- a second, real year of U.S. emergency department visits.")

    print("\nDone. Comparison data (all three combined) is generated separately by generate_rr_comparison.py.")
