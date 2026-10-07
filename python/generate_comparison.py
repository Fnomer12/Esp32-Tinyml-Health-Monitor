

import json
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

CSV_PATH = "health_data_4class_combined.csv"
CLASS_NAMES = ["Normal", "Warning", "High", "Critical"]
N_CLASSES = len(CLASS_NAMES)
CLASS_COLORS = {"Normal": "#2e8b3d", "Warning": "#c8781f", "High": "#b5652a", "Critical": "#c0392b"}
RANDOM_STATE = 42
N_FOLDS = 5
HR_NOISE_STD, TEMP_NOISE_STD, SPO2_NOISE_STD = 5.0, 0.3, 2.0

RAW_FEATURES = ["heart_rate_bpm", "temperature_c", "spo2_percent"]

SMOTE_STRATEGY = {1: 12000, 2: 5000, 3: 5000}
UNDERSAMPLE_STRATEGY = {0: 20000}

OUT_CHARTS_DIR = "../public/charts-comparison"
OUT_DATA_DIR = "../data"
os.makedirs(OUT_CHARTS_DIR, exist_ok=True)
os.makedirs(OUT_DATA_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
y = df["status"]

X_clean = df[RAW_FEATURES].copy()

rng = np.random.default_rng(RANDOM_STATE)
X_noisy = X_clean.copy()
X_noisy["heart_rate_bpm"] += rng.normal(0, HR_NOISE_STD, len(df))
X_noisy["temperature_c"] += rng.normal(0, TEMP_NOISE_STD, len(df))
X_noisy["spo2_percent"] += rng.normal(0, SPO2_NOISE_STD, len(df))
X_noisy["spo2_percent"] = X_noisy["spo2_percent"].clip(0, 100)

out = {}
out["classNames"] = CLASS_NAMES
out["totalRows"] = int(len(df))
out["smoteStrategy"] = {CLASS_NAMES[k]: v for k, v in SMOTE_STRATEGY.items()}
out["features"] = RAW_FEATURES

def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_CHARTS_DIR, name), dpi=150)
    plt.close()

def make_pipeline(model):
    return ImbPipeline(steps=[("under", RandomUnderSampler(sampling_strategy=UNDERSAMPLE_STRATEGY, random_state=RANDOM_STATE)),
                               ("smote", SMOTE(sampling_strategy=SMOTE_STRATEGY, random_state=RANDOM_STATE, k_neighbors=5)),
                               ("clf", model)])

def make_models():
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=60, random_state=RANDOM_STATE, n_jobs=-1),
    }

counts = df["status"].value_counts().sort_index()
out["classCounts"] = [{"name": CLASS_NAMES[i], "count": int(counts.get(i, 0))} for i in range(N_CLASSES)]

plt.figure(figsize=(6, 4))
plt.bar(CLASS_NAMES, [c["count"] for c in out["classCounts"]],
        color=[CLASS_COLORS[c["name"]] for c in out["classCounts"]])
plt.title("How many REAL examples of each health status (before any oversampling)")
plt.xlabel("Health status")
plt.ylabel("Number of readings")
for i, c in enumerate(out["classCounts"]):
    plt.text(i, c["count"] + max(counts) * 0.01, f"{c['count']:,}", ha="center")
savefig("cmp_class_distribution.png")

idx_train, idx_test = train_test_split(df.index, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

dt_clean = make_pipeline(make_models()["Decision Tree"])
dt_clean.fit(X_clean.loc[idx_train], y.loc[idx_train])
cm_clean = confusion_matrix(y.loc[idx_test], dt_clean.predict(X_clean.loc[idx_test]))
out["confusionMatrixClean"] = cm_clean.tolist()

dt_noisy = make_pipeline(make_models()["Decision Tree"])
dt_noisy.fit(X_noisy.loc[idx_train], y.loc[idx_train])
cm_noisy = confusion_matrix(y.loc[idx_test], dt_noisy.predict(X_noisy.loc[idx_test]))
out["confusionMatrixNoisy"] = cm_noisy.tolist()

def plot_confusion(cm, filename, subtitle):
    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion matrix — {subtitle}\n(rows = the real answer, columns = the model's guess)")
    plt.colorbar(label="Number of readings")
    tick_marks = np.arange(N_CLASSES)
    plt.xticks(tick_marks, CLASS_NAMES, rotation=20)
    plt.yticks(tick_marks, CLASS_NAMES)
    plt.xlabel("What the model guessed")
    plt.ylabel("What the answer actually was")
    thresh = cm.max() / 2
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                      color="white" if cm[i, j] > thresh else "black", fontsize=11)
    savefig(filename)

plot_confusion(cm_clean, "cmp_confusion_matrix.png", "clean data")
plot_confusion(cm_noisy, "cmp_confusion_matrix_noisy.png", "realistic noisy data")

report = classification_report(y.loc[idx_test], dt_noisy.predict(X_noisy.loc[idx_test]),
                                target_names=CLASS_NAMES, output_dict=True, zero_division=0)
out["perClassRecallNoisy"] = [
    {"name": name, "recall": round(report[name]["recall"] * 100, 1), "support": int(report[name]["support"])}
    for name in CLASS_NAMES
]

plt.figure(figsize=(6, 4))
recalls = [r["recall"] for r in out["perClassRecallNoisy"]]
plt.bar(CLASS_NAMES, recalls, color=[CLASS_COLORS[n] for n in CLASS_NAMES])
plt.ylabel("Recall (%) — how many of the real cases it actually caught")
plt.title("Per-class recall, Decision Tree on noisy data (Comparison)")
plt.ylim(0, 110)
for i, r in enumerate(recalls):
    plt.text(i, r + 2, f"{r:.1f}%", ha="center")
savefig("cmp_per_class_recall.png")

importances = dt_noisy.named_steps["clf"].feature_importances_
FEATURE_LABELS = {"heart_rate_bpm": "Heart rate", "temperature_c": "Temperature", "spo2_percent": "SpO2"}
out["featureImportance"] = [
    {"feature": FEATURE_LABELS[f], "value": round(float(v), 4)} for f, v in zip(RAW_FEATURES, importances)
]
sorted_fi = sorted(out["featureImportance"], key=lambda f: -f["value"])
plt.figure(figsize=(7, 4.5))
plt.barh([f["feature"] for f in sorted_fi][::-1], [f["value"] for f in sorted_fi][::-1], color="#b5652a")
plt.xlabel("How much the model relies on this input (0 to 1)")
plt.title("Feature importance (Comparison, noisy-trained)")
for i, f in enumerate(sorted_fi[::-1]):
    plt.text(f["value"] + 0.005, i, f"{f['value']*100:.0f}%", va="center")
savefig("cmp_feature_importance.png")

X_train_under, y_train_under = RandomUnderSampler(sampling_strategy=UNDERSAMPLE_STRATEGY, random_state=RANDOM_STATE).fit_resample(
    X_noisy.loc[idx_train], y.loc[idx_train]
)
X_train_sm, y_train_sm = SMOTE(sampling_strategy=SMOTE_STRATEGY, random_state=RANDOM_STATE, k_neighbors=5).fit_resample(
    X_train_under, y_train_under
)
small_tree = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
small_tree.fit(X_train_sm, y_train_sm)
plt.figure(figsize=(19, 9))
plot_tree(small_tree, feature_names=[FEATURE_LABELS[f] for f in RAW_FEATURES],
          class_names=CLASS_NAMES, filled=True, rounded=True, fontsize=8)
plt.title("Decision tree (simplified to 3 levels) — Comparison, trained on noisy + SMOTE-balanced data")
savefig("cmp_decision_tree.png")

# ---- also export this same tree as JSON, for the interactive diagram on the website ----
# (same tree, same fit -- the PNG and the JSON can never disagree)
def _tree_to_json(tree_, node_id, feature_names, class_names):
    left = tree_.children_left[node_id]
    right = tree_.children_right[node_id]
    if left == right:  # leaf
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

out["decisionTree"] = _tree_to_json(
    small_tree.tree_, 0, [FEATURE_LABELS[f] for f in RAW_FEATURES], CLASS_NAMES
)

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
scoring = {"accuracy": "accuracy", "precision": "precision_macro", "recall": "recall_macro", "f1": "f1_macro"}
datasets = {"clean": X_clean, "noisy": X_noisy}

scorecards = []
for ds_name, X in datasets.items():
    for model_name, model in make_models().items():
        pipe = make_pipeline(model)
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
        scorecards.append({
            "dataset": ds_name, "model": model_name,
            "accuracy": round(float(scores["test_accuracy"].mean()) * 100, 1),
            "precision": round(float(scores["test_precision"].mean()) * 100, 1),
            "recall": round(float(scores["test_recall"].mean()) * 100, 1),
            "f1": round(float(scores["test_f1"].mean()) * 100, 1),
        })
out["scorecards"] = scorecards

X_full_under, y_full_under = RandomUnderSampler(sampling_strategy=UNDERSAMPLE_STRATEGY, random_state=RANDOM_STATE).fit_resample(X_noisy, y)
X_full_sm, y_full_sm = SMOTE(sampling_strategy=SMOTE_STRATEGY, random_state=RANDOM_STATE, k_neighbors=5).fit_resample(X_full_under, y_full_under)
size_kb = {}
for model_name, model in make_models().items():
    model.fit(X_full_sm, y_full_sm)
    size_kb[model_name] = round(len(pickle.dumps(model)) / 1024, 1)
out["modelSize"] = [{"model": "Decision Tree", "kb": size_kb["Decision Tree"]},
                     {"model": "Random Forest", "kb": size_kb["Random Forest"]}]
out["modelSizeRatio"] = round(size_kb["Random Forest"] / size_kb["Decision Tree"], 1)

plt.figure(figsize=(5, 4))
plt.bar(size_kb.keys(), size_kb.values(), color=["#b5652a", "#2A6F9A"])
plt.ylabel("Model file size (KB)")
plt.title("Model footprint: Decision Tree vs Random Forest (Comparison)")
for i, (name, val) in enumerate(size_kb.items()):
    plt.text(i, val + max(size_kb.values()) * 0.02, f"{val:,.1f} KB", ha="center")
savefig("cmp_dt_vs_rf_size.png")

with open(os.path.join(OUT_DATA_DIR, "chart-data-comparison.json"), "w") as f:
    json.dump(out, f, indent=2)
with open("chart-data-comparison.json", "w") as f:
    json.dump(out, f, indent=2)

def get(ds, model):
    return next(s for s in scorecards if s["dataset"] == ds and s["model"] == model)

print(f"Total rows: {out['totalRows']}")
print("Features used:", RAW_FEATURES)
print("Class counts (real, before SMOTE):", {c['name']: c['count'] for c in out['classCounts']})
print(f"Decision Tree  clean={get('clean','Decision Tree')['accuracy']}%  noisy={get('noisy','Decision Tree')['accuracy']}%  (macro F1 noisy={get('noisy','Decision Tree')['f1']}%)")
print(f"Random Forest  clean={get('clean','Random Forest')['accuracy']}%  noisy={get('noisy','Random Forest')['accuracy']}%  (macro F1 noisy={get('noisy','Random Forest')['f1']}%)")
print(f"Model size ratio: {out['modelSizeRatio']}x")
print("Per-class recall (Decision Tree, noisy):", out["perClassRecallNoisy"])
print("Wrote PNGs to", OUT_CHARTS_DIR, "and chart-data-comparison.json to", OUT_DATA_DIR)
