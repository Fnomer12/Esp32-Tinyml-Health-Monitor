# ESP32 Health Monitor — Model Explainer

**Final Year Project — Design and Implementation of an ESP32-Based TinyML Health Monitoring and Early Warning System**

A data and model walkthrough for a wearable-style health monitor that reads heart rate, SpO2, body temperature, and estimated respiratory rate, scores them against the official **NEWS2** early-warning protocol, and classifies patient status on-device using a TinyML **Transformer**. This app is the defense-prep companion site: it shows the real dataset, the full five-model comparison, and the reasoning behind every design decision — no slides, no hand-waving, just the numbers.

![Next.js](https://img.shields.io/badge/Next.js-14.2.35-000000?logo=next.js&logoColor=white)
![React](https://img.shields.io/badge/React-18.3.1-61DAFB?logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Overview

The core system is an ESP32-WROOM-32 wired to a MAX30102 (heart rate, SpO2, raw PPG) and a DS18B20 (body temperature). Respiratory rate is not measured directly — it's estimated from the MAX30102's raw PPG waveform. All four vitals are banded using the official NHS/Royal College of Physicians **NEWS2** chart, then classified on-device by a tiny Transformer model. Results are shown on an OLED display, raised as a buzzer/LED alert, and streamed over BLE to the companion **Skairo Health** mobile app.

This repository is the web app used to present and interrogate the machine-learning side of that system: which models were tried, how they were scored, and why the Transformer is the one that ships on the chip.

## What's inside

| Page | Route | What it shows |
|---|---|---|
| Category 1 — Cardiac Triage | `/` | Real cardiac/ED triage data, four vitals, official NEWS2 bands — Category 1 trained alone |
| Category 2 — NHAMCS 2021 | `/category2` | A real year of U.S. ED visits, four vitals, official NEWS2 bands — Category 2 trained alone |
| Category 3 — NHAMCS 2022 | `/category3` | A second real year of U.S. ED visits — Category 3 trained alone |
| Comparison | `/comparison` | All three real datasets combined — real respiratory rate, real clinical bands, no synthetic data |
| Model Selection | `/model-selection` | Why the Transformer was chosen: accuracy, MCC, size, and how its attention mechanism works |

Every chart on every page is hand-drawn inline SVG (`components/charts/VizCharts.tsx`) — no charting library, no external dependency, just the data.

## The dataset

Three real, public sources were combined into one 4-vital dataset — nothing synthetic:

| Source | Rows |
|---|---|
| Cardiac / ED Triage (Zenodo) | 5,808 |
| NHAMCS 2021 | 14,127 |
| NHAMCS 2022 | 14,009 |
| **Total** | **33,944** |

**Vitals:** heart rate, SpO2, body temperature, respiratory rate
**Scoring:** official NEWS2 chart (Dec 2022 edition, SpO2 Scale 1)
**Classes:** `Normal`, `Warning`, `High`, `Critical`
**Noise model (robustness testing):** ±5 bpm heart rate, ±2 breaths/min respiratory rate, ±2% SpO2, ±0.3°C temperature — simulating real sensor imprecision rather than scoring on clean lab data

## Model comparison

Five models were trained on the full combined dataset and scored on the **noisy** test set — the one that matters, since real sensor readings are never perfectly clean:

| Model | Accuracy | MCC | Size |
|---|---|---|---|
| Decision Tree | 68.2% | 0.567 | 5.4 KB |
| Random Forest | 67.8% | 0.566 | 1,776.9 KB |
| Neural Network | 67.7% | 0.564 | 34.4 KB |
| XGBoost | 67.8% | 0.564 | 425.8 KB |
| **Transformer** | **69.2%** | **0.58** | **13.3 KB** |

The Transformer wins on both accuracy and MCC while staying small enough for the ESP32's flash — 3,412 parameters, next to nothing beside Random Forest's 1.7+ MB (329x larger than Decision Tree).

### Why a Transformer, and how it works

```
Input:    x = [HR, SpO2, Temperature, RR]   (normalized)

1. Feature embedding
     each of the 4 vitals is projected into its own small embedding vector

2. Self-attention
     every vital "attends to" every other vital, learning a weight for
     how much it should influence the reading (e.g. a borderline SpO2
     can be weighted more heavily when RR is also elevated)

3. Feed-forward + non-linearity
     the attended features are combined and passed through a small
     dense layer

4. Classification head
     a final linear layer outputs one logit per health-status class

5. Softmax -> argmax
     logits become a probability per class; the highest-probability
     class is the prediction

Output:   Normal | Warning | High | Critical
```

A feature-importance pass across the dataset shows where the predictive signal sits — heart rate (46.7%) and respiratory rate (32.4%) carry almost 80% combined, with SpO2 at 17.1% and temperature at 3.9%. The ESP32-WROOM-32's Xtensa LX6 core includes a single-precision floating-point unit, so the Transformer's matrix multiplications and softmax run natively on-chip rather than through software emulation.

## System architecture

```
                SENSORS                           ESP32-WROOM-32                 OUTPUT
        ┌───────────────────┐             ┌───────────────────────────┐   ┌──────────────────┐
        │ MAX30102           │   I2C/PPG   │ 1. Read HR, SpO2, raw PPG │   │ OLED display      │
        │  → HR, SpO2, PPG   │────────────▶│ 2. Estimate RR from PPG   │──▶│ Buzzer / LED alert│
        │                     │             │ 3. NEWS2 banding          │   │ BLE → Skairo      │
        │ DS18B20             │   1-Wire    │ 4. On-device Transformer  │   │   Health app       │
        │  → Temperature      │────────────▶│    classification         │   └──────────────────┘
        └───────────────────┘             └───────────────────────────┘
                                              ▲ fully offline — no phone,
                                                no cloud, no internet required
```

> A hand-drawn concept sketch of the physical enclosure is in `public/figures/device_sketch_pencil.png` — the form factor shown there (wrist-worn, OLED + LED + buzzer on the front, sensors on the underside) is illustrative; it has not yet been built or tested as final hardware.

## Tech stack

- **Next.js** 14.2.35 (App Router)
- **React** 18.3.1 / **TypeScript** 5
- Zero charting dependencies — all visualizations are hand-written inline SVG
- Python scripts (`python/`) generate the underlying `data/*.json` chart data from the raw datasets

## Project structure

```
model-explainer-web/
├── app/
│   ├── page.tsx                 # Category 1 — Cardiac Triage
│   ├── category2/page.tsx       # Category 2 — NHAMCS 2021
│   ├── category3/page.tsx       # Category 3 — NHAMCS 2022
│   ├── comparison/page.tsx      # Comparison — all three datasets combined
│   ├── model-selection/page.tsx # Model Selection — Transformer vs. the rest
│   └── layout.tsx
├── components/
│   ├── ModelExplainer.tsx
│   ├── Category2Explainer.tsx
│   ├── Category3Explainer.tsx
│   ├── ComparisonExplainer.tsx
│   ├── ModelSelectionExplainer.tsx
│   ├── PageNav.tsx
│   └── charts/VizCharts.tsx     # dependency-free inline SVG chart library
├── data/                        # generated chart data (JSON) per page
├── python/                      # scripts that generate data/*.json from the raw datasets
└── public/
    ├── charts*/                 # exported chart PNGs
    └── figures/                 # hand-drawn concept figures
```

## Getting started

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view it.

| Script | What it does |
|---|---|
| `npm run dev` | Start the local dev server |
| `npm run build` | Production build |
| `npm run start` | Serve the production build |
| `npm run lint` | Lint the project |

## License

Released under the [MIT License](LICENSE).
