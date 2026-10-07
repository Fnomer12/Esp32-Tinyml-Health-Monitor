"use client";


import chartData from "@/data/chart-data-rr.json";
import { BarChart, GroupedBarChart, HorizontalBarChart, MODEL_COLORS } from "./charts/VizCharts";

const MODEL_ORDER = ["Decision Tree", "Random Forest", "Neural Network", "XGBoost", "Transformer"];

const noisy = MODEL_ORDER.map((m) => chartData.scorecards.find((s) => s.model === m && s.dataset === "noisy")!);
const sizeOf = (m: string) => chartData.modelSize.find((s) => s.model === m)!.kb;
const best = noisy.slice().sort((a, b) => b.accuracy - a.accuracy)[0];
const dt = noisy.find((n) => n.model === "Decision Tree")!;
const tfm = noisy.find((n) => n.model === "Transformer")!;

const SECTIONS = [
  { id: "rms1", title: "What changed" },
  { id: "rms2", title: "Noisy-data accuracy" },
  { id: "rms3", title: "MCC score" },
  { id: "rms4", title: "File size" },
  { id: "rms5", title: "How the Transformer works" },
  { id: "rms6", title: "Recommendation" },
];

export default function RRModelSelectionExplainer() {
  return (
    <>
      <div className="topbar">
        <div className="topbar-inner">
          <div className="brand">
            ESP32 HEALTH MONITOR &middot; <b>MODEL SELECTION &mdash; REAL DATA, FOUR VITALS</b>
          </div>
        </div>
      </div>

      <div className="wrap">
        <div className="hero">
          <div className="eyebrow">Which model goes on the chip &mdash; asked again, on real data</div>
          <h1>Model selection, with real data and respiratory rate</h1>
          <p className="lede">
            This re-asks the model question on the real, 3-source, 4-vital dataset ({chartData.totalRows.toLocaleString()}{" "}
            rows: Cardiac Triage, NHAMCS 2021, NHAMCS 2022), scored with the real NEWS2 chart. All five models,
            including the Transformer, are trained and compared on equal footing here &mdash; and it is the
            Transformer that this project deploys on the ESP32.
          </p>
        </div>

        <div className="layout">
          <nav className="side">
            <div className="side-title">Charts</div>
            <ul className="navlist">
              {SECTIONS.map((g, i) => (
                <li key={g.id}>
                  <a href={`#${g.id}`}>
                    <span className="num">{String(i + 1).padStart(2, "0")}</span>
                    {g.title}
                  </a>
                </li>
              ))}
            </ul>
          </nav>

          <div className="sections">
            <section className="card" id="rms1">
              <h2>Two things changed</h2>
              <p className="simple-explain">
                The data is now real, not synthetic, and has a fourth vital (respiratory rate, genuinely
                measured). The Transformer has also been retrained on this real, 4-vital shape and is now part
                of the comparison below, alongside Decision Tree, Random Forest, Neural Network, and XGBoost.
              </p>
            </section>

            <section className="card" id="rms2">
              <h2>Noisy-data accuracy</h2>
              <div className="chart-frame">
                <BarChart
                  data={noisy.map((n) => ({ label: n.model, value: n.accuracy, color: MODEL_COLORS[n.model] }))}
                  valueSuffix="%"
                />
              </div>
              <p className="simple-explain">
                {best.model} is highest at {best.accuracy}% under simulated noise, with Decision Tree close
                behind at {dt.accuracy}%. The other three land within a point of Decision Tree, too &mdash; on
                accuracy alone the five models are close, which is why file size and architecture end up
                mattering more for the final choice.
              </p>
            </section>

            <section className="card" id="rms3">
              <h2>MCC score</h2>
              <div className="chart-frame">
                <GroupedBarChart
                  categories={MODEL_ORDER}
                  series={[{ name: "MCC (noisy)", color: "#2a78d6", values: noisy.map((n) => n.mcc) }]}
                  showLegend={false}
                />
              </div>
              <p className="simple-explain">
                MCC is a stricter accuracy check that can&apos;t be fooled by uneven class sizes. The Transformer
                scores highest at {tfm.mcc}, with Decision Tree close behind at {dt.mcc} &mdash; the five models
                are statistically close here.
              </p>
            </section>

            <section className="card" id="rms4">
              <h2>File size</h2>
              <div className="chart-frame">
                <BarChart
                  data={MODEL_ORDER.map((m) => ({ label: m, value: sizeOf(m), color: MODEL_COLORS[m] }))}
                  valueSuffix=" KB"
                  logScale
                />
              </div>
              <p className="simple-explain">
                Decision Tree costs {sizeOf("Decision Tree")} KB, the Transformer {sizeOf("Transformer")} KB
                &mdash; both tiny next to Random Forest&apos;s {sizeOf("Random Forest").toLocaleString()} KB (
                {chartData.modelSizeRatio}x bigger than Decision Tree) and XGBoost&apos;s{" "}
                {sizeOf("XGBoost").toLocaleString()} KB.
              </p>
            </section>

            <section className="card" id="rms5">
              <h2>How the Transformer works</h2>
              <p className="simple-explain">
                The Transformer is a neural network built around <b>self-attention</b>: instead of applying one
                fixed set of if-else thresholds like a decision tree, it learns how much weight to give each
                input relative to the others, separately for every individual reading. Here the four inputs are
                the vitals themselves &mdash; heart rate, SpO2, temperature, and respiratory rate &mdash; so the
                model is, in effect, learning which combination of vitals matters most for the pattern in front
                of it, rather than applying one fixed rule to every patient.
              </p>
              <div className="chart-frame">
                <pre className="algo-block">
{`Input:    x = [HR, SpO2, Temperature, RR]   (normalized)

1. Feature embedding
     each of the 4 vitals is projected into its
     own small embedding vector

2. Self-attention
     every vital "attends to" every other vital,
     learning a weight for how much it should
     influence the reading (e.g. a borderline SpO2
     can be weighted more heavily when RR is also
     elevated)

3. Feed-forward + non-linearity
     the attended features are combined and passed
     through a small dense layer

4. Classification head
     a final linear layer outputs one logit per
     health-status class

5. Softmax -> argmax
     logits become a probability per class; the
     highest-probability class is the prediction

Output:   Normal | Warning | High | Critical`}
                </pre>
              </div>
              <p className="simple-explain">
                This Transformer is tiny by design &mdash; {chartData.transformerParamCount.toLocaleString()}{" "}
                parameters in total &mdash; trained on the full {chartData.totalRows.toLocaleString()}-row
                dataset. The project doesn&apos;t expose the model&apos;s raw attention weights, but a
                feature-importance pass across that same dataset shows where the predictive signal actually
                sits:
              </p>
              <div className="chart-frame">
                <HorizontalBarChart
                  data={chartData.featureImportance.map((f) => ({ label: f.feature, value: f.value * 100, color: "#e87ba4" }))}
                  valueSuffix="%"
                />
              </div>
              <p className="simple-explain">
                Heart rate and respiratory rate carry most of the weight
                ({(chartData.featureImportance[0].value * 100).toFixed(1)}% and{" "}
                {(chartData.featureImportance[1].value * 100).toFixed(1)}% &mdash; almost 80% combined), with
                SpO2 contributing {(chartData.featureImportance[2].value * 100).toFixed(1)}% and temperature the
                smallest share at {(chartData.featureImportance[3].value * 100).toFixed(1)}%. The
                Transformer&apos;s attention is expected to lean on the same two vitals most heavily, since it is
                learning from the same underlying data.
              </p>
            </section>

            <section className="card" id="rms6">
              <h2>Recommendation</h2>
              <p className="simple-explain">
                <b>The Transformer is the model this project deploys on the ESP32.</b> On the noisy test set
                &mdash; the one that matters, since real sensor readings are never perfectly clean &mdash; it
                scores highest of all five models: {tfm.accuracy}% accuracy and {tfm.mcc} MCC, edging out
                Decision Tree&apos;s {dt.accuracy}% and {dt.mcc}. It does this at {sizeOf("Transformer")} KB and{" "}
                {chartData.transformerParamCount.toLocaleString()} parameters &mdash; next to nothing compared
                with Random Forest&apos;s {sizeOf("Random Forest").toLocaleString()} KB or XGBoost&apos;s{" "}
                {sizeOf("XGBoost").toLocaleString()} KB, and still small enough to sit comfortably in the
                ESP32&apos;s flash once quantized.
              </p>
              <p className="simple-explain">
                The accuracy edge over Decision Tree is modest &mdash; about a single point &mdash; so the real
                case for the Transformer is architectural, not just a leaderboard win. Its attention mechanism
                lets the model weigh how vitals interact for each individual reading, rather than applying the
                same fixed thresholds to every patient. The ESP32-WROOM-32&apos;s Xtensa LX6 core includes a
                single-precision floating-point unit, so the matrix multiplications and softmax a Transformer
                needs at inference run natively on-chip rather than through slow software emulation &mdash;
                keeping classification fast enough for real-time use.
              </p>
              <p className="simple-explain">
                Adding a fourth vital, switching to real data, and bringing the Transformer into the comparison
                changed the answer from an earlier, smaller-scale round: on this dataset, the Transformer is
                both the most accurate model under noise and small enough for the chip.
              </p>
            </section>
          </div>
        </div>
      </div>
    </>
  );
}
