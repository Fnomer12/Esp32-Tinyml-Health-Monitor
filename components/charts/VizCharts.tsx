/**
 * VizCharts
 * ----------
 * Small, dependency-free interactive SVG charts (hover tooltips via native
 * <title>, direct value labels, legends) used to replace the static
 * matplotlib PNGs on the explainer pages. No charting library — plain SVG,
 * driven straight from chart-data.json, so the numbers can never drift
 * from the pictures.
 */

export const STATUS_COLORS: Record<string, string> = {
  Normal: "#0ca30c",
  Warning: "#fab219",
  High: "#ec835a",
  Critical: "#d03b3b",
};

export const MODEL_COLORS: Record<string, string> = {
  "Decision Tree": "#eb6834",
  "Random Forest": "#2a78d6",
  "Neural Network": "#1baf7a",
  XGBoost: "#eda100",
  Transformer: "#e87ba4",
};

const INK = "#1c231e";
const MUTED = "#566056";
const BASELINE = "#c3c2b7";
const GRID = "#e1e0d9";

function fmt(v: number, suffix: string, decimals: number = 1) {
  const scale = Math.pow(10, decimals);
  const r = Math.round(v * scale) / scale;
  return `${Number.isInteger(r) ? r : r.toFixed(decimals)}${suffix}`;
}

export function Legend({ items }: { items: { label: string; color: string }[] }) {
  return (
    <div className="viz-legend">
      {items.map((it) => (
        <span key={it.label} className="viz-legend-item">
          <span className="viz-swatch" style={{ background: it.color }} />
          {it.label}
        </span>
      ))}
    </div>
  );
}

type BarDatum = { label: string; value: number; color: string };

export function BarChart({
  data,
  height = 260,
  valueSuffix = "",
  logScale = false,
  decimals = 1,
}: {
  data: BarDatum[];
  height?: number;
  valueSuffix?: string;
  logScale?: boolean;
  decimals?: number;
}) {
  const width = 600;
  const padding = { top: 30, right: 16, bottom: 40, left: 16 };
  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;
  const barGap = 14;
  const barW = (plotW - barGap * (data.length - 1)) / data.length;

  const values = data.map((d) => d.value);
  const hasNeg = values.some((v) => v < 0);
  const max = Math.max(...values, 0) * (logScale ? 1 : 1.2) || 1;
  const min = hasNeg ? Math.min(...values, 0) * 1.2 : 0;
  const range = max - min || 1;

  const scaleY = (v: number) => {
    if (logScale) {
      const lv = Math.log10(Math.max(v, 0.05) + 1);
      const lmax = Math.log10(max + 1) || 1;
      return padding.top + plotH * (1 - lv / lmax);
    }
    return padding.top + plotH * (1 - (v - min) / range);
  };
  const zeroY = scaleY(0);

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Bar chart: ${data.map((d) => `${d.label} ${fmt(d.value, valueSuffix, decimals)}`).join(", ")}`}
      style={{ width: "100%", height: "auto", display: "block" }}
    >
      {[0.25, 0.5, 0.75].map((t) => (
        <line key={t} x1={padding.left} x2={width - padding.right} y1={padding.top + plotH * t} y2={padding.top + plotH * t} stroke={GRID} strokeWidth={1} />
      ))}
      <line x1={padding.left} x2={width - padding.right} y1={zeroY} y2={zeroY} stroke={BASELINE} strokeWidth={1} />
      {data.map((d, i) => {
        const x = padding.left + i * (barW + barGap);
        const y0 = zeroY;
        const y1 = scaleY(d.value);
        const barY = Math.min(y0, y1);
        const barH = Math.max(Math.abs(y1 - y0), 2);
        const labelY = d.value >= 0 ? barY - 7 : barY + barH + 14;
        return (
          <g key={d.label} className="viz-bar-g">
            <title>{`${d.label}: ${fmt(d.value, valueSuffix, decimals)}`}</title>
            <rect x={x} y={barY} width={barW} height={barH} rx={4} fill={d.color} className="viz-rect" />
            <text x={x + barW / 2} y={labelY} textAnchor="middle" fontSize={12} fontWeight={700} fill={INK}>
              {fmt(d.value, valueSuffix, decimals)}
            </text>
            <text x={x + barW / 2} y={padding.top + plotH + 18} textAnchor="middle" fontSize={11.5} fill={MUTED}>
              {d.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export function HorizontalBarChart({
  data,
  height,
  valueSuffix = "",
}: {
  data: BarDatum[];
  height?: number;
  valueSuffix?: string;
}) {
  const width = 600;
  const rowH = 46;
  const h = height ?? data.length * rowH + 20;
  const padding = { top: 10, right: 50, bottom: 10, left: 110 };
  const plotW = width - padding.left - padding.right;
  const max = Math.max(...data.map((d) => d.value)) * 1.15 || 1;

  return (
    <svg
      viewBox={`0 0 ${width} ${h}`}
      role="img"
      aria-label={`Horizontal bar chart: ${data.map((d) => `${d.label} ${fmt(d.value, valueSuffix)}`).join(", ")}`}
      style={{ width: "100%", height: "auto", display: "block" }}
    >
      {data.map((d, i) => {
        const y = padding.top + i * rowH;
        const barW = (d.value / max) * plotW;
        const barH = rowH - 16;
        return (
          <g key={d.label}>
            <title>{`${d.label}: ${fmt(d.value, valueSuffix)}`}</title>
            <text x={padding.left - 10} y={y + barH / 2 + 4} textAnchor="end" fontSize={12.5} fill={INK} fontWeight={600}>
              {d.label}
            </text>
            <rect x={padding.left} y={y} width={plotW} height={barH} rx={4} fill="#f1f2ef" />
            <rect x={padding.left} y={y} width={Math.max(barW, 3)} height={barH} rx={4} fill={d.color} className="viz-rect" />
            <text x={padding.left + barW + 8} y={y + barH / 2 + 4} fontSize={12} fontWeight={700} fill={INK}>
              {fmt(d.value, valueSuffix)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export function GroupedBarChart({
  categories,
  series,
  height = 280,
  valueSuffix = "",
  showLegend = true,
}: {
  categories: string[];
  series: { name: string; color: string; values: number[] }[];
  height?: number;
  valueSuffix?: string;
  showLegend?: boolean;
}) {
  const width = 600;
  const padding = { top: 30, right: 16, bottom: 40, left: 16 };
  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;
  const groupGap = 22;
  const groupW = (plotW - groupGap * (categories.length - 1)) / categories.length;
  const barGap = 4;
  const barW = (groupW - barGap * (series.length - 1)) / series.length;

  const allValues = series.flatMap((s) => s.values);
  const hasNeg = allValues.some((v) => v < 0);
  const max = Math.max(...allValues, 0) * 1.2 || 1;
  const min = hasNeg ? Math.min(...allValues, 0) * 1.2 : 0;
  const range = max - min || 1;
  const scaleY = (v: number) => padding.top + plotH * (1 - (v - min) / range);
  const zeroY = scaleY(0);

  return (
    <div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`Grouped bar chart comparing ${series.map((s) => s.name).join(", ")} across ${categories.join(", ")}`}
        style={{ width: "100%", height: "auto", display: "block" }}
      >
        {[0.25, 0.5, 0.75].map((t) => (
          <line key={t} x1={padding.left} x2={width - padding.right} y1={padding.top + plotH * t} y2={padding.top + plotH * t} stroke={GRID} strokeWidth={1} />
        ))}
        <line x1={padding.left} x2={width - padding.right} y1={zeroY} y2={zeroY} stroke={BASELINE} strokeWidth={1} />
        {categories.map((cat, ci) => {
          const gx = padding.left + ci * (groupW + groupGap);
          return (
            <g key={cat}>
              {series.map((s, si) => {
                const v = s.values[ci];
                const bx = gx + si * (barW + barGap);
                const y0 = zeroY;
                const y1 = scaleY(v);
                const barY = Math.min(y0, y1);
                const barH = Math.max(Math.abs(y1 - y0), 2);
                const labelY = v >= 0 ? barY - 5 : barY + barH + 12;
                return (
                  <g key={s.name}>
                    <title>{`${s.name} — ${cat}: ${fmt(v, valueSuffix)}`}</title>
                    <rect x={bx} y={barY} width={barW} height={barH} rx={3} fill={s.color} className="viz-rect" />
                    <text x={bx + barW / 2} y={labelY} textAnchor="middle" fontSize={10} fontWeight={700} fill={INK}>
                      {fmt(v, valueSuffix)}
                    </text>
                  </g>
                );
              })}
              <text x={gx + groupW / 2} y={padding.top + plotH + 20} textAnchor="middle" fontSize={11.5} fill={MUTED}>
                {cat}
              </text>
            </g>
          );
        })}
      </svg>
      {showLegend && series.length > 1 && <Legend items={series.map((s) => ({ label: s.name, color: s.color }))} />}
    </div>
  );
}

const BLUE_SEQ = ["#fbfbfa", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"];

export function ConfusionMatrix({
  matrix,
  classNames,
}: {
  matrix: number[][];
  classNames: string[];
}) {
  const n = classNames.length;
  const cell = 62;
  const labelW = 86;
  const labelH = 34;
  const width = labelW + n * cell;
  const height = labelH + n * cell + 6;
  const maxVal = Math.max(...matrix.flat(), 1);

  const colorFor = (v: number) => {
    if (v === 0) return BLUE_SEQ[0];
    const t = Math.sqrt(v / maxVal);
    const idx = Math.max(1, Math.min(BLUE_SEQ.length - 1, Math.round(t * (BLUE_SEQ.length - 1))));
    return BLUE_SEQ[idx];
  };

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Confusion matrix: rows are the actual class, columns are the predicted class, for ${classNames.join(", ")}`}
      style={{ width: "100%", height: "auto", display: "block" }}
    >
      {classNames.map((c, j) => (
        <text key={c} x={labelW + j * cell + cell / 2} y={labelH - 10} textAnchor="middle" fontSize={11} fill={MUTED}>
          {c}
        </text>
      ))}
      {matrix.map((row, i) =>
        row.map((v, j) => {
          const x = labelW + j * cell;
          const y = labelH + i * cell;
          const bg = colorFor(v);
          const isDiag = i === j;
          const light = v / maxVal > 0.55;
          return (
            <g key={`${i}-${j}`}>
              <title>{`Actual ${classNames[i]}, predicted ${classNames[j]}: ${v}`}</title>
              <rect
                x={x + 1}
                y={y + 1}
                width={cell - 2}
                height={cell - 2}
                rx={4}
                fill={bg}
                stroke={isDiag ? "#1f6f4a" : "#e0e4de"}
                strokeWidth={isDiag ? 2 : 1}
                className="viz-rect"
              />
              <text x={x + cell / 2} y={y + cell / 2 + 4} textAnchor="middle" fontSize={13} fontWeight={isDiag ? 700 : 500} fill={light ? "#ffffff" : INK}>
                {v}
              </text>
            </g>
          );
        })
      )}
      {classNames.map((c, i) => (
        <text key={c} x={labelW - 8} y={labelH + i * cell + cell / 2 + 4} textAnchor="end" fontSize={11} fill={MUTED}>
          {c}
        </text>
      ))}
      <text x={6} y={labelH + 2} fontSize={10} fill={MUTED}>
        Actual ↓ / Predicted →
      </text>
    </svg>
  );
}

/* ---------------------------------------------------------------------- */
/* RecallBarChart — per-class recall where some classes genuinely have no */
/* real test examples (Category 2's High/Critical). A "no data" class is  */
/* drawn as a dashed, hollow outline labeled "no data" instead of a       */
/* misleading 0% bar.                                                     */
/* ---------------------------------------------------------------------- */
export function RecallBarChart({
  data,
  height = 260,
}: {
  data: { name: string; recall: number; support: number; hasData?: boolean }[];
  height?: number;
}) {
  const width = 600;
  const padding = { top: 30, right: 16, bottom: 40, left: 16 };
  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;
  const barGap = 14;
  const barW = (plotW - barGap * (data.length - 1)) / data.length;
  const max = 110;
  const scaleY = (v: number) => padding.top + plotH * (1 - v / max);
  const zeroY = scaleY(0);
  const noDataH = plotH * 0.3;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Recall by class: ${data
        .map((d) => (d.hasData === false ? `${d.name}, no data` : `${d.name}, ${d.recall}% recall`))
        .join(", ")}`}
      style={{ width: "100%", height: "auto", display: "block" }}
    >
      {[0.25, 0.5, 0.75].map((t) => (
        <line key={t} x1={padding.left} x2={width - padding.right} y1={padding.top + plotH * t} y2={padding.top + plotH * t} stroke={GRID} strokeWidth={1} />
      ))}
      <line x1={padding.left} x2={width - padding.right} y1={zeroY} y2={zeroY} stroke={BASELINE} strokeWidth={1} />
      {data.map((d, i) => {
        const x = padding.left + i * (barW + barGap);
        const color = STATUS_COLORS[d.name] ?? "#999";
        const noData = d.hasData === false;
        const barH = noData ? noDataH : Math.max(zeroY - scaleY(d.recall), 2);
        const barY = noData ? zeroY - barH : scaleY(d.recall);
        return (
          <g key={d.name}>
            <title>{noData ? `${d.name}: no real cases in this dataset` : `${d.name}: ${d.recall}% recall, out of ${d.support.toLocaleString()} real cases`}</title>
            <rect
              x={x}
              y={barY}
              width={barW}
              height={barH}
              rx={4}
              fill={noData ? "none" : color}
              stroke={color}
              strokeWidth={noData ? 1.5 : 0}
              strokeDasharray={noData ? "4 3" : undefined}
              opacity={noData ? 0.6 : 1}
              className="viz-rect"
            />
            <text x={x + barW / 2} y={barY - 7} textAnchor="middle" fontSize={11.5} fontWeight={700} fill={noData ? MUTED : INK}>
              {noData ? "no data" : `${d.recall}%`}
            </text>
            <text x={x + barW / 2} y={padding.top + plotH + 18} textAnchor="middle" fontSize={11.5} fill={MUTED}>
              {d.name}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/* ---------------------------------------------------------------------- */
/* DecisionTreeDiagram — renders the model's own recursive yes/no rules   */
/* straight from the training script's tree JSON. The real trees grow     */
/* far too deep to show in full (thousands of nodes), so this draws the   */
/* top `maxDepth` real questions and folds everything below a cut branch  */
/* into one pale "N more decisions below" box — a true, honest summary    */
/* of what's hidden, not a fake shallow tree.                             */
/* ---------------------------------------------------------------------- */
type DTLeaf = { leaf: boolean; prediction: string; counts: number[]; totalSamples: number };
type DTInternal = { leaf: boolean; feature: string; threshold: number; left: DTNode; right: DTNode };
// The raw JSON (via Next.js resolveJsonModule) infers `leaf` as plain `boolean`,
// not a discriminated `true | false` literal, so this type can't be a strict
// discriminated union -- it's a superset shape and we narrow with `.leaf` at
// runtime instead (see isLeafNode below).
type DTNode = DTLeaf & Partial<DTInternal> | DTInternal & Partial<DTLeaf>;

function isLeafNode(node: DTNode): node is DTLeaf & { leaf: true } {
  return !!node.leaf;
}

type DTDisplayNode = {
  x: number;
  y: number;
  isLeaf: boolean;
  truncated: boolean;
  feature?: string;
  threshold?: number;
  prediction?: string;
  counts: number[];
  totalSamples: number;
  left?: DTDisplayNode;
  right?: DTDisplayNode;
  hiddenCount?: number;
};

function dtSummarize(node: DTNode): { totalSamples: number; counts: number[] } {
  if (isLeafNode(node)) return { totalSamples: node.totalSamples!, counts: node.counts! };
  const l = dtSummarize(node.left!);
  const r = dtSummarize(node.right!);
  return { totalSamples: l.totalSamples + r.totalSamples, counts: l.counts.map((c, i) => c + r.counts[i]) };
}

function dtCountInternal(node: DTNode): number {
  if (isLeafNode(node)) return 0;
  return 1 + dtCountInternal(node.left!) + dtCountInternal(node.right!);
}

export function DecisionTreeDiagram({
  tree,
  classNames,
  maxDepth = 3,
}: {
  tree: DTNode;
  classNames: string[];
  maxDepth?: number;
}) {
  const colW = 134;
  const rowH = 96;
  let slot = 0;

  function build(node: DTNode, depth: number): DTDisplayNode {
    const atLimit = depth >= maxDepth;
    const leaf = isLeafNode(node);
    if (leaf || atLimit) {
      const x = slot * colW + colW / 2;
      slot += 1;
      if (leaf) {
        return { x, y: depth * rowH, isLeaf: true, truncated: false, prediction: node.prediction, counts: node.counts!, totalSamples: node.totalSamples! };
      }
      const sum = dtSummarize(node);
      const hidden = dtCountInternal(node);
      const topIdx = sum.counts.indexOf(Math.max(...sum.counts));
      return { x, y: depth * rowH, isLeaf: true, truncated: true, prediction: classNames[topIdx], counts: sum.counts, totalSamples: sum.totalSamples, hiddenCount: hidden };
    }
    const left = build(node.left!, depth + 1);
    const right = build(node.right!, depth + 1);
    return { x: (left.x + right.x) / 2, y: depth * rowH, isLeaf: false, truncated: false, feature: node.feature, threshold: node.threshold, counts: [], totalSamples: 0, left, right };
  }

  const root = build(tree, 0);
  const width = Math.max(slot * colW, colW);
  const height = (maxDepth + 1) * rowH + 36;

  const nodes: DTDisplayNode[] = [];
  const edges: { x1: number; y1: number; x2: number; y2: number; label: string }[] = [];
  (function walk(n: DTDisplayNode) {
    nodes.push(n);
    if (!n.isLeaf && n.left && n.right) {
      edges.push({ x1: n.x, y1: n.y + 24, x2: n.left.x, y2: n.left.y, label: "yes" });
      edges.push({ x1: n.x, y1: n.y + 24, x2: n.right.x, y2: n.right.y, label: "no" });
      walk(n.left);
      walk(n.right);
    }
  })(root);

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Decision tree diagram showing the top ${maxDepth} levels of yes/no questions the model learned, with deeper branches folded into summary boxes`}
      style={{ width: "100%", height: "auto", display: "block" }}
    >
      {edges.map((e, i) => (
        <g key={i}>
          <line x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2} stroke={BASELINE} strokeWidth={1.5} />
          <text x={(e.x1 + e.x2) / 2} y={(e.y1 + e.y2) / 2 - 4} textAnchor="middle" fontSize={10} fill={MUTED}>
            {e.label}
          </text>
        </g>
      ))}
      {nodes.map((n, i) => {
        if (n.isLeaf) {
          const topIdx = n.counts.indexOf(Math.max(...n.counts));
          const color = STATUS_COLORS[classNames[topIdx]] ?? "#999";
          const boxH = n.truncated ? 58 : 44;
          return (
            <g key={i}>
              <title>
                {n.truncated
                  ? `${n.hiddenCount} more real questions below this point, covering ${n.totalSamples.toLocaleString()} training samples, leaning ${n.prediction}`
                  : `Leaf: ${n.prediction}, ${n.totalSamples.toLocaleString()} samples`}
              </title>
              <rect x={n.x - 56} y={n.y} width={112} height={boxH} rx={6} fill={n.truncated ? "#fff" : color} opacity={n.truncated ? 1 : 0.92} stroke={color} strokeWidth={n.truncated ? 1.5 : 0} strokeDasharray={n.truncated ? "4 3" : undefined} />
              <text x={n.x} y={n.y + 18} textAnchor="middle" fontSize={11} fontWeight={700} fill={n.truncated ? INK : "#fff"}>
                {n.truncated ? `${n.prediction}-leaning` : n.prediction}
              </text>
              <text x={n.x} y={n.y + 32} textAnchor="middle" fontSize={9.5} fill={n.truncated ? MUTED : "#fff"}>
                {n.totalSamples.toLocaleString()} samples
              </text>
              {n.truncated && (
                <text x={n.x} y={n.y + 48} textAnchor="middle" fontSize={9} fill={MUTED}>
                  +{n.hiddenCount} more below
                </text>
              )}
            </g>
          );
        }
        return (
          <g key={i}>
            <title>{`Is ${n.feature} ≤ ${n.threshold}?`}</title>
            <rect x={n.x - 60} y={n.y} width={120} height={26} rx={5} fill="#fff" stroke={INK} strokeWidth={1.2} />
            <text x={n.x} y={n.y + 17} textAnchor="middle" fontSize={10.5} fontWeight={600} fill={INK}>
              {n.feature} ≤ {n.threshold}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/* ---------------------------------------------------------------------- */
/* Histogram — stacked, per-class binned distribution for one sensor.     */
/* Replaces the old feature_distributions.png: real bin counts straight   */
/* from the training data, stacked and colored by health status, with a   */
/* hover tooltip per segment.                                             */
/* ---------------------------------------------------------------------- */
export function Histogram({
  binEdges,
  counts,
  classNames,
  unit = "",
  height = 220,
}: {
  binEdges: number[];
  counts: Record<string, number[]>;
  classNames: string[];
  unit?: string;
  height?: number;
}) {
  const width = 600;
  const padding = { top: 14, right: 12, bottom: 32, left: 12 };
  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;
  const nBins = binEdges.length - 1;
  const barGap = 1;
  const barW = (plotW - barGap * (nBins - 1)) / nBins;

  const totals = Array.from({ length: nBins }, (_, i) => classNames.reduce((sum, c) => sum + (counts[c]?.[i] ?? 0), 0));
  const maxTotal = Math.max(...totals, 1);
  const scaleH = (v: number) => (v / maxTotal) * plotH;

  const fmtEdge = (v: number) => (Number.isInteger(v) ? `${v}` : v.toFixed(1));

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Histogram from ${fmtEdge(binEdges[0])}${unit} to ${fmtEdge(binEdges[binEdges.length - 1])}${unit}, stacked by health status`}
      style={{ width: "100%", height: "auto", display: "block" }}
    >
      {Array.from({ length: nBins }).map((_, i) => {
        const x = padding.left + i * (barW + barGap);
        let yCursor = padding.top + plotH;
        return (
          <g key={i}>
            {classNames.map((c) => {
              const v = counts[c]?.[i] ?? 0;
              if (v === 0) return null;
              const h = scaleH(v);
              const y = yCursor - h;
              yCursor = y;
              return (
                <g key={c}>
                  <title>{`${fmtEdge(binEdges[i])}–${fmtEdge(binEdges[i + 1])}${unit}, ${c}: ${v.toLocaleString()}`}</title>
                  <rect x={x} y={y} width={Math.max(barW, 1)} height={Math.max(h, 0)} fill={STATUS_COLORS[c] ?? "#999"} className="viz-rect" />
                </g>
              );
            })}
          </g>
        );
      })}
      <text x={padding.left} y={height - 8} textAnchor="start" fontSize={10.5} fill={MUTED}>
        {fmtEdge(binEdges[0])}{unit}
      </text>
      <text x={width - padding.right} y={height - 8} textAnchor="end" fontSize={10.5} fill={MUTED}>
        {fmtEdge(binEdges[binEdges.length - 1])}{unit}
      </text>
    </svg>
  );
}
