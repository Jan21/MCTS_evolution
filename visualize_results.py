"""Visualization and export utilities for benchmark results."""

from __future__ import annotations

import json

from A_star import SolveResult


# -- Console display -----------------------------------------------------------

def format_comparison_table(
    algo_names: list[str],
    all_results: dict[str, dict[int, tuple[SolveResult, float | None]]],
    env_indices: list[int],
):
    """Print a side-by-side comparison table of algorithm results."""
    col_w = max(7, *(len(n) for n in algo_names))

    header = "           " + "".join(f"{n:>{col_w}}" for n in algo_names)
    print(header)

    metrics_by_algo: dict[str, list[float]] = {n: [] for n in algo_names}

    for idx in sorted(env_indices):
        row = f"env_{idx:<5d} "
        for name in algo_names:
            res = all_results[name]
            if idx in res:
                _, metric = res[idx]
                if metric is not None:
                    row += f"{metric:>{col_w}.0f}"
                    metrics_by_algo[name].append(metric)
                else:
                    row += f"{'FAIL':>{col_w}}"
            else:
                row += f"{'-':>{col_w}}"
        print(row)

    print()

    row = f"{'avg':<11}"
    for name in algo_names:
        m = metrics_by_algo[name]
        if m:
            row += f"{sum(m) / len(m):>{col_w}.2f}"
        else:
            row += f"{'-':>{col_w}}"
    print(row)

    row = f"{'min':<11}"
    for name in algo_names:
        m = metrics_by_algo[name]
        if m:
            row += f"{min(m):>{col_w}.0f}"
        else:
            row += f"{'-':>{col_w}}"
    print(row)

    row = f"{'max':<11}"
    for name in algo_names:
        m = metrics_by_algo[name]
        if m:
            row += f"{max(m):>{col_w}.0f}"
        else:
            row += f"{'-':>{col_w}}"
    print(row)

    row = f"{'solved':<11}"
    for name in algo_names:
        res = all_results[name]
        total = len(res)
        solved = len(metrics_by_algo[name])
        row += f"{f'{solved}/{total}':>{col_w}}"
    print(row)


def format_discovery_stats(
    algo_names: list[str],
    all_results: dict[str, dict[int, tuple[SolveResult, float | None]]],
):
    """Print plan discovery stats from SolveResult.all_plans."""
    for name in algo_names:
        results = all_results[name]
        total_plans = 0
        best_iters = []
        best_times = []

        for _, (solve_result, _) in results.items():
            plans = solve_result.all_plans
            total_plans += len(plans)
            if plans:
                best_entry = min(plans, key=lambda p: p.cost)
                best_iters.append(best_entry.stats.iteration)
                best_times.append(best_entry.stats.wall_time)

        parts = [f"{name}: found {total_plans} plans"]
        if best_iters:
            avg_iter = sum(best_iters) / len(best_iters)
            avg_time = sum(best_times) / len(best_times)
            parts.append(f"best at iter {avg_iter:.0f} ({avg_time:.2f}s)")
        print(", ".join(parts))


# -- JSON export ---------------------------------------------------------------

def build_json_data(
    algo_names: list[str],
    all_results: dict[str, dict[int, tuple[SolveResult, float | None]]],
) -> dict:
    """Build a JSON-serializable dict of all benchmark results."""
    data: dict = {"algorithms": {}}
    for name in algo_names:
        algo_data: dict = {"environments": {}, "discovery": []}
        results = all_results[name]
        for env_idx, (solve_result, metric) in sorted(results.items()):
            algo_data["environments"][str(env_idx)] = {
                "cost": metric,
                "plans_found": len(solve_result.all_plans),
            }
            for entry in solve_result.all_plans:
                algo_data["discovery"].append({
                    "env": env_idx,
                    "cost": entry.cost,
                    "iteration": entry.stats.iteration,
                    "wall_time": round(entry.stats.wall_time, 4),
                    "node_count": entry.stats.node_count,
                    "rollout_count": entry.stats.rollout_count,
                })
        metrics = [m for m in (r[1] for r in results.values()) if m is not None]
        algo_data["summary"] = {
            "solved": len(metrics),
            "total": len(results),
            "avg_cost": round(sum(metrics) / len(metrics), 2) if metrics else None,
            "min_cost": min(metrics) if metrics else None,
            "max_cost": max(metrics) if metrics else None,
        }
        data["algorithms"][name] = algo_data
    return data


def save_json(data: dict, path: str = "benchmark_results.json"):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {path}")


# -- HTML generation -----------------------------------------------------------

def generate_html(data: dict, path: str = "benchmark_results.html"):
    algo_names = list(data["algorithms"].keys())
    env_indices = sorted(
        set().union(*(
            (int(k) for k in ad["environments"])
            for ad in data["algorithms"].values()
        ))
    )

    colors = [
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
        "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    ]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>A* Benchmark Results</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #f5f5f5; color: #333; padding: 24px; }}
  h1 {{ margin-bottom: 8px; }}
  .subtitle {{ color: #666; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; }}
  .card {{ background: #fff; border-radius: 8px; padding: 20px;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .card.full {{ grid-column: 1 / -1; }}
  .card h2 {{ margin-bottom: 16px; font-size: 1.1em; color: #555; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
  th, td {{ padding: 8px 12px; text-align: right; border-bottom: 1px solid #eee; }}
  th {{ background: #fafafa; font-weight: 600; }}
  td:first-child, th:first-child {{ text-align: left; }}
  tr.summary {{ font-weight: 600; background: #f9f9f9; }}
  tr.summary td {{ border-top: 2px solid #ddd; }}
  .best {{ color: #2a7; font-weight: 600; }}
  .fail {{ color: #c33; }}
  .chart-container {{ position: relative; height: 300px; }}
  .chart-container.tall {{ height: 400px; }}
  .summary-cards {{ display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }}
  .stat-card {{ background: #fff; border-radius: 8px; padding: 16px 20px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1); min-width: 150px; }}
  .stat-card .label {{ font-size: 0.8em; color: #888; text-transform: uppercase; }}
  .stat-card .value {{ font-size: 1.5em; font-weight: 700; margin-top: 4px; }}
</style>
</head>
<body>
<h1>A* Benchmark Results</h1>
<p class="subtitle">{len(env_indices)} environments, {len(algo_names)} algorithms: {', '.join(algo_names)}</p>

<div class="summary-cards">
"""
    for i, name in enumerate(algo_names):
        s = data["algorithms"][name]["summary"]
        color = colors[i % len(colors)]
        html += f"""  <div class="stat-card" style="border-left: 4px solid {color}">
    <div class="label">{name} avg cost</div>
    <div class="value">{s['avg_cost'] if s['avg_cost'] is not None else '-'}</div>
    <div class="label">{s['solved']}/{s['total']} solved</div>
  </div>
"""

    html += """</div>
<div class="grid">

<!-- Cost Comparison Table -->
<div class="card">
<h2>Per-Environment Costs</h2>
<table>
<tr><th>Env</th>"""

    for name in algo_names:
        html += f"<th>{name}</th>"
    html += "</tr>\n"

    for idx in env_indices:
        costs = {}
        for name in algo_names:
            envs = data["algorithms"][name]["environments"]
            c = envs.get(str(idx), {}).get("cost")
            if c is not None:
                costs[name] = c
        best_cost = min(costs.values()) if costs else None

        html += f"<tr><td>env_{idx}</td>"
        for name in algo_names:
            envs = data["algorithms"][name]["environments"]
            c = envs.get(str(idx), {}).get("cost")
            if c is None:
                html += '<td class="fail">FAIL</td>'
            elif c == best_cost and len(costs) > 1:
                html += f'<td class="best">{c:.0f}</td>'
            else:
                html += f"<td>{c:.0f}</td>"
        html += "</tr>\n"

    for label, key in [("Avg", "avg_cost"), ("Min", "min_cost"), ("Max", "max_cost")]:
        html += f'<tr class="summary"><td>{label}</td>'
        for name in algo_names:
            v = data["algorithms"][name]["summary"][key]
            fmt = f"{v:.2f}" if key == "avg_cost" and v is not None else (f"{v:.0f}" if v is not None else "-")
            html += f"<td>{fmt}</td>"
        html += "</tr>\n"

    html += """</table>
</div>

<!-- Bar Chart: Per-env costs -->
<div class="card">
<h2>Cost Comparison</h2>
<div class="chart-container">
<canvas id="costChart"></canvas>
</div>
</div>

<!-- Discovery Timeline -->
<div class="card full">
<h2>Plan Discovery Timeline (cost vs iteration)</h2>
<div class="chart-container tall">
<canvas id="timelineChart"></canvas>
</div>
</div>

<!-- Discovery Timeline by wall time -->
<div class="card full">
<h2>Plan Discovery Timeline (cost vs wall time)</h2>
<div class="chart-container tall">
<canvas id="wallTimeChart"></canvas>
</div>
</div>

</div>

<script>
const data = """ + json.dumps(data) + """;
const algoNames = """ + json.dumps(algo_names) + """;
const envIndices = """ + json.dumps(env_indices) + """;
const colors = """ + json.dumps(colors[:len(algo_names)]) + """;

// Bar chart: per-env costs
{
  const datasets = algoNames.map((name, i) => ({
    label: name,
    data: envIndices.map(idx => {
      const c = data.algorithms[name].environments[String(idx)]?.cost;
      return c ?? 0;
    }),
    backgroundColor: colors[i] + '99',
    borderColor: colors[i],
    borderWidth: 1,
  }));

  new Chart(document.getElementById('costChart'), {
    type: 'bar',
    data: { labels: envIndices.map(i => 'env_' + i), datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'top' } },
      scales: { y: { beginAtZero: true, title: { display: true, text: 'Cost' } } },
    }
  });
}

// Discovery timeline: cost vs iteration
{
  const datasets = [];
  algoNames.forEach((name, i) => {
    const disc = data.algorithms[name].discovery;
    if (!disc.length) return;

    const scatterData = disc.map(d => ({ x: d.iteration, y: d.cost }));
    datasets.push({
      label: name + ' (all plans)',
      data: scatterData,
      backgroundColor: colors[i] + '40',
      borderColor: colors[i],
      pointRadius: 2,
      showLine: false,
      type: 'scatter',
    });

    const sorted = [...disc].sort((a,b) => a.iteration - b.iteration);
    let runningBest = {};
    const bestLine = [];
    sorted.forEach(d => {
      if (!runningBest[d.env] || d.cost < runningBest[d.env]) {
        runningBest[d.env] = d.cost;
        const vals = Object.values(runningBest);
        bestLine.push({ x: d.iteration, y: vals.reduce((a,b)=>a+b,0) / vals.length });
      }
    });

    datasets.push({
      label: name + ' (avg best)',
      data: bestLine,
      borderColor: colors[i],
      borderWidth: 2,
      pointRadius: 0,
      showLine: true,
      type: 'line',
      fill: false,
    });
  });

  new Chart(document.getElementById('timelineChart'), {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'top' } },
      scales: {
        x: { title: { display: true, text: 'Iteration' } },
        y: { title: { display: true, text: 'Plan Cost' }, beginAtZero: true },
      },
    }
  });
}

// Wall time chart
{
  const datasets = [];
  algoNames.forEach((name, i) => {
    const disc = data.algorithms[name].discovery;
    if (!disc.length) return;

    const scatterData = disc.map(d => ({ x: d.wall_time, y: d.cost }));
    datasets.push({
      label: name + ' (all plans)',
      data: scatterData,
      backgroundColor: colors[i] + '40',
      borderColor: colors[i],
      pointRadius: 2,
      showLine: false,
      type: 'scatter',
    });

    const sorted = [...disc].sort((a,b) => a.wall_time - b.wall_time);
    let runningBest = {};
    const bestLine = [];
    sorted.forEach(d => {
      if (!runningBest[d.env] || d.cost < runningBest[d.env]) {
        runningBest[d.env] = d.cost;
        const vals = Object.values(runningBest);
        bestLine.push({ x: d.wall_time, y: vals.reduce((a,b)=>a+b,0) / vals.length });
      }
    });

    datasets.push({
      label: name + ' (avg best)',
      data: bestLine,
      borderColor: colors[i],
      borderWidth: 2,
      pointRadius: 0,
      showLine: true,
      type: 'line',
      fill: false,
    });
  });

  new Chart(document.getElementById('wallTimeChart'), {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'top' } },
      scales: {
        x: { title: { display: true, text: 'Wall Time (s)' } },
        y: { title: { display: true, text: 'Plan Cost' }, beginAtZero: true },
      },
    }
  });
}
</script>
</body>
</html>"""

    with open(path, "w") as f:
        f.write(html)
    print(f"Visualization saved to {path}")
