"""Tiny HTTP server for live MCTS debugging.

Serves the visualizer HTML with trace data loaded via polling,
so you can step through MCTS in the IDE debugger and see the
tree/DAG/grid update in real time.

Usage:
    1. Set a breakpoint in MCTS/v1.py's main loop
    2. Start this server:  python live_debug_server.py
    3. Open http://localhost:8075 in browser
    4. Run your MCTS code with debug_hook:

        from mcts_tracer import MCTSTracer
        algo = MCTS_V1(cfg=cfg, trace=True)
        result = algo.solve(grid_env, state,
            debug_hook=lambda algo, it: algo.tracer.dump_live())

    5. Each time the debugger hits the hook (or you step past it),
       the JSON file updates and the browser auto-refreshes.

    Alternatively, call algo.tracer.dump_live() manually in the
    VS Code debug console at any breakpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEMPLATE_PATH = ROOT / "mcts_visualizer.html"
TRACE_FILENAME = "mcts_live_trace.json"


def build_live_html() -> str:
    """Patch the visualizer HTML to load trace data via fetch + polling."""
    template = TEMPLATE_PATH.read_text()

    # Replace the embedded placeholder with a fetch-based loader
    loader_js = f"""
(function() {{
  // Live-mode: fetch trace JSON and poll for updates
  const TRACE_URL = "/{TRACE_FILENAME}";
  const POLL_MS = 500;
  let _lastLen = 0;

  async function fetchTrace() {{
    try {{
      const resp = await fetch(TRACE_URL + "?t=" + Date.now());
      if (!resp.ok) return null;
      const text = await resp.text();
      if (text.length === _lastLen) return null;  // no change
      _lastLen = text.length;
      return JSON.parse(text);
    }} catch(e) {{ return null; }}
  }}

  // Initial load: wait for first valid trace, then boot the app
  async function boot() {{
    let data = null;
    const status = document.getElementById("live-status");
    while (!data) {{
      data = await fetchTrace();
      if (!data) {{
        if (status) status.textContent = "Waiting for {TRACE_FILENAME}...";
        await new Promise(r => setTimeout(r, POLL_MS));
      }}
    }}
    if (status) status.textContent = "Connected";
    window.__LIVE_TRACE = data;
    window.__initApp(data);

    // Poll for updates
    setInterval(async () => {{
      const fresh = await fetchTrace();
      if (fresh) {{
        window.__LIVE_TRACE = fresh;
        window.__reloadTrace(fresh);
      }}
    }}, POLL_MS);
  }}

  if (document.readyState === "loading") {{
    document.addEventListener("DOMContentLoaded", boot);
  }} else {{
    boot();
  }}
}})();
"""

    # We need to:
    # 1. Replace `const TRACE = __TRACE_DATA_PLACEHOLDER__;` with a placeholder
    # 2. Wrap the init in functions the loader can call
    # 3. Add a status indicator

    # Add live status badge to the top bar
    patched = template.replace(
        '<span class="title">MCTS Debugger</span>',
        '<span class="title">MCTS Debugger</span>'
        '<span id="live-status" style="font-size:11px;color:#2ecc71;'
        'margin-left:4px;"></span>'
    )

    # Replace the data init + the init() call at the bottom with
    # function-wrapped versions that the loader calls
    patched = patched.replace(
        "const TRACE = __TRACE_DATA_PLACEHOLDER__;",
        "let TRACE = {};  // will be set by live loader"
    )

    # Wrap init() so it can be called by the loader with data
    patched = patched.replace(
        "\ninit();\n",
        """
window.__initApp = function(data) {
  // Re-initialize all data structures from fresh trace
  TRACE = data;
  _reloadDataStructures();
  init();
};

window.__reloadTrace = function(data) {
  // Hot-reload: preserve current position if possible
  const oldIter = getCurrentIterationIndex();
  const oldStepInIter = currentSubStep - (iterationBoundaries[oldIter] || 0);
  const wasAtEnd = currentSubStep >= allSubSteps.length - 1;

  TRACE = data;
  _reloadDataStructures();

  // Restore position or jump to latest
  if (wasAtEnd && allSubSteps.length > 0) {
    // Was at end -> jump to new end
    setSubStep(allSubSteps.length - 1);
  } else if (oldIter < iterationBoundaries.length) {
    // Try to restore same position
    const newStart = iterationBoundaries[oldIter] || 0;
    setSubStep(newStart + oldStepInIter);
  } else if (allSubSteps.length > 0) {
    setSubStep(allSubSteps.length - 1);
  }

  // Update slider range
  slider.max = Math.max(0, events.length - 1);
};

function _reloadDataStructures() {
  // Clear and rebuild from TRACE
  events.length = 0;
  (TRACE.events || []).forEach(e => events.push(e));

  Object.keys(treeNodesData).forEach(k => delete treeNodesData[k]);
  (TRACE.tree_nodes || []).forEach(n => { treeNodesData[n.id] = {...n}; });

  treeEdgesData.length = 0;
  (TRACE.tree_edges || []).forEach(e => { treeEdgesData.push(e); });

  Object.assign(grid, TRACE.grid || {});
  Object.assign(initialState, TRACE.initial_state || {});
  Object.assign(initialPlan, TRACE.initial_plan || {});

  // Rebuild sub-steps
  allSubSteps.length = 0;
  iterationBoundaries.length = 0;
  events.forEach((ev, i) => {
    iterationBoundaries.push(allSubSteps.length);
    const steps = buildSubSteps(ev, i);
    steps.forEach((s, j) => {
      allSubSteps.push({ ...s, eventIndex: i, stepIndex: j, totalSteps: steps.length });
    });
  });

  // Rebuild lookup
  Object.keys(expandedChildToEvent).forEach(k => delete expandedChildToEvent[k]);
  events.forEach((ev, i) => {
    if (ev.expanded_child_id !== null && ev.expanded_child_id !== undefined) {
      expandedChildToEvent[ev.expanded_child_id] = i;
    }
  });
}

// Don't auto-init; wait for live loader
"""
    )

    # Inject the loader script right before </head>
    patched = patched.replace("</head>", f"<script>{loader_js}</script>\n</head>")

    return patched


class LiveHandler(SimpleHTTPRequestHandler):
    """Serve the live visualizer and trace JSON."""

    def __init__(self, *args, html_content="", trace_dir="", **kwargs):
        self._html = html_content
        self._trace_dir = trace_dir
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            content = self._html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", len(content))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)
        elif self.path.startswith(f"/{TRACE_FILENAME}"):
            trace_path = Path(self._trace_dir) / TRACE_FILENAME
            if trace_path.exists():
                content = trace_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", len(content))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, f"{TRACE_FILENAME} not found yet")
        else:
            super().do_GET()

    def log_message(self, format, *args):
        # Suppress polling noise
        if TRACE_FILENAME in str(args):
            return
        super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description="Live MCTS debug server")
    parser.add_argument("--port", type=int, default=8075)
    parser.add_argument("--trace-dir", type=str, default=str(ROOT),
                        help=f"Directory containing {TRACE_FILENAME}")
    args = parser.parse_args()

    html_content = build_live_html()
    trace_dir = args.trace_dir

    def handler(*a, **kw):
        return LiveHandler(*a, html_content=html_content,
                          trace_dir=trace_dir, **kw)

    server = HTTPServer(("localhost", args.port), handler)
    print(f"Live MCTS debugger at http://localhost:{args.port}")
    print(f"Watching for {TRACE_FILENAME} in {trace_dir}")
    print("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
