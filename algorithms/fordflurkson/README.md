Ford–Fulkerson implementation (mirror of Dinic's structure)

This folder mirrors the `algorithms/dinics` layout and provides a Ford–Fulkerson
implementation compatible with the existing visualization and analysis tooling.

Key points:
- `code/ff.py` : Ford–Fulkerson implementation (DFS by default, `method='bfs'` for Edmonds–Karp)
- `code/runner.py` : Single-graph runner that produces the same JSON layout as Dinic's runner
- `code/batch_run.py` : Batch runner matching Dinic's behavior
- `code/data_store.py` : Save/load execution JSON (includes optional detailed iteration logs)
- `code/graph_loader.py`, `code/graph_generator.py`, `code/metrics.py` : Utilities copied from Dinic to keep behavior consistent

Use the `visualizer` from `algorithms/dinics` without modification; the JSON layout matches Dinic's expectations.
