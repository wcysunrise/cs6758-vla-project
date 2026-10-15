# Actual execution status

- Actual GPU smoke: completed successfully; 162 steps; 177.802042 s; excluded from formal metrics.
- Formal evaluation: 5/5 completed; 5 successes; 0 task failures.
- Success rate: 100% on one task and five fixed initial states.
- Model strict-loaded and frozen; FP32; no training.
- All formal MP4s decode fully and match CSV SHA256 values and trace lengths.
- Four offline protocol/integrity tests passed; scripts and Notebook cells compile.
- Startup errors: checkpoint config factory and inherited Matplotlib backend; targeted fixes applied before completed formal episodes. Logs retained.
- Runtime shutdown is recorded in cloud_shutdown.json after exported results are verified.
- GitHub delivery: reviewed source and genuine results are published with explicit user authorization via the existing GitHub browser account.

These are measured robot outcomes. Historic local macOS preflight failures and
empty schema files are excluded from this result directory.
