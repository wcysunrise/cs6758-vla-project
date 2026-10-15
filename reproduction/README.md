# Executed source provenance

`executed_source/` preserves the cloud source files whose SHA256 values are in
`results/executed_source_sha256.json`. The final root evaluator has the identical
Python AST and the config is byte-identical to the executed cloud version.
Final documentation reflects actual results; the final Notebook also creates a
fresh reproduction output directory and exports tests. The copied cloud Notebook
is the prepared interface, not an executed Notebook transcript; genuine execution
evidence is the raw console logs, per-step traces, manifests and videos.
