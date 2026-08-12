# Discovery task prompt (v2)

The text below is the `--task` payload for `artemis discovery create`. It encodes the
lessons from run `b35519db` (2026-08-12), where a mis-specified gate assertion cost
~15 of 18 versions. Keep the four sections — goal, verified facts, protocol,
metric contract — when adapting it for other tasks.

---

## Task text (copy verbatim into `--task`)

Enable native bidirectional LSTMSequence execution through oneDNN in the Intel GPU
plugin. Preserve forward/reverse behavior and decomposition/TensorIterator fallback
for unsupported cases (non-zero clip, non-default activations, dynamic shapes,
partial sequence lengths, devices without oneDNN). Then improve measured
native-vs-decomposed latency without adding debug code or unnecessary copies.

VERIFIED FACTS (runner-confirmed; you may build on these):
- cldnn format ybfx requested for lstm_seq output 0 is byte-identical to oneDNN's
  TNC dst_layer [T,N,2H] over OpenVINO [N,D,T,H]; the zero-copy Y binding is
  correct as-is. Do not add private buffers, chained reorders, or output repacking.
- The oneDNN GPU RNN implementation reports impl_info_str "ocl:simple:any"; exec-graph
  implementation names are not evidence of cldnn-OCL selection.
- The create-path and cache-load-path W/R descriptor index formulas differ
  intentionally: create sees pre-reorder W [D,4H,I], load sees post-reorder
  [D,I,4H]. Do not unify them.
- Everything else from earlier runs is hypothesis, not fact. Before building on any
  prior conclusion, reproduce it against a failing gate and the runner log.

PROTOCOL (mandatory):
- For every failing gate, quote the exact failing assertion or comparison line from
  the runner log and classify it as (a) harness assertion, (b) numerical mismatch,
  or (c) build failure, before proposing any implementation change.
- "completed" does not mean "passed": read gate outcomes and metric values, not
  lifecycle status.
- Do not spend versions re-running unchanged code (the platform rejects identical
  changesets); repetition for noise estimation belongs inside the benchmark.
- Allowed files: src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp,
  src/plugins/intel_gpu/src/graph/impls/onednn/lstm_seq_onednn.cpp,
  src/plugins/intel_gpu/src/graph/impls/onednn/lstm_seq_onednn.hpp,
  src/plugins/intel_gpu/src/graph/impls/onednn/primitive_onednn_base.h, and the
  functional tests under src/plugins/intel_gpu/tests/functional/. Edits outside
  this set need a stated reason.

METRIC CONTRACT (mandatory):
- Use worker metrics only. Do not create agent/LLM-scored metrics; if any exist in
  the schema, they must carry importance 0.
- Importance: bidir_b10_native_single_primitive 0.25,
  bidir_b1_native_single_primitive 0.15, bidir_b10_speedup_x 0.35,
  bidir_b1_speedup_x 0.25; all other metrics 0.
- A version is only a success when all focused gates pass AND
  bidir_b1_native_matches_decomposed = bidir_b10_native_matches_decomposed = 1 AND
  both native_single_primitive metrics are 1. Speedup on mismatched outputs is
  worthless.

---

## Post-create verification (operator step, before letting the baseline proceed)

```bash
artemis --output-format json discovery get <run-id> | jq '.metricsSchema'
```

Require: zero entries with `"source": "agent"`, and the four importance weights
above. If the schema differs, stop the run and recreate with corrected task text —
the schema is derived by the agent at create time and cannot be fixed afterwards.
