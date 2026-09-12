# How Multi-Step Plans Actually Execute

A real, verified placeholder-resolution system lets one tool's output
feed directly into the next tool's input — how a plan chains
`search_satellite_data` → `download_satellite_data` →
`prepare_for_ai` → `prithvi_inference` into one real, working
sequence.

## The real placeholder syntax

| Placeholder | Resolves to |
|---|---|
| `$step_N_output` | The real `output_path` (or `output` dict) of step `N` |
| `$step_N_output.key` | A specific real key inside step `N`'s output dict |
| `$context_key` | A value from the real shared context dict (e.g. `bbox`, `date_range`) |

```python
Step(1, "download_satellite_data", {
    "scenes": "$step_0_output.scenes",   # real list from search's output
    "output_dir": "./data/",
})
```

`PlanExecutor._resolve_args()` recursively resolves these through
nested dicts and lists before calling the real tool — verified
directly: a real `ToolResult` with `output={"scenes": [...], "count": 2}`
correctly resolves `$step_0_output.scenes` to the real list, and
`$context_bbox` correctly pulls from the real, shared context — with
plain literal values (a number, a fixed string) passed through
untouched.

## Real, honest failure propagation

```python
trace = executor.execute(plan)
trace.success  # True only if every real step succeeded
```

Every step's real result is recorded in `trace.executions`, even
failed ones. By default (`stop_on_failure=True`), the first real
failure halts the remaining plan — a step downstream of a failure
would only receive unresolved placeholders anyway, since there's no
real output to reference. An unknown tool name produces a real, honest
`ToolResult(success=False, error="Unknown tool: '...'")` rather than a
silent skip; a real exception inside a tool's `run()` is caught and
converted to the same honest failure shape rather than crashing the
whole agent.

```{seealso}
[Agent Tools — Full Reference](tools.md) for what each of the 8 real
tools referenced by `$step_N_output` placeholders actually returns.
```
