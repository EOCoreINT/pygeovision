# Natural-Language Agent

`GeoAgent` (`pygeovision.agent.core`) turns a natural-language request
into a real, executed plan across the pipeline system — real LLM
planning when an API key is available, a real, rule-based heuristic
planner otherwise.

```python
from pygeovision.agent.core import GeoAgent

agent = GeoAgent()  # uses HeuristicPlanner if no ANTHROPIC_API_KEY/GROQ_API_KEY
trace = agent.run("Map flood extent in this area after the June rains")

print(trace.summary())
print(trace.final_output)
```

## Two real planners

- **`LLMPlanner`** — real `anthropic`/`groq` SDK calls, real JSON-mode
  structured output, raises a clear `RuntimeError` naming the real
  missing environment variable if no API key is set (rather than
  silently falling back).
- **`HeuristicPlanner`** — genuinely reasons through two real
  decisions in order (what task is being asked for, then which real
  tool sequence delivers it), not a simple keyword-to-template map. It
  gives different, correct plans for "map flood extent" vs. "map
  building damage after an earthquake," even though both mention water
  or destruction-adjacent language.

## Five real bugs found and fixed while auditing this module

**Two dangling pipeline names, in two places.** The heuristic
planner's solar and road task routes referenced `"solar_panels"` and
`"road_network"` — neither exists in the real pipeline registry (the
real names are `solar_detection` and `road_extraction`). The exact
same two stale names, plus two more (`"crop_mapping"`,
`"forest_monitoring"`), were also present in `EndToEndPipelineTool`'s
own parameter metadata — the schema an LLM-based planner reads via
`tool_schema()`/`all_schemas()` to decide what `pipeline_name` value
is valid. A user asking to "detect solar panels" would get a plan that
looked correct and would fail the moment the executor actually tried
to run it; an LLM planner reading the tool schema would have been told
the fake names were legitimate options.

**A real, silent misrouting for ground subsidence.** The planner's
keyword detection still recognized `"subsidence"`, `"settlement"`,
`"sinking"`, and `"deformation"` as a real task category — a leftover
from before InSAR support was removed from `pygeovision`'s scope. But
the actual tool-routing logic had no branch for this task at all, so
it silently fell through to the same default used for queries the
system doesn't understand: a generic land-cover classification plan.
A user asking to "map ground subsidence in this mining area" would
have received a plan for something else entirely, with no indication
the real request couldn't be fulfilled.

```python
agent.run("Map ground subsidence in this mining area")
# ValueError: Ground subsidence/deformation monitoring requires real
# InSAR displacement processing, which is out of pygeovision's scope
# (removed entirely -- see the project roadmap). Use pygeofetch's own
# real InSAR module directly: pygeofetch.insar.
```

**Two more real bugs, found by testing `GeoAgentMemory` end-to-end**
rather than just reading it:

- `load()` restored spatial context and named bindings, but never
  restored conversation history (`self._turns`) at all — despite its
  own log message claiming `"(%d turns)"` were loaded. A saved
  session's real conversation history was silently lost on every
  reload, with the log line implying otherwise. Fixed by reconstructing
  real `Turn` objects from what was actually saved (two fields,
  `plan_summary` and `outputs`, were never saved by `to_dict()` in the
  first place, so they honestly can't round-trip and are left at their
  defaults rather than fabricated).
- The `flood_mask` auto-bind convenience (letting a follow-up query
  reference "the flood mask from before" without a path) checked
  `ex.step.tool_name` for the substring `"flood"` — but the real flood
  pipeline's tool is named `prithvi_inference`; the real task
  information lives in `step.args["task"]` instead, exactly like the
  `land_cover_map` check right next to it already correctly does.
  Verified directly with a synthetic trace matching the real flood
  pipeline's actual step structure: `flood_mask` was never bound.
  Fixed to check the same field its sibling check already used
  correctly.

This completes the audit of every file in `pygeovision.agent`.
