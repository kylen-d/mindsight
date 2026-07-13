# Pipeline YAML: Loader Internals

This page documents **how the YAML loader works** — the code path a config file
takes through `mindsight/config_compat.py`, the precedence rules, and how a plugin
author's flags flow from YAML into `from_args`. For the authoritative **catalogue
of valid keys**, see [Pipeline YAML Schema](../reference/pipeline-yaml-schema.md);
this page does not repeat it.

```bash
python MindSight.py --pipeline my_pipeline.yaml   # CLI: merges YAML into the namespace
```

---

## Two entry points

`mindsight/config_compat.py` exposes two loaders with different jobs:

| Function | Returns | Used by |
|---|---|---|
| `load_pipeline(path, ns=None)` | a merged argparse `Namespace` | the CLI (`--pipeline`) and the GUI import path |
| `load_yaml(path)` | a `PipelineConfig` (pydantic) over schema defaults | the schema-based config route / round-trip export |

Both read the file with `yaml.safe_load` and flatten nested sections into
dot-separated paths via `_flatten` (`detection.model`, `gaze.ray_length`, ...).
The difference is the target: `load_pipeline` sets attributes on an argparse
namespace through **`_YAML_MAP`**; `load_yaml` builds a `PipelineConfig` through
**`YAML_ALIASES`** (old keys → canonical schema paths, with `PATH_MIRRORS`
fan-out). The rest of this page follows the `load_pipeline` (CLI/GUI) route, since
that is what plugin authors interact with.

---

## The flow: YAML → argparse → from_args

```
my_pipeline.yaml
      │  yaml.safe_load + _flatten
      ▼
flat dot-paths ── _YAML_MAP ─────────►  ns.<dest>        (core flags)
               ── plugins: passthrough ►  ns.<dest>        (ANY plugin flag)
               ── phenomena: list ──────►  ns.<toggle/param>
               ── aux_streams: list ────►  ns.aux_streams (AuxStreamConfig list)
      │
      ▼
argparse Namespace  ──►  factory.build_from_namespace(ns)  ──►  plugin.from_args(ns)
```

`load_pipeline` mutates the namespace in four steps (see
`config_compat.load_pipeline`):

1. **Flat/nested sections** — every flattened key found in `_YAML_MAP` is copied to
   its argparse dest (subject to precedence, below).
2. **Phenomena list** — `phenomena:` is a **list** of strings or single-key dicts;
   toggles resolve through `_PHENOMENA_TOGGLES` and params through
   `_PHENOMENA_PARAMS`.
3. **Plugins passthrough** — `plugins:` is a flat dict whose keys map **directly**
   to argparse dests (hyphens → underscores).
4. **Aux streams** — `aux_streams:` is a list of dicts parsed into
   `AuxStreamConfig` instances.

---

## `_YAML_MAP`: keys are an explicit allowlist

`_YAML_MAP` is a fixed `dict[yaml_dot_path -> argparse_dest]`. Only keys present in
this table reach the namespace; everything else is **silently ignored** — no error,
no warning. This is the single most common reason a hand-written YAML "does
nothing": a mistyped or invented key is simply dropped.

Because the mapping is explicit, the dot-path and the argparse dest often differ.
The notable renames:

| YAML key | argparse dest |
|---|---|
| `output.save_video` | `save` |
| `output.log_csv` | `log` |
| `output.summary_csv` | `summary` |
| `output.heatmaps` | `heatmap` |
| `participants.csv` / `participants.ids` | `participant_csv` / `participant_ids` |
| `depth.enabled` | `depth` |

!!! warning "Phantom keys that silently no-op"
    These keys appear in older docs but are **not** in `_YAML_MAP`, so the loader
    drops them: `detection.device`, `detection.imgsz`, `detection.tracker`,
    `gaze.backend`, `output.heatmap` (the real key is `output.heatmaps`),
    `output.output_dir`, `performance.frame_skip`, `performance.resize`, and a
    nested/list `participants:` block. `participants` must be `participants.csv`
    or `participants.ids`; performance keys are `fast`, `skip_phenomena`,
    `lite_overlay`, `no_dashboard`, `profile`. The valid catalogue is in the
    [schema reference](../reference/pipeline-yaml-schema.md).

---

## Precedence: `_explicit_cli` vs the `_is_default` heuristic

Whether a YAML value may overwrite a namespace attribute is decided by
`_should_set(ns, attr, explicit)`:

- **CLI route.** `mindsight.cli._args` attaches `ns._explicit_cli` — the exact
  frozenset of dests the user actually typed on the command line. When present,
  **YAML wins for every dest the user did not type**, and the legacy heuristic is
  bypassed entirely. So `--pipeline foo.yaml` applies cleanly, but a flag you also
  typed on the command line always beats the file.
- **GUI / synthetic route.** A namespace without `_explicit_cli` falls back to
  `_is_default(ns, attr)`: a value is treated as "default-like" (and therefore
  overwritable by YAML) when it is `None`, `False`, `0`, `0.0`, or an empty
  list/string. This is a best-effort check — argparse does not record which values
  the user set — so a deliberately-`False` GUI value can be overwritten by YAML.

Resolution order, effectively:

1. Explicitly-typed CLI flags (highest).
2. YAML values.
3. argparse defaults (lowest).

---

## `plugins:` passthrough — how a plugin author's flags flow

The `plugins:` section is the mechanism a plugin author relies on. It is a **flat**
dict (not nested per-plugin blocks), and every key maps directly to an argparse
dest with hyphens converted to underscores:

```yaml
plugins:
  gaze-boost: true          # -> ns.gaze_boost = True
  gaze-boost-factor: 1.5    # -> ns.gaze_boost_factor = 1.5
  iris-refine: true         # -> ns.iris_refine = True
```

There is no hardcoded per-plugin mapping — **any** argparse dest can be set this
way, which is exactly how the bundled known-good preset drives ~20 core flags
through `plugins:`. The value lands on the namespace, `factory.build_from_namespace`
runs, and the plugin's `from_args(ns)` reads it back via `getattr(ns, "gaze_boost",
...)`. This is why plugin `from_args` methods use `getattr` with defaults: the dest
may or may not have been set by YAML, CLI, or argparse.

!!! note
    A nested `plugins: {my_plugin: {threshold: 0.8}}` form does **not** work — the
    value assigned to `ns.my_plugin` would be a dict, not the scalar the flag
    expects. Flatten it to `plugins: {my-plugin-threshold: 0.8}`.

---

## `phenomena:` list format

```yaml
phenomena:
  - mutual_gaze                 # bare string: toggle on with defaults
  - joint_attention:            # dict: toggle + per-tracker params
      ja_window: 10
      ja_quorum: 0.6
```

Toggle names resolve through `_PHENOMENA_TOGGLES` (e.g. `social_referencing` →
`social_ref`, `gaze_following` → `gaze_follow`); params resolve through
`_PHENOMENA_PARAMS`. A mapping-style `phenomena: {joint_attention: true}` fails the
`isinstance(item, list)` check and is dropped — `phenomena` must be a **list**.

## `aux_streams:` list format

Each entry is a dict with `source`, `video_type`, `stream_label`, `participants`,
and optional `auto_detect_faces`. Entries missing `source`, `stream_label`, or
`participants` are skipped; an unknown `video_type` warns and falls back to
`custom`. Valid `video_type` values are `wide_closeup`, `face_closeup`,
`eye_only`, and `custom`.

## `adaptive_ray` back-compat shim

`gaze.adaptive_ray` is now an enum (`off` / `extend` / `snap`). For legacy files
that set it to a bool, `load_pipeline` maps `true` → `snap` when
`gaze.adaptive_snap` is also set, else `extend`; `false` → `off`.

---

## Reference

- [Pipeline YAML Schema](../reference/pipeline-yaml-schema.md) — the authoritative,
  validated catalogue of every key and its type.
- [Config Dataclasses](../reference/config-dataclasses.md) — the dataclasses the
  namespace is ultimately turned into.
