# Navix benchmarks

A benchmark scores an algorithm from-scratch, across every registered
navix environment, at a fixed frame budget. Two presets exist today:
`100K` and `1M` (see `navix.benchmarks.Navix100K`, `Navix1M`), one
folder per submitted entry.

## Submit your own entry

1. **Pick a preset** - `benchmarks/100K/` or `benchmarks/1M/`, or both.
2. **Create your entry's folder**: `benchmarks/<preset>/<your-entry>/`,
   e.g. `benchmarks/1M/my-algo/`.
3. **Write `run.py`** - subclass `AlgorithmEntry`, overriding `train`
   to build your algorithm for a given env_id and train it, then hand
   an instance to the preset:

   ```python
   from navix.benchmarks import AlgorithmEntry, Navix1M  # or Navix100K
   from navix.environments.registry import make

   class MyEntry(AlgorithmEntry):
       def train(self, env_id, budget, rng):
           env = make(env_id)
           agent = MyAgent(hparams=MyHparams(budget=budget), env=env)
           return agent.train(rng)

   entry = MyEntry(
       name=..., author=..., paper_url=...,
       navix_commit_url=..., algorithm_commit_url=...,
   )
   benchmark = Navix1M()
   raw = benchmark.run(entry)          # TrainingCurve, one env stacked per row
   summary = benchmark.summary(raw)    # Dict[str, Array] - the leaderboard's table row
   benchmark.submit_entry(entry, raw)  # writes summary.json/details.json/diagnostics.npz next to run.py
   ```

   It must run as `python run.py`, no arguments - read your own
   metadata from the sibling `config.yml` you add next, not from CLI
   flags. Per navix's "never vendor an external algorithm's code" rule
   ([#130](https://github.com/epignatelli/navix/issues/130)): your
   implementation stays in your own repo, and `train` (building
   whatever it needs for `env_id`, then training it) is the only seam
   that has to touch navix - it doesn't have to build a navix `Agent`
   at all, as long as it trains something and returns `(model, logs)`.
   `cost_analysis` has a protocol-agnostic default (compiles `train`
   and reads its FLOPs/memory/compile-time) - override it only if
   that default doesn't fit your algorithm.

4. **Add `config.yml`** with your entry's provenance:

   ```yaml
   name: MyAlgo
   author: your-github-handle
   paper_url: https://arxiv.org/abs/...
   navix_commit_url: https://github.com/epignatelli/navix/commit/<sha>
   algorithm_commit_url: https://github.com/<you>/<your-repo>/commit/<sha>
   ```

   `author` must be a valid GitHub handle; both commit URLs must be
   full URLs ending in a commit SHA, not a bare SHA - a SHA alone
   doesn't say which repo it's from. `AlgorithmEntry` validates both
   when `run.py` constructs it. For a navix-shipped agent,
   `algorithm_commit_url` is the same commit as `navix_commit_url`.

5. **Add `requirements.txt`** - pinned dependencies for `run.py`
   itself (not navix's own dependencies in general, not your
   algorithm repo's).
6. **Optional `README.md`** - free-form notes about your entry.
7. **Run `python run.py`** - besides printing a summary, this writes
   `summary.json`/`details.json`/`diagnostics.npz` into your entry's
   own folder (see below) - commit them alongside `run.py`/
   `config.yml`/`requirements.txt`, so your results are there to
   review without anyone re-running training.
8. **Open a PR** adding your `benchmarks/<preset>/<your-entry>/`
   folder. No submission portal yet (tracked in #130).

## What you get back

`run` returns a `TrainingCurve`. It holds:

- `episodic_returns`, `length`, `flops`, `memory_bytes`,
  `compile_time_seconds`, `fps`, `wall_time` - `episodic_returns`/
  `length`/`fps`/`wall_time` are per-update curves;
  `flops`/`memory_bytes`/`compile_time_seconds` (from
  `AlgorithmEntry.cost_analysis`) never have a time axis.
- `info: Dict` - free-form, for whatever an entry wants to attach
  beyond the fields above.

`Navix1M().run(entry)` stacks one `TrainingCurve` per environment
along a leading axis, with `info={"env_ids": (...)}` recording which
row is which. What ran it - `entry.gpu_type`/`cpu_type`/`ram_bytes`/
`cuda_version`/`cudnn_version`/`jax_version`/`jaxlib_version`,
auto-detected when `AlgorithmEntry(...)` is constructed - lives on
`entry` instead, since you already have it; it's the context
`fps`/`wall_time` need, since both are only meaningful across results
measured on the same hardware/software. Call `result.last_percent_mean()`
on a `TrainingCurve` to reduce its curve fields to scalars along their
trailing axis only (the last 20% of training, by default) - on `raw`
that means one scalar per environment, still stacked.
`last_percent_variance()` and `convergence_rate()` reduce the same
way, over the same window.

`benchmark.summary(raw)` returns `Dict[str, Array]` instead - the
leaderboard's table row for this entry: `episodic_returns`' bias
(`last_percent_mean`), `returns_variance` (`last_percent_variance`)
and `returns_convergence_rate` (`convergence_rate`), plus `flops`/
`memory_bytes`/`compile_time_seconds`/`fps`/`wall_time`'s bias - all
meaned across every environment in `raw`. `length` isn't a column
here (it's diagnostic, not something a leaderboard ranks on) but is
still on `raw` itself, and on `details` below. Which columns a
`summary` produces is protocol-specific - a different `Benchmark`
subclass can return an entirely different set of columns.

`benchmark.details(raw)` returns `Dict[str, Array]` too, but one row
per environment instead of a single aggregate - the same reduction
`summary` means further, stopped one step earlier (`length` included
this time). This is a breakdown of the *benchmark run itself* (how
did each environment individually score?), not a leaderboard's per-
algorithm click-through page - what that page plots for one entry
depends on the algorithm (a scalar curve plots differently from a
world model's rollout video), not on the benchmark protocol, so
that's for whatever reads `raw` (or your own `logs`) to decide, not
something `Benchmark` provides.

`benchmark.submit_entry(entry, raw)` writes three files, one per
level of a leaderboard page's progressive disclosure - cheapest and
coarsest first, so a reader only pays for what they actually open:

- `summary.json` - `entry`'s provenance/hardware fields plus
  `benchmark.summary(raw)`. What a table listing every entry reads.
- `details.json` - `env_ids` plus `benchmark.details(raw)`. What
  expanding one entry's row reads next.
- `diagnostics.npz` - `raw` itself, the full per-env training curves.
  What opening that entry's own click-through page reads last.

All three land in the directory of whichever script called
`submit_entry` - your entry's own folder, alongside `config.yml`.

See `navix/benchmarks/__init__.py`'s module docstring for the full
design, and [issue #130](https://github.com/epignatelli/navix/issues/130)
for the leaderboard proposal this implements.
