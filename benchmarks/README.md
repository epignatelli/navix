# Navix benchmarks

A benchmark scores an algorithm from-scratch, across every registered
navix environment, at a fixed frame budget. Two presets exist today:
`100K` and `1M` (see `navix.benchmark.Navix100K`, `Navix1M`), one
folder per submitted entry.

## Submit your own entry

1. **Pick a preset** - `benchmarks/100K/` or `benchmarks/1M/`, or both.
2. **Create your entry's folder**: `benchmarks/<preset>/<your-entry>/`,
   e.g. `benchmarks/1M/my-algo/`.
3. **Write `run.py`** - build your algorithm as a navix `Agent` per
   environment and hand it to the preset:

   ```python
   from navix.benchmark import AlgorithmEntry, Navix1M  # or Navix100K

   def make_agent(env):
       ...  # your Environment -> Agent factory
       return MyAgent(hparams=..., env=env)

   entry = AlgorithmEntry(
       name=..., author=..., paper_url=...,
       navix_commit_url=..., algorithm_commit_url=...,
       agent_factory=make_agent,
   )
   raw = Navix1M.run(entry)          # Dict[str, BenchmarkResult], one per env_id
   summary = Navix1M.summary(raw)    # one BenchmarkResult, aggregated - the leaderboard's table row
   details = Navix1M.details(raw)    # what the leaderboard shows on click-through
   ```

   It must run as `python run.py`, no arguments - read your own
   metadata from the sibling `config.yml` you add next, not from CLI
   flags. Per navix's "never vendor an external algorithm's code" rule
   ([#130](https://github.com/epignatelli/navix/issues/130)): your
   implementation stays in your own repo, and `agent_factory` is the
   only seam that has to touch navix.

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
7. **Open a PR** adding your `benchmarks/<preset>/<your-entry>/`
   folder. No submission portal yet (tracked in #130).

## What you get back

`Navix1M.run(entry)` returns `Dict[str, BenchmarkResult]`, one
`BenchmarkResult` per `env_id`. Each `BenchmarkResult` holds:

- `returns`, `episode_length`, `flops`, `memory_bytes`,
  `compile_time_seconds`, `fps`, `wall_time` - `returns`/
  `episode_length`/`fps`/`wall_time` are full per-update curves;
  `flops`/`memory_bytes`/`compile_time_seconds` (from
  `Agent.cost_analysis`) are always scalar.
- `info: Dict` - free-form, for whatever an entry wants to attach
  beyond the fields above.

Call `navix.benchmark.last_percent_mean` on a `BenchmarkResult` via
`jax.tree.map(last_percent_mean, result)` to reduce its curve fields
to scalars (the last 20% of training, by default).

`Navix1M.summary(raw)` reduces and averages `raw` across every
environment into one aggregate `BenchmarkResult` - the single
comparable score a leaderboard's table shows. `Navix1M.details(raw)`
returns the content a leaderboard shows for one entry beyond that -
for this preset, `raw` itself (full per-environment training curves).

See `navix/benchmark.py`'s module docstring for the full design, and
[issue #130](https://github.com/epignatelli/navix/issues/130) for the
leaderboard proposal this implements.
