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
   result = Navix1M.run(entry)
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

`Navix1M.run(entry)` returns a `BenchmarkResult`:

- `summary: Metrics` - each field's last-fifth-of-training mean,
  meaned across every environment. The single-number score.
- `history: Dict[str, Metrics]` - one `Metrics` per `env_id`, holding
  full per-update curves for `returns`/`episode_length`/`fps`/
  `wall_time` (plot these directly for training curves) - call
  `.last_fifth_mean()` on one to reduce it to a per-env scalar.
  `flops`/`memory_bytes`/`compile_time_seconds` come from
  `Agent.cost_analysis`, a single measurement rather than a curve, so
  they stay scalar here too.

This shape only fits the from-scratch-per-environment protocol above
- a future continual-learning or one-shot-generalisation benchmark
would need its own result type.

See [issue #130](https://github.com/epignatelli/navix/issues/130) for
the full design rationale behind `Benchmark`/`AlgorithmEntry`/
`BenchmarkResult`.
