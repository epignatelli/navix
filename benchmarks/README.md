# Navix benchmarks

Each preset (`1K/`, `100K/`, `1M/` - see `navix.benchmarks.Navix1K`,
`Navix100K`, `Navix1M`) scores an algorithm from-scratch, across every
registered navix environment, at a fixed training budget (1,000 /
100,000 / 1,000,000 frames per environment respectively). `1K` is not a
performance benchmark - the budget is too small for any algorithm to
converge - it's a fast, cheap smoke test that an entry runs end-to-end.

## Structure

One folder per algorithm entry, per preset:

```
benchmarks/<preset>/<entry>/
  run.py              # entrypoint - must run with `python run.py`, no arguments
  config.yml           # name, author, paper_url, navix_sha, algorithm_sha
  requirements.txt     # pinned dependencies for this entry's own driver code
  README.md            # optional - free-form notes specific to this entry
```

- **`run.py`** builds the algorithm's `Agent` for each environment and calls the preset's `.run()` (e.g. `Navix1M(entry).run()`, see `navix.benchmarks.Benchmark`). It must be runnable as `python run.py` with no CLI arguments - anything it needs comes from the sibling `config.yml`.
- **`config.yml`** holds this entry's static, frozen provenance: `name`, `author` (this implementation's author, not necessarily the paper's), `paper_url`, `navix_sha` (the navix commit this result was produced against), and `algorithm_sha` (the commit of the algorithm implementation's own repo - equal to `navix_sha` for a navix-shipped agent like PPO/Dreamer/PQN, since the implementation lives in this repo).
- **`requirements.txt`** pins this entry's own driver code's dependencies - not navix's own dependencies in general, and not an external algorithm's own repo's (which manages those itself).

This mirrors the per-entry folder layout from [issue #130](https://github.com/epignatelli/navix/issues/130) (the navix leaderboard proposal), which is also where the design rationale for `Benchmark`/`AlgorithmEntry`/`BenchmarkResult` lives.

## What a `BenchmarkResult` reports

Every entry's result is shaped identically, regardless of algorithm - the same fixed metric set every navix agent's logs already support (`navix.plotting.MANDATORY_METRICS`), aggregated across every environment the preset covers: `returns`, `success_rate`, `episode_length`, `fps`, `wall_time`. `success_rate` is the metric to compare across algorithms (bounded `[0, 1]`, comparable regardless of an environment's raw reward scale); `wall_time`/`fps` are the cost columns. Per-environment, per-update detail stays available on `result.logs` for deeper inspection.

## Submitting a new entry

Per issue #130's "never vendor an external algorithm's code" rule: if you didn't write the algorithm as part of navix's own team, its implementation stays in your own repo - navix's job is to make it runnable and reproducible against navix environments, not to own a copy of it. `AlgorithmEntry.agent_factory` is the seam for this - point it at your own training entrypoint, wrapped to build an `Environment -> Agent`-shaped agent, the same way `benchmarks/<preset>/navix-ppo/run.py` does for navix's own PPO.

There's no submission portal yet (tracked in #130) - for now, open a PR adding your `benchmarks/<preset>/<your-entry>/` folder in the shape described above.
