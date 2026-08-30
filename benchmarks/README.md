# Navix benchmarks

A benchmark scores an algorithm from scratch, independently across a
fixed set of navix environments, at a fixed frame budget. Two presets
exist today - `100K` and `1M` (see `navix.benchmarks.Navix100K`,
`Navix1M`) - one folder per submitted entry, per preset.

## How it works

You give a benchmark an `AlgorithmEntry`: a subclass overriding
`train(self, env_id, budget, rng)` to build your algorithm for a given
environment and train it, returning a `TrainingCurve` -
`episodic_returns`/`lengths` (masked-mean over completed episodes,
one point per training update) plus an optional `diagnostics` dict for
whatever else helps debug your algorithm (loss, learning rate, ...).
That's the only requirement - your `train` doesn't have to build a
navix `Agent` at all, as long as it trains something and returns a
`TrainingCurve`.

`train` runs inside a `jax.jit`/`jax.vmap` trace, so it can only
return things that are actually computable there. Wall-clock timing
and cost can't be - a jitted function's `time.time()` calls only fire
once, at trace time, not per real call - so a benchmark measures those
itself, from outside any trace: `benchmark.run(entry)` gives you a
`BenchmarkResult`, `TrainingCurve` plus `wall_time`/`fps` (real,
`jax.block_until_ready`-timed measurements, compile time excluded) and
`cost` (a `CostAnalysis` - FLOPs, peak memory, compile time - from
compiling `train` separately).

A preset (`Navix1M`/`Navix100K`) trains your entry independently
across `env_ids` (a small, curated default spanning several
environment families - not every registered environment; override it
by subclassing if you want different coverage) and `seeds` (10 by
default, more than one always required - vmapped together, so there's
no per-seed timing breakdown), then stacks one `BenchmarkResult` per
environment along a leading axis.

`benchmark.summary(raw)` reduces that down to one leaderboard table
row: `episodic_returns`' bias (mean over the last 20% of training),
`returns_variance` and `returns_convergence_rate` (how much it still
fluctuates, how fast it got there), plus `flops`/`memory_bytes`/
`compile_time_seconds`/`fps`/`wall_time`'s bias - all meaned across
every environment. `benchmark.details(raw)` gives you the same
columns one step earlier - one row per environment, plus `env_ids` so
you know which row is which and `lengths` (not on `summary`, since a
leaderboard doesn't rank on it, but a useful per-env diagnostic).

`benchmark.submit_entry(entry, raw)` writes three files - cheapest and
coarsest first, so a reader only pays for what they open:

- `summary.json` - `entry`'s provenance/hardware fields plus
  `benchmark.summary(raw)`. What a table listing every entry reads.
- `details.json` - `benchmark.details(raw)`. What expanding one
  entry's row reads next.
- `diagnostics.npz` - the full per-env training curves (resampled to
  a fixed number of points, so file size doesn't grow with training
  length), for whatever wants to plot a single entry's curves.

All three land in the directory of whichever script called
`submit_entry` - your entry's own folder, alongside `config.yml`.

## Submit your own entry

1. **Pick a preset** - `benchmarks/100K/` or `benchmarks/1M/`, or both.
2. **Create your entry's folder**: `benchmarks/<preset>/<your-entry>/`,
   e.g. `benchmarks/1M/my-algo/`.
3. **Write `run.py`**:

   ```python
   from navix.benchmarks import AlgorithmEntry, Navix1M, TrainingCurve  # or Navix100K
   from navix.environments.registry import make

   class MyEntry(AlgorithmEntry):
       def train(self, env_id, budget, rng):
           env = make(env_id)
           agent = MyAgent(hparams=MyHparams(budget=budget), env=env)
           _, logs = agent.train(rng)
           # build a TrainingCurve from your own logs, however you track them
           return TrainingCurve(
               episodic_returns=...,  # one point per update
               lengths=...,           # one point per update
           )

   entry = MyEntry(
       name=..., author=..., paper_url=...,
       navix_commit_url=..., algorithm_commit_url=...,
   )
   benchmark = Navix1M()
   raw = benchmark.run(entry)
   summary = benchmark.summary(raw)
   benchmark.submit_entry(entry, raw)  # writes summary.json/details.json/diagnostics.npz next to run.py
   ```

   It must run as `python run.py`, no arguments - read your own
   metadata from the sibling `config.yml` you add next, not from CLI
   flags. Per navix's "never vendor an external algorithm's code" rule
   ([#130](https://github.com/epignatelli/navix/issues/130)): your
   implementation stays in your own repo, and `train` is the only seam
   that has to touch navix. `cost_analysis` has a protocol-agnostic
   default (compiles `train` and reads its FLOPs/memory/compile-time)
   - override it only if that default doesn't fit your algorithm.

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
   doesn't say which repo it's from. `AlgorithmEntry` validates both,
   along with `train`'s return shape, the moment `run.py` constructs
   it - a malformed entry fails immediately, not partway through a
   real (possibly hours-long) training run. For a navix-shipped agent,
   `algorithm_commit_url` is the same commit as `navix_commit_url`.

5. **Add `requirements.txt`** - pinned dependencies for `run.py`
   itself (not navix's own dependencies in general, not your
   algorithm repo's).
6. **Optional `README.md`** - free-form notes about your entry.
7. **Run `python run.py`** - besides printing a summary, this writes
   `summary.json`/`details.json`/`diagnostics.npz` into your entry's
   own folder - commit them alongside `run.py`/`config.yml`/
   `requirements.txt`, so your results are there to review without
   anyone re-running training.
8. **Open a PR** adding your `benchmarks/<preset>/<your-entry>/`
   folder. No submission portal yet (tracked in #130).

See `navix/benchmarks/__init__.py`'s module docstring for the full
design, and [issue #130](https://github.com/epignatelli/navix/issues/130)
for the leaderboard proposal this implements.
