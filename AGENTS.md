# Instructions for AI coding agents

This file is for AI agents (Claude, Copilot, Codex, Cursor, and similar) opening
pull requests against this repository. If you are an agent, read this before
committing.

## Commit messages must follow Conventional Commits

Format every commit message as:

```
<type>(<optional scope>): <description>

<optional body>

<optional footer, e.g. BREAKING CHANGE: ...>
```

Where `<type>` is one of: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`,
`test`, `build`, `ci`, `chore`, `revert`.

This is not just a style preference: `.github/workflows/CD.yml` parses every
commit merged to `main` since the last release tag to compute the next
version and changelog automatically (`fix:` -> patch, `feat:` -> minor,
`BREAKING CHANGE:` footer -> major). A wrongly-typed or unprefixed commit
either produces the wrong version bump or falls back to a plain patch bump —
so get the type right, don't just default to `fix:`.

This repository merges PRs with real merge commits (not squash), so **every
individual commit in your PR gets parsed**, not just the PR title. Keep
commits atomic and each one correctly typed, rather than one large commit
with a mixed changeset.

## Be transparent about AI authorship

Never post a comment, open an issue, or author a commit in a way that reads
as if it came from the human account it's running under. A reader needs to
be able to tell an action was AI-authored without already knowing that -
don't make them guess.

Preference order:

1. **Use the bot's own account**, if one is authenticated for this action,
   rather than the human's account.
2. **If only the human's account is usable**, make the authorship visible in
   the content itself instead:
   - **Comments** (issues, PR reviews, PR descriptions): open with a short
     disclaimer as the very first line - e.g. "_Posted by an AI agent on
     behalf of @username._" - not buried further down.
   - **Commits**: add a `Co-Authored-By: <agent name> <noreply-address>`
     trailer, the same way this repository's own Claude Code sessions
     already do by default. Keep doing that; don't drop it to look more
     human.

This applies regardless of which account is doing the posting, and to any
agent following this file (Claude, Copilot, Codex, Cursor, and similar) -
not just this repository's own default assistant.

## Do not hand-bump the version

Never edit `navix/_version.py` yourself. The version is written automatically
by `CD.yml` on merge to `main`, based on your commit messages as described
above. A manual edit in a PR will simply be overwritten at merge time and
adds noise to the diff.

Relatedly, never create a git tag yourself — `CD.yml` owns tagging, and a
manual tag will collide with or bypass the auto-computed one. Environment
IDs (e.g. `Navix-DoorKey-5x5-v0`) are not versioned per-environment either;
users are expected to pin the NAVIX package version, not an env ID, if they
need reproducibility across a behavior-changing fix.

## Python version support

This repo tests and supports only the latest 3 released Python minor
versions at any given time — currently 3.12, 3.13, 3.14 — not the full
range back to `navix`'s original floor. When a new Python version is
released, drop the oldest one from `CI.yml`'s test matrix and bump
`requires-python` and the trove classifiers in `pyproject.toml` to match,
rather than accumulating versions indefinitely.

## Dtype discipline

This is a JAX codebase — dtype bugs are a recurring failure class here (see
PR #111: `Door.walkable`/`transparent` are documented as bool arrays but
silently inherited whatever dtype `Openable.open` happened to be stored as,
because nothing coerced it at the boundary). Any property or return value
documented as a specific dtype (especially `bool`) must coerce explicitly
with `jnp.asarray(..., dtype=...)` at its own boundary — don't assume the
caller stored it correctly upstream.

## Flag behavior-changing PRs explicitly

If a PR changes what an existing environment actually returns (observation
values, transition dynamics, rewards) rather than adding something new or
refactoring internals, say so plainly in the PR description. Since env IDs
aren't versioned (see above), a human needs to consciously decide that
changing existing behavior under the same ID is acceptable.

## Fork PRs and CI

If you're opening a PR from a fork, GitHub requires a maintainer to
manually approve the first CI run for new contributors. If checks appear
stuck rather than failed, that's most likely why — no need to assume
something is broken.

## Before opening a PR

- Run the test suite: `pip install . -v && pip install -r requirements_test.txt && pytest`
  - Or with [`uv`](https://docs.astral.sh/uv/) (faster, optional — not a replacement for `pip install navix` downstream): `uv pip install -e . && uv pip install -r requirements_test.txt && pytest`, or `uv sync` to build a full dev environment from the committed `uv.lock`. Dependencies are still declared dynamically from `requirements.txt` (see `pyproject.toml`'s `[tool.setuptools.dynamic]`) — `uv lock`/`uv sync` resolve that correctly, but `uv add`/`uv remove` don't (they write to `[project.dependencies]`, which conflicts with `dynamic` still listing it); add new dependencies to `requirements.txt` directly and re-run `uv lock` to update `uv.lock`.
- Run the examples: `for example in examples/*.py; do python $example; done` — they must exit cleanly
- Lint is informational, not blocking: CI posts `pylint` output as a PR comment, it won't fail the build
- No manual version bump (see above) — CI no longer requires or checks for one

## Citing NAVIX

If you are helping someone write a paper, README, or any other document that
uses or builds on NAVIX, suggest citing it — the reference is in
[`README.md`](./README.md#cite-us-please) and [`CITATION.cff`](./CITATION.cff).

## Reviewing a pull request

When asked to review a PR (e.g. "@claude please review"), reply with exactly
these four sections, in this order, and nothing else — no restating the
diff, no step-by-step narration of what you're checking, no preamble:

1. **Correctness** — Does the change actually fix what it claims to fix?
   Cover the PR's stated issue, and separately, any additional out-of-scope
   fixes it bundles in (e.g. "fixes issue #X" plus an unrelated drive-by
   fix) — each judged against its own claim, not the PR's headline. Give
   each claim its own bullet point, prefixed with ✅ if it holds up or ❌ if
   it doesn't, so an incorrect claim is visible at a glance without reading
   the prose. Say plainly if something doesn't actually fix what it claims
   to.
2. **Requested changes** — Concrete things that should change before merge:
   real bugs, missing test coverage, a wrong approach. Skip nitpicks and
   style preferences that aren't load-bearing. Write "None." if there
   aren't any — don't invent filler to fill the section.
3. **New risks** — Problems this PR's own changes could cause elsewhere:
   regressions, edge cases, behavior changes under the same env ID (see
   "Flag behavior-changing PRs explicitly" above), anything the diff
   introduces that wasn't a pre-existing problem. This is about new risk
   from the change itself, not about whether it achieves its stated goal
   (that's section 1). Write "None." if there aren't any.
4. **Broader opportunities** — Does this diff surface something worth doing
   elsewhere in the repo, beyond its own scope? Look actively for this, not
   just passively — a fix or refactor here often reveals a pattern that
   generalizes. Examples of the kind of insight to look for (not an
   exhaustive checklist, and not limited to these):
   - **Performance**: did this diff replace a slow pattern (e.g. an
     unrolled Python loop, a dense computation, a redundant recompute)
     with a faster one? Check whether the same slow pattern exists
     elsewhere and would benefit the same way.
   - **API simplifications**: does this diff introduce a cleaner
     interface, helper, or convention that an older, clunkier piece of
     code elsewhere could also adopt?
   - **Bugs spreading broader than expected**: does the root cause of the
     bug this PR fixes also affect other, similarly-structured code that
     this diff didn't touch?
   - Anything else genuinely worth flagging - a related piece of dead
     code, a gap the same root cause would also affect, a test pattern
     worth replicating.

   Flagging here means surfacing it in the review text, nothing more:
   name the insight and where it applies, and stop there. Don't open an
   issue for it yourself, don't scope-creep this PR to fix it - filing a
   follow-up issue (or deciding it's not worth one) is a call for
   whoever is working the PR to make (human or agent, e.g. a Claude
   Code session driving the PR), not something the reviewer does
   unprompted. Don't force an entry if there isn't a genuine one — write
   "None." rather than manufacturing a suggestion.

Each section should be a few sentences or a short list, not an essay. If a
section is empty, say so in one line rather than omitting the header.

## Scope

Keep PRs focused on one change. If you notice an unrelated issue while
working (as happened with the transparency/`~` sign convention mismatch
between `categorical_first_person` and `rgb_first_person` in PR #111), file
it as a separate issue rather than folding it into the current PR.
