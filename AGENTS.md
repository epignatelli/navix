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

## Do not hand-bump the version

Never edit `navix/_version.py` yourself. The version is written automatically
by `CD.yml` on merge to `main`, based on your commit messages as described
above. A manual edit in a PR will simply be overwritten at merge time and
adds noise to the diff.

## Before opening a PR

- Run the test suite: `pip install . -v && pip install -r requirements_test.txt && pytest`
- Run the examples: `for example in examples/*.py; do python $example; done` — they must exit cleanly
- Lint is informational, not blocking: CI posts `pylint` output as a PR comment, it won't fail the build
- No manual version bump (see above) — CI no longer requires or checks for one

## Scope

Keep PRs focused on one change. If you notice an unrelated issue while
working (as happened with the transparency/`~` sign convention mismatch
between `categorical_first_person` and `rgb_first_person` in PR #111), file
it as a separate issue rather than folding it into the current PR.
