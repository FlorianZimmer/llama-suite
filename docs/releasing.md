# Releasing llama-suite

This repo now has a changelog and local annotated tags, but the tags and GitHub releases still need to be published.

## Recommended Backfill

These milestones map cleanly onto the existing history:

1. `v0.1.0` on commit `d9b7018`
   Foundation watcher/config release after the early cleanup and Python rewrite.
2. `v0.2.0` on commit `1076950`
   First Web UI milestone.
3. `v0.3.0` on commit `7701d45`
   Operations-console milestone before the README and CI refresh.

## Commands

To publish the existing local tags:

```bash
git push origin v0.1.0 v0.2.0 v0.3.0
```

After the current README and CI work is committed, cut the next release from that merge commit and continue the changelog from `Unreleased`.

## GitHub Release Notes

Use the matching sections from [CHANGELOG.md](../CHANGELOG.md) as the release body.

- `v0.1.0`: watcher/config foundation
- `v0.2.0`: first Web UI release
- `v0.3.0`: operations console milestone

## Going Forward

- Keep one changelog section per release.
- Tag from merge commits on `main` only.
- Prefer short, concrete release notes over auto-generated commit dumps.
