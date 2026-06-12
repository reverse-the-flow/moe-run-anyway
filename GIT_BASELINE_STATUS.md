# Git Baseline Status

Date: 2026-06-12

Workspace: `/home/codexlab/moe-run-anyway-project`

The workspace was initialized as a git repository on branch `main`.

The Codex model-tests pass could not create the first commit because no git
identity was configured in the repository. After that pass finished, a
repository-local identity was added:

- `user.name = Codex on codexlab`
- `user.email = codexlab@gx10-81e8.local`

The first baseline commit was then created:

- `c6483c6 Initialize MoE run-anyway project`

That commit includes the copied source/context snapshot plus the new model-test
registry, validator, plan, tests, and status files. Generated logs, caches, and
run directories remain ignored by `.gitignore`.
