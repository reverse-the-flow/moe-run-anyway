Implemented the repo hygiene and model-test direction artifacts.

Added:
- [.gitignore](/home/codexlab/moe-run-anyway-project/.gitignore)
- [model_target_registry.json](/home/codexlab/moe-run-anyway-project/memory-moe-mvp/data/model_target_registry.json)
- [model_target_registry.py](/home/codexlab/moe-run-anyway-project/memory-moe-mvp/model_target_registry.py)
- [model-target-test-plan.md](/home/codexlab/moe-run-anyway-project/memory-moe-mvp/docs/model-target-test-plan.md)
- [test_model_target_registry.py](/home/codexlab/moe-run-anyway-project/memory-moe-mvp/tests/test_model_target_registry.py)
- [MODEL_TESTS_STATUS.md](/home/codexlab/moe-run-anyway-project/MODEL_TESTS_STATUS.md)

Validation run:
- `python3 memory-moe-mvp/model_target_registry.py` passed, 5 targets covered.
- `python3 -m unittest discover -s memory-moe-mvp/tests` passed: 21 tests.
- `python3 -m py_compile ...` passed for touched/new probe-related Python.

Git state:
- Repo is initialized on `main`.
- No commits exist yet.
- I did not create the baseline commit because `git config user.name` and `git config user.email` are unset.
- Ignored local logs/caches/run dirs are now separated from source artifacts.