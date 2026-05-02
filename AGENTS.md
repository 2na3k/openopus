# Workflow

## Branching
- `main` is the stable branch. Never push directly to it.
- Create feature branches from `main`: `feature/<description>`
- After finishing work, open a PR from `feature/<description>` → `main`

## PR & Feedback
- All changes go through a PR.
- Address PR feedback by pushing new commits to the feature branch.
- Never force push.

## Dev Loop
1. Create a feature branch from `main`.
2. Write unit tests for the feature.
3. Run tests: `uv run pytest`
4. Lint and format: `uv run ruff check src/ && uv run ruff format src/`
5. Type-check: `uv run mypy src/`
6. Commit to the feature branch.
7. Push and open a PR.
