# Contributing

The repository is [uzuz3737/ml-ops](https://github.com/uzuz3737/ml-ops). Each member works in their own clone and commits under their own identity. Ownership is recorded in [Team work](docs/TEAM_WORK.md), [P0 handoff](docs/P0_HANDOFF.md) and [CODEOWNERS](.github/CODEOWNERS).

| Member | Branch suggestion | Own implementation |
| --- | --- | --- |
| @uzuz3737 (P0, you) | `codex/p0-integration` | Integration, Airflow, runtime contracts, deployment/rollback, CI/CD, releases |
| @OuanEng (P1) | `codex/p1-data-validation` | UCI data, TFDV, shared features, feature drift |
| @pairot230 (P2) | `codex/p2-model-tracking` | Models, MLflow/Registry, candidate evaluation, labeled quality |
| @thanachaithongbai-hue (P3) | `codex/p3-serving` | FastAPI, exact-version model watcher/ACK, performance, dashboards |

Before starting, fetch the shared base, handle your local uncommitted changes and create your branch. Do not overwrite another person's files. Review the callable-stage contracts before implementing adapters; P0 owns shared integration files and each component lead supplies its own stage implementation and tests.

```powershell
git status
git switch main
git pull --ff-only
git switch -c codex/p1-data-validation
```

Use the [setup commands](docs/GETTING_STARTED.md) and run Ruff, pytest and the documentation checker before your PR. Tests must verify behavior, failure cases and the consumer handoff. Update the relevant contracts, setup instructions and evidence. A passing framework fixture is not a successful full lifecycle.

Open a PR using the repository template and request the reviewer named in [Team work](docs/TEAM_WORK.md). Keep the description concrete: what changed, why, actual tests, and remaining gaps. P0 integrates after review and verifies affected boundaries. GitHub review enforcement and collaborator access must be configured by a repository administrator; CODEOWNERS itself does not grant access or enable branch protection.

Keep `.env`, datasets, model artifacts, generated logs and caches out of Git. Small synthetic test fixtures are appropriate; record source/version/checksums for generated real outputs. Disclose AI-assisted work and be able to explain every submitted line.
