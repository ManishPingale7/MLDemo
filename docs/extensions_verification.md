# Docker and DVC verification

Verified on 26 November 2026. The model and data-generating rules were unchanged.

| Check | Observed result |
|---|---|
| Local Docker image | Built with Python 3.11.14; Docker Engine 29.9.2 and Compose 2.40.3 |
| Running container | Healthy; six real HTTP predictions passed; static dashboard/docs, label rejection and reload passed |
| Prediction consistency | All six container scores matched the local fitted pipeline within 1e-12 |
| Offline serving | Repeated HTTP checks inside a container started with `--network none`; passed |
| Dependency consistency | `pip check` passed in the main environment, DVC environment and serving image |
| DVC 3.67.1 | Saved two versions, deleted the sandbox cache, pulled the first version from local storage and verified its exact SHA-256 |
| Full dataset tracking | DVC discovered `data/transactions.csv.dvc`; cache and local remote were in sync |
| Existing project checks | 40 pytest tests passed; Ruff passed; selected model gate passed and baseline gate failed as intended |
| Updated notebook 2 | Fresh-kernel execution passed in 55.72 seconds; saved with outputs |

The unchanged notebook 1 retains its previously executed outputs and 10.93-second
measurement. The README's full-run timing describes the earlier complete `run_all.py`
execution; 55.72 seconds above is the separate updated notebook 2 rerun.

[Hosted run 45836850937](https://github.com/ManishPingale7/MLDemo/actions/runs/45836850937)
passed both Windows and Linux jobs at commit `ecf531d`, including full dataset pointer
discovery and DVC installation/restoration on both
systems and a standard Dockerfile build plus real HTTP checks on Linux.

The local managed environment required a temporary build-secret certificate mount for
pip's HTTPS requests. The normal committed Dockerfile was verified by hosted Linux CI;
it contains no environment-specific certificate paths. Container files are readable
by the unprivileged runtime user, including when source files have restrictive host permissions.

Reports in the distributed archive: `models/container_smoke.json`,
`models/container_offline.json`, `models/dvc_demo.json` and `models/dvc_demo_output.txt`.
The archived container report refers to release-003; a subsequent notebook run created
release-004 with the same model and threshold. Staging always picks the current active release.

Not tested: Docker Desktop/WSL setup on a physical Windows 11 laptop, the Windows
double-click launcher's interactive behaviour, or visual dashboard interactions in a
browser. Windows CI verifies Python and DVC behaviour, not Docker Desktop installation.
There is no cloud deployment, image publishing, S3 storage or automatic model promotion.
