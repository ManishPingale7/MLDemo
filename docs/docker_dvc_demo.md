# Docker and optional DVC demo

Docker packages the fitted model and prediction service into a runnable image.
DVC records which exact dataset file belongs to a Git commit. Neither needs S3, ECS or a cloud account here.

## 1. Docker: run the existing model (3 minutes live)

Before presenting, install Docker Desktop on Windows 11 with its WSL 2 backend and
Linux containers. Start Docker Desktop. Install this project's Python requirements
using the existing setup instructions. The Python `docker` package does not install Docker Desktop.

Open Command Prompt in the project folder. The distributed archive includes a trained
release; for a fresh Git clone, first run `python scripts/run_all.py` in the activated
project environment to generate one. Stop the native dashboard with Ctrl+C if it uses port 8765.

```bat
.venv\Scripts\python.exe scripts\prepare_container.py
docker compose up --build -d --wait --wait-timeout 60
docker compose ps
.venv\Scripts\python.exe scripts\smoke_container.py
```

On Linux, replace `.venv\Scripts\python.exe` with `.venv/bin/python` and use `/` in script paths.

Expected: staging prints the selected model and threshold, Compose shows a healthy
container, and the smoke script prints `PASS container`. Open **http://127.0.0.1:8765**,
start balanced traffic and inspect a prediction. The API reference is at `/docs`.

Say: “The image contains our fitted model, Python dependencies, FastAPI and dashboard.
Starting a container runs predictions; it does not train another model.”

Pre-build before class: the first build downloads Python, a system library and pinned
Python packages. Afterwards, `docker compose up -d --no-build` works without internet.
The serving-only dependencies match the notebook versions. MLflow and Jupyter continue
to run in the normal laptop environment, outside this container.

Stop with `docker compose down`. After accepting a new model locally, run staging and
`docker compose up --build -d --wait` again. **Reload model** reads the release inside
the current container; it does not fetch a newer file from your laptop. There is no
host model volume. To serve a previous local release, run `python scripts/deploy_model.py
--rollback`, then stage and rebuild. No model training happens during image building.

## 2. DVC: change a dataset and restore it (3 minutes live)

DVC is optional and has its own pinned environment so it does not change the notebook
dependencies. Install Git for Windows. Run **START-DVC.cmd** once with internet access,
after generating the full dataset. For subsequent offline demos:

```bat
.venv-dvc\Scripts\python.exe scripts\dvc_demo.py
```

Linux setup and run:

```sh
python3.11 -m venv .venv-dvc
.venv-dvc/bin/python -m pip install -r requirements-dvc.txt
.venv-dvc/bin/python scripts/dvc_demo.py
```

The script prints real Git and DVC commands. In a disposable folder under `.runtime/` it:

1. Copies 200 rows from the generated data and saves version one using `dvc add`, a Git commit and `dvc push`.
2. Corrects one amount by one rupee and saves version two.
3. Shows `dvc diff`: the dataset file changed. DVC compares files, not individual CSV rows.
4. Restores version one's Git pointer, deletes the sandbox CSV and cache, then runs `dvc pull` from local storage.
5. Verifies the restored file's SHA-256 matches version one and that the original dataset is untouched.

Expected final line: `PASS DVC: two versions saved; version one restored byte-for-byte
from local storage; original data unchanged.` Results are saved in `models/dvc_demo.json`.

Say: “Git stores the small `.dvc` pointer; DVC stores the dataset contents.
Together they let us recover the exact data used for an experiment.”

The actual project also includes `.dvc/config` and `data/transactions.csv.dvc`, tracking
the full table. Its remote is the ignored local `.dvc-storage/` folder. Populate that
storage on a new laptop with these commands **after generating data**:

```bat
.venv-dvc\Scripts\python.exe -m dvc add data/transactions.csv
.venv-dvc\Scripts\python.exe -m dvc push
.venv-dvc\Scripts\python.exe -m dvc status
```

After intentionally changing the generator/config/data, commit the updated `.dvc`
pointer with those source changes before recording a new MLflow experiment. Its Git
SHA then identifies the committed pointer. `dvc pull` requires data already saved to
this local storage; a Git clone alone does not download the CSV. Storage on the same
disk is a teaching remote, not an off-machine backup. Sandbox copies can be deleted
after the demo; only the original full table is used to train the project models.

## 3. Where CI fits (1 minute live)

GitHub Actions runs the original model/API checks on Linux and Windows. It additionally
builds and starts Docker on Linux, sends real HTTP predictions, and checks DVC restoration
on both operating systems. JSON results appear in the uploaded artifacts.

The hosted image uses the **tiny CI model**; the laptop image packages the **full model**.
There is no image registry push or automatic production deployment. The notebook's
`run_ci_local.py` runs core checks; run the Docker and DVC commands above separately,
keeping both notebooks short.

Fallback: show saved JSON results if Docker Desktop is unavailable, and run
`python scripts/run_demo.py` for the same API without Docker. If DVC is unavailable,
show the `.dvc` pointer and saved restoration result. State which live command failed.
