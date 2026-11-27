# Publish the tested Docker image to GHCR

Add `scripts/train_release.py` and `.github/workflows/publish-image.yml` to the project.
Keep the existing Dockerfile and `ci.yml`. The publishing workflow is separate because
notebook 2 rewrites `ci.yml` during its demonstration.

The sequence is: push to main -> existing CI passes -> full-data training -> quality
gate -> image build -> HTTP smoke check -> GHCR push. Pull requests do not publish.
The workflow checks out the exact commit that passed CI. It compares the full-data
candidate with an always-approve bootstrap baseline, not a historical production champion.
It does not evaluate the final test period or change the local MLflow registry.

## Enable publishing

From the project folder, on the `main` branch:

```text
git add scripts/train_release.py .github/workflows/publish-image.yml docs/publish_image.md
git commit -m "Publish tested full-model Docker images to GHCR"
git push origin main
```

Open the repository's Actions tab. After `Fraud model and API checks` succeeds,
`Publish Docker image` runs automatically. Both new files must be included in the push.
The workflow must exist on the default branch. No manually created token or secret
is needed for workflow publishing: it uses `GITHUB_TOKEN` with `packages: write`.

The image name is `ghcr.io/manishpingale7/mldemo:sha-FULL_COMMIT_SHA`.
`FULL_COMMIT_SHA` means the actual 40-character commit ID, not literal text.
The publish job summary prints complete pull and run commands to copy.
There is deliberately no `latest` tag: select the specific commit you want to demonstrate.

## Run the published image on Windows

Start Docker Desktop with Linux containers. Stop any existing dashboard on port 8765.
For anonymous classroom downloads, after the first publication open the GitHub package
`mldemo`, go to Package settings and change its visibility to Public. The repository
being public does not automatically make a newly published package public.

Then copy the exact image name from the workflow summary into these commands:

```bat
docker pull ghcr.io/manishpingale7/mldemo:sha-FULL_COMMIT_SHA
docker run --rm --name fraud-demo -p 127.0.0.1:8765:8765 ghcr.io/manishpingale7/mldemo:sha-FULL_COMMIT_SHA
```

Open http://localhost:8765. This runs the application on your laptop; GHCR stores
the image and does not host the running API. After pulling, the image runs offline.
Stop it with `docker stop fraud-demo` in another terminal.

If you keep the package private, first run `docker login ghcr.io -u ManishPingale7`
and use a personal access token (classic) with `read:packages` as the password prompt
input. Do not put the token in source files. No GitHub account password is used here.

To roll back, stop the current container and run the image tagged with an earlier
successful commit. No retraining is required to run or restore a published image.

## Verification so far

Full-data training, validation gate, isolated release creation, API smoke requests,
and Docker staging passed locally. Selected Random Forest threshold: 0.60;
validation precision: 0.9630, recall: 0.9559, PR-AUC: 0.9500.
Ruff and workflow YAML, referenced script paths and Bash syntax checks passed.
GHCR authentication, publishing and pulling the new image have not been executed.

GitHub documentation, accessed 2026:
[Container registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
and [workflow_run trigger](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflow_run).
