# Deployment Module Documentation

This folder contains the deployment and model-serving layer for the project. It has two related responsibilities:

1. Keep track of model/config versions through a simple experiment registry.
2. Serve predictions through a FastAPI application and package that app into Docker, Heroku, or EC2.

The deployment module is split across two parts:

- [vcs.py](vcs.py) manages experiment snapshots, commits, deployment state, and restoring older versions.
- [accelera_deployment/](accelera_deployment) contains the FastAPI service, the model loading logic, and the deployment helper script.

## High-level flow

1. A trained model and supporting artifacts are stored in [models/](models).
2. [config.json](config.json) maps logical artifact names to those files.
3. [accelera_deployment/modelservice.py](accelera_deployment/modelservice.py) loads the artifacts and exposes prediction helpers.
4. [accelera_deployment/server.py](accelera_deployment/server.py) exposes the HTTP API.
5. [accelera_deployment/deployment.py](accelera_deployment/deployment.py) builds a Docker image and can deploy it to Heroku or EC2.

## What each file does

### [accelera_deployment/server.py](accelera_deployment/server.py)

This is the FastAPI app.

At startup it uses a FastAPI lifespan handler to:

- call `service.load()` so the model artifacts are loaded once,
- print the local and internal base URLs,
- print the documentation URL,
- enumerate the available routes.

It defines four prediction endpoints:

- `POST /predict` for the default classifier (`knn_model`).
- `POST /predict/classifier/{model_name}` for named classifier models.
- `POST /predict/regressor/{model_name}` for named regression models.
- `POST /predict/csv` as a legacy compatibility route for file uploads only.

The first three routes all accept input on the same path in any of these formats:

- JSON body (`application/json`): `{"input": [[...], [...]]}`.
- Multipart CSV upload (`multipart/form-data` with `file=<data.csv>`).
- Multipart NPY upload (`multipart/form-data` with `file=<data.npy>`).

Each endpoint validates input and returns a JSON response or an HTTP error.

### [accelera_deployment/modelservice.py](accelera_deployment/modelservice.py)

This file contains the prediction engine.

What happens here:

- `ModelService.load()` reads [config.json](config.json).
- It loads each pickled artifact from disk into memory.
- It stores the preprocessing pipeline as `preprocessing_pipeline.pkl`.
- It stores the target inverse scaler as `target_scaling.pkl`.

Input preparation works like this:

- The input is converted to a NumPy array.
- A 1D sample is reshaped into a 2D matrix.
- The preprocessing pipeline is applied if it exists.
- The final matrix is converted to `float32`.

Classifier behavior:

- `knn_model` uses the loaded KNN classifier and returns `model.predict(X)`.
- `ordinal_logistic` and `ordinal_logistic_threshold_weights` use learned threshold weights, compute class probabilities, and return the argmax class.

Regressor behavior:

- `linear_regression` and `linear_regression_weights` use a weight vector with a bias term.
- If a target scaler is available, predictions are converted back to the original scale using `inverse_transform()`.

### [accelera_deployment/deployment.py](accelera_deployment/deployment.py)

This script automates packaging and deployment.

It provides these command groups:

- `prepare` writes the deployment requirements file and regenerates the Dockerfile.
- `build` runs `docker build`.
- `run-local` starts the Docker container locally.
- `local` runs prepare, build, and local container start in sequence.
- Heroku commands handle login, app creation, pushing the container, releasing it, and opening the app.

Important detail:

- The script changes the working directory to the project root before doing anything.
- The generated Dockerfile copies the deployment service files, `config.json`, and any model files referenced in the config.

### [vcs.py](vcs.py)

This file behaves like a tiny experiment/version control system for model artifacts.

It stores metadata under `experiments/` and uses [experiments/experiments.json](experiments/experiments.json) as the index.

Commands:

- `init` creates the experiments directory and initializes the index.
- `commit -m "message"` snapshots the current `config.json` and `models/` directory into a new experiment folder.
- `log` prints the commit history.
- `show <hash>` prints commit details and lists the stored files.
- `deploy <hash>` restores that experiment into the active `config.json` and `models/` paths.
- `status` prints the current HEAD and deployed commit.

### [model.py](model.py)

This is a separate training/demo script.

It is a standalone demo training script for a separate tabular example and is not the source of the deployment artifacts used by the service.

It is not the serving API itself, but it explains where the model artifacts in the deployment flow can come from.

## Available routes

All routes are `POST` routes.

| Route | Purpose | Input |
| --- | --- | --- |
| `/predict` | Default classifier endpoint using `knn_model` | JSON body or multipart CSV/NPY file |
| `/predict/classifier/{model_name}` | Run a named classifier | JSON body or multipart CSV/NPY file |
| `/predict/regressor/{model_name}` | Run a named regressor | JSON body or multipart CSV/NPY file |
| `/predict/csv` | Legacy file-only endpoint for `knn_model` | Multipart CSV/NPY file |

### `POST /predict`

JSON request body:

```json
{
  "input": [[5.1, 3.5, 1.4, 0.2]]
}
```

Multipart file request examples:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@sample_input.csv"
```

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@features.npy"
```

Behavior:

- If `input` is missing, the server returns `400`.
- The request is routed to `service.predict_classifier("knn_model", input)`.
- The response contains a `predictions` array.

Example response:

```json
{
  "predictions": [0]
}
```

If your deployment is for housing data, the JSON feature order, CSV headers, and NPY tensor shape should all match the housing training schema.

### `POST /predict/classifier/{model_name}`

Supported classifier names in the current code:

- `knn_model`
- `ordinal_logistic`
- `ordinal_logistic_threshold_weights`

Example request:

```json
{
  "input": [[5.1, 3.5, 1.4, 0.2]]
}
```

Behavior:

- Missing input returns `400`.
- Unsupported content types return `415`.
- Unknown or unavailable models return `404`.
- Internal prediction errors return `500`.

Example response:

```json
{
  "model": "knn_model",
  "task": "classifier",
  "predictions": [0]
}
```

### `POST /predict/regressor/{model_name}`

Supported regressor names in the current code:

- `linear_regression`
- `linear_regression_weights`

Example request:

```json
{
  "input": [[1.2, 3.4, 5.6]]
}
```

Behavior:

- Missing input returns `400`.
- Unsupported content types return `415`.
- Unknown or unavailable models return `404`.
- Internal prediction errors return `500`.

Example response:

```json
{
  "model": "linear_regression",
  "task": "regressor",
  "predictions": [42.7]
}
```

## Files used at runtime

The service expects the following artifacts to exist according to [config.json](config.json):

- `./models/knn_model.pkl`
- `./models/linear_regression_weights.pkl`
- `./models/ordinal_logistic_threshold_weights.pkl`
- `./models/preprocessing_pipeline.pkl`
- `./models/target_scaling.pkl`
- `./models/total_remaining_features.pkl`

The Docker build process copies these configured files into the container and `prepare` fails fast if any configured artifact is missing. The ready-to-upload sample file in this repo is just a placeholder; replace it with your housing-data CSV if that is the schema you are serving.

## Local run

Typical local workflow from the `deployment_module` directory:

```bash
python accelera_deployment/deployment.py prepare
python accelera_deployment/deployment.py build
python accelera_deployment/deployment.py run-local
```

Or do it in one step:

```bash
python accelera_deployment/deployment.py local
```

The container listens on port `8000` by default, or on the value of the `PORT` environment variable.

## Heroku deployment

The deployment script supports container-based Heroku deployment:

```bash
python accelera_deployment/deployment.py heroku-deploy --app accelera1 --create
```

Useful subcommands:

```bash
python accelera_deployment/deployment.py heroku-login
python accelera_deployment/deployment.py heroku-create --app accelera1
python accelera_deployment/deployment.py heroku-container-login
python accelera_deployment/deployment.py heroku-push --app accelera1
python accelera_deployment/deployment.py heroku-release --app accelera1
python accelera_deployment/deployment.py heroku-open --app accelera1
```

## EC2 deployment

The EC2 flow is built for a single Ubuntu or Amazon Linux instance where you can SSH in with a key pair and run Docker.

Required inputs:

- EC2 public IP or DNS name.
- SSH username, usually `ec2-user` on Amazon Linux or `ubuntu` on Ubuntu images.
- SSH private key file path, such as `~/.ssh/key.pem`.
- An open security-group rule for the port you expose, usually `8000` or `80`.
- Docker installed on the instance, or the `--install-docker` flag if the host has `sudo` and a supported package manager.

Example:

```bash
python accelera_deployment/deployment.py ec2-deploy \
  --host 1.2.3.4 \
  --user ec2-user \
  --key ~/.ssh/key.pem \
  --port 8000 \
  --install-docker
```

What the command does:

1. Regenerates the deployment Dockerfile and requirements file locally.
2. Syncs the `deployment_module/` folder to the EC2 instance with `rsync`.
3. Optionally installs and starts Docker on the EC2 host.
4. Builds the container image on EC2.
5. Stops any previous container with the same name.
6. Starts the new container and exposes the chosen port.

Helpful follow-up commands:

```bash
python accelera_deployment/deployment.py ec2-stop --host 1.2.3.4 --user ec2-user --key ~/.ssh/key.pem
python accelera_deployment/deployment.py ec2-logs --host 1.2.3.4 --user ec2-user --key ~/.ssh/key.pem
```

## Experiment/version workflow

If you use `vcs.py`, the intended flow is:

1. Run `python vcs.py init` once.
2. Save or update your active models in `deployment_module/models/`.
3. Run `python vcs.py commit -m "message"` to snapshot the current state.
4. Use `python vcs.py log` and `python vcs.py show <hash>` to inspect history.
5. Use `python vcs.py deploy <hash>` to restore a specific version.
6. Rebuild and run the deployment after restoring the desired version.

## Notes and limitations

- `server.py` always returns `knn_model` for the plain `/predict` and `/predict/csv` routes.
- `modelservice.py` will raise a `ValueError` if a requested model name is not present in the loaded artifacts.
- The API assumes the input arrays have the same shape and feature order as the preprocessing pipeline expects.
- The deployment script writes a Dockerfile into [Dockerfile](Dockerfile) and a deployment requirements file into [accelera_deployment/requirements.txt](accelera_deployment/requirements.txt).
- The EC2 deployment command assumes the host is reachable over SSH and that inbound traffic is allowed on the chosen port.

## Quick reference

- API docs: `http://localhost:8000/docs`
- Default base URL: `http://localhost:8000`
- Docker container port: `8000`
