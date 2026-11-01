# Fair Income API

A machine learning model served as a production-style web API, with a **built-in fairness audit**.

The model predicts whether a person's income is above $50K from census data (the UCI Adult dataset, a standard fairness benchmark). Sex and race are **not** used as model inputs, but every trained model is audited across both groups, because leaving a sensitive attribute out does not make a model fair.

**Live demo:** [add your deployed URL here]

## What this project shows

- **ML pipeline:** scikit-learn preprocessing + logistic regression, trained and evaluated on held-out data
- **Fairness audit, written from scratch:** per-group true/false positive rates, demographic parity difference, equalized odds difference, and bootstrap confidence intervals
- **Deployment:** FastAPI service, input validation with Pydantic, Docker image, free cloud hosting
- **Monitoring:** request counts, latency and live positive-prediction rate (a simple drift signal)
- **Engineering practice:** unit and API tests (pytest), linting (ruff), and CI on every push (GitHub Actions) that builds the Docker image and smoke-tests the running container

## Results

Fill this in after running `python -m training.train` on the real data:

| Metric | Value |
|---|---|
| Accuracy | [ ] |
| ROC AUC | [ ] |
| Equalized odds difference (sex) | [ ] (95% CI [ ] to [ ]) |
| Demographic parity difference (sex) | [ ] |
| Equalized odds difference (race) | [ ] (95% CI [ ] to [ ]) |

The full report is served at `/fairness`.

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service and model status |
| POST | `/predict` | Income prediction for one applicant |
| GET | `/fairness` | Fairness audit from training time |
| GET | `/metrics` | Traffic, latency and positive-prediction rate |
| GET | `/docs` | Interactive API documentation (Swagger) |

Example request:

```bash
curl -X POST http://localhost:7860/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 38, "education_num": 13, "hours_per_week": 40,
       "capital_gain": 0, "capital_loss": 0,
       "workclass": "Private", "marital_status": "Married-civ-spouse",
       "occupation": "Prof-specialty"}'
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python -m training.train              # downloads the Adult dataset and trains
uvicorn app.main:app --port 7860      # then open http://localhost:7860/docs

python -m pytest                      # tests (use synthetic data, no download needed)
```

## Run with Docker

```bash
docker build -t fair-income-api .
docker run -p 7860:7860 fair-income-api
```

## Deploy (free)

**Option A: Hugging Face Spaces**

1. Create a new Space at huggingface.co/new-space and choose **Docker** as the SDK.
2. Push this repository to the Space's git remote.
3. In the Space's `README.md`, keep this header at the very top:
   ```yaml
   ---
   title: Fair Income API
   sdk: docker
   app_port: 7860
   ---
   ```
4. The Space builds the Docker image and serves the API. Your URL looks like `https://<username>-fair-income-api.hf.space/docs`.

**Option B: Render**

1. Create a new **Web Service** on render.com and connect this GitHub repository.
2. Choose **Docker** as the runtime. Render sets the `PORT` variable automatically.

## Project structure

```
app/
  main.py        FastAPI app: endpoints, model loading, monitoring
  schemas.py     Pydantic request/response validation
training/
  train.py       Data loading, training, evaluation, fairness audit
  fairness.py    Fairness metrics and bootstrap confidence intervals
tests/           Unit tests for metrics, API tests
Dockerfile       Self-contained image (model is trained at build time)
.github/workflows/ci.yml   Lint, tests, Docker build and smoke test
```

## Next steps

- Bias mitigation: group-specific thresholds or equalized-odds post-processing, and compare the accuracy/fairness trade-off
- Log predictions and alert when the live positive rate drifts from the training rate
- Model registry and versioned deployments

## Data

UCI Adult (Census Income) dataset, loaded from OpenML. Used for research and education only; this model is a demonstration and must not be used for real decisions about people.
