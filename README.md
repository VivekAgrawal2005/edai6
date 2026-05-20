# Email Intelligence API System

Backend-only FastAPI service for email intelligence using traditional NLP, scikit-learn, XGBoost, NLTK, SQLite, and deterministic templates. No LLMs or generative AI APIs are used.

## What it does

- Detects spam vs not spam
- Detects whether a reply is needed
- Predicts intent/category
- Generates template-based reply drafts
- Stores emails, predictions, and processing logs in SQLite

## Tech Stack

- Python 3.10+
- FastAPI
- scikit-learn
- XGBoost
- NLTK
- SQLite
- joblib

## Project Layout

- `app/` - FastAPI app, preprocessing, services, database helpers, schemas
- `ml_models/` - training scripts and saved artifacts
- `dataset/` - Enron corpus file `emails.csv`
- `templates/` - reply templates in JSON
- `tests/` - pytest suite

## Setup

1. Create and activate your Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Optional environment variables in `.env`:

```env
DATABASE_URL=sqlite:///./email_intelligence.db
DATASET_PATH=dataset/emails.csv
TEMPLATES_PATH=templates/reply_templates.json
SPAM_MODEL_PATH=ml_models/spam/spam_model.pkl
SPAM_VECTORIZER_PATH=ml_models/spam/vectorizer.pkl
REPLY_MODEL_PATH=ml_models/reply_needed/reply_model.pkl
REPLY_VECTORIZER_PATH=ml_models/reply_needed/vectorizer.pkl
INTENT_MODEL_PATH=ml_models/intent/intent_model.pkl
INTENT_VECTORIZER_PATH=ml_models/intent/vectorizer.pkl
```

## Run the API

```bash
uvicorn app.main:app --reload
```

## API Endpoints

- `GET /health`
- `POST /analyze-email`
- `POST /predict-spam`
- `POST /predict-reply-needed`
- `POST /predict-intent`
- `POST /generate-reply`
- `POST /analyze-batch`

## Example Request

```bash
curl -X POST http://127.0.0.1:8000/analyze-email ^
  -H "Content-Type: application/json" ^
  -d "{\"email_id\":\"12345\",\"sender\":\"client@company.com\",\"subject\":\"Meeting tomorrow\",\"body\":\"Can we schedule a meeting tomorrow at 4 PM?\",\"timestamp\":\"2026-05-16T10:30:00\"}"
```

## Training

The Enron corpus in `dataset/emails.csv` is used with weak labels derived from deterministic heuristics.

Spam:

```bash
python ml_models/spam/train_spam.py --max-rows 50000 --model logreg
```

Reply-needed:

```bash
python ml_models/reply_needed/train_reply_classifier.py --max-rows 50000 --model xgb
```

Intent:

```bash
python ml_models/intent/train_intent.py --max-rows 50000 --model svm
```

Add `--tune` to run a small `GridSearchCV` hyperparameter search.

## Architecture

1. Request enters FastAPI.
2. Text is cleaned with NLTK-assisted preprocessing.
3. Spam classifier runs first.
4. If not spam, reply-needed and intent classifiers run.
5. If reply is allowed by rules, a deterministic template is selected and filled.
6. Results and logs are stored in SQLite.

## Response Rules

- If spam is true, reply is disabled and reply text is null.
- If reply_needed is false, reply text is null.
- If intent is newsletter or invoice, reply text is null.

## Notes

- The runtime can operate even if the `.pkl` artifacts are not trained yet; it falls back to deterministic heuristics.
- Models are loaded once at application startup.

## Model Manifest & Validation

- Manifest: [ml_models/manifest.json](ml_models/manifest.json#L1) contains workspace-relative paths, model types and `best_params` found during tuning.
- Validation report: [ml_models/validation_report.json](ml_models/validation_report.json#L1) contains inference timings and sample predictions produced by `ml_models/validate_models.py`.

Run the validator (PowerShell example using the workspace virtualenv):

```powershell
$env:PYTHONPATH='C:\Users\vivek\OneDrive\Desktop\SEM 6\email'
& 'c:\DL CP\myenv\Scripts\python.exe' ml_models/validate_models.py
```

This writes `ml_models/validation_report.json` and prints a short summary.

## Quick Run (Windows PowerShell)

Use your project virtualenv Python to ensure dependencies are available and set `PYTHONPATH` so `app` imports resolve from the project root.

Run tests:

```powershell
$env:PYTHONPATH='C:\Users\vivek\OneDrive\Desktop\SEM 6\email'
& 'c:\DL CP\myenv\Scripts\python.exe' -m pytest -q tests
```

Start the API server:

```powershell
$env:PYTHONPATH='C:\Users\vivek\OneDrive\Desktop\SEM 6\email'
& 'c:\DL CP\myenv\Scripts\python.exe' -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

PowerShell POST example (use `Invoke-RestMethod` to avoid quoting issues):

```powershell
$body = @{ sender='alice@example.com'; subject='Meeting request'; body='Can we meet tomorrow at 3pm?' } | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/analyze-email' -Method Post -Body $body -ContentType 'application/json' | ConvertTo-Json -Depth 5
```
