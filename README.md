# ChurnSignal - Local AI-Powered Telco Churn Assistant

ChurnSignal is a proof of concept for predicting telecom customer churn and making the result usable by a marketing team. A local open-source LLM collects customer information in natural language, validates the complete 19-field profile, calls a classical machine-learning ensemble, and returns a churn score with a practical retention action.

The solution runs locally. Ollama structures the conversation; it does not replace the churn model or invent missing customer values.

## Assessment coverage

| Requested deliverable | Implementation |
|---|---|
| Churn classification model | Logistic Regression and CatBoost ensemble trained and evaluated in `notebooks/etislat.ipynb` |
| Conversational model-input pipeline | Stateful 19-field collection and clarification flow in `src/chatbot.py` |
| Open-source LLM | Local `qwen3:1.7b` through Ollama |
| Structured natural-language extraction | Pydantic-constrained JSON patch extraction plus deterministic typo and shorthand normalization |
| API | FastAPI endpoints for health, prediction, chat, reset, reports, and bulk scoring |
| Understandable result | Probability, threshold comparison, risk level, profile signals, and next-best action |
| Architecture diagram | `diagram/system-architecture.png`, with editable SVG and Mermaid source |
| Supporting evidence | Notebook figures in `imgs/` and the concise PDF project report in `output/pdf/` |

## Architecture

![ChurnSignal system architecture](diagram/system-architecture.png)

The interaction path is deliberately separated into two responsibilities:

1. **Language layer:** the chatbot and local LLM extract only customer facts, normalize them to the training schema, remember confirmed values, and request missing or conflicting fields.
2. **Decision layer:** the saved classical models calculate the churn score. This keeps the prediction deterministic, testable, and independent from LLM wording.

The complete source diagram is available as:

- `diagram/system-architecture.png` - presentation-ready image.
- `diagram/system-architecture.svg` - editable vector asset.
- `diagram/system-architecture.mmd` - Mermaid source.

## Dataset and problem definition

- **Rows:** 7,043 customers.
- **Original columns:** 21.
- **Target:** `Churn`, mapped from Yes/No to 1/0.
- **Positive class:** 1,869 churners, or 26.54% of the dataset.
- **Negative class:** 5,174 retained customers, or 73.46%.
- **Model inputs:** 19 fields after excluding `Customer_ID` and the target.

The imbalance explains why accuracy is not used alone. A model predicting “No Churn” for everyone would be about 73.46% accurate while catching zero churners.

![Target distribution](imgs/fig-01-target-distribution.png)

## Notebook workflow and why each stage exists

| Stage | What is done | Why it is necessary |
|---|---|---|
| Schema inspection | Shape, samples, dtypes, column names, unique values, and target counts | Detects parsing problems and establishes what one row represents |
| Header cleaning | Leading and trailing whitespace is removed | Prevents schema and inference failures caused by visually hidden characters |
| `Total_Charges` repair | Blank strings are converted to missing numeric values; the 11 zero-tenure rows are filled with 0 | Uses a domain-consistent value instead of blind median imputation |
| Integrity checks | Missingness, duplicates, unique customer IDs, and phone/internet dependency rules | Prevents logically impossible profiles and accidental identifier leakage |
| EDA | Numeric distributions, category counts, churn-conditioned plots, and outlier views | Shows class imbalance, skew, segment behavior, and candidate relationships |
| Statistical tests | Chi-square plus Cramer's V, Mann-Whitney U plus rank-biserial effect, and Spearman correlation | Separates statistical evidence from practical effect size without relying on normality |
| Holdout split | Stratified 80/20 split using `random_state=42` | Preserves churn prevalence and protects 1,409 customers for final evaluation |
| Pipeline preprocessing | One-hot encoding and scaling are fitted inside cross-validation | Prevents validation-fold leakage and stabilizes Logistic Regression optimization |
| Baseline | Most-frequent DummyClassifier | Proves why accuracy alone is misleading |
| Model screening | Logistic Regression, Decision Tree, Random Forest, XGBoost, and CatBoost | Compares linear, bagged, and boosted hypotheses before selecting finalists |
| Tuning | Grid search for the small Logistic space and randomized search for CatBoost | Matches search cost to the size of each hyperparameter space |
| OOF prediction | Every training row is scored by a fold model that did not train on it | Enables honest ensemble-weight and threshold selection without touching the test set |
| Feature engineering | Six domain summaries are created | Makes service depth, support coverage, newness, and payment behavior explicit |
| Ensemble | Raw Logistic probability and feature-engineered CatBoost probability are blended | Combines stable linear structure with non-linear interactions |
| Threshold selection | The action threshold is selected from training-only OOF scores | Converts ranking quality into a business operating point without test leakage |
| Final test | Frozen models, weights, and threshold are evaluated once | Produces the unbiased final PoC estimate |
| Persistence parity | Models and JSON configuration are saved, reloaded, and compared | Proves that API inference reproduces notebook inference |

## Feature engineering

| Feature | Definition | Intended signal |
|---|---|---|
| `Num_Services` | Count of six active add-on services | Overall product depth and switching friction |
| `Num_Support_Services` | Security, backup, device protection, and tech support count | Protective support coverage |
| `Num_Streaming_Services` | Streaming TV plus streaming movies | Entertainment engagement |
| `Is_New_Customer` | Tenure less than or equal to six months | Early-life churn exposure |
| `Auto_Payment` | Automatic bank transfer or automatic credit card | Lower billing friction |
| `No_Security_And_No_Support` | No online security and no technical support | Explicit high-risk service interaction |

The engineered CatBoost gain is intentionally reported as small. Tree models can already learn many interactions from raw flags, so a large improvement was neither assumed nor claimed.

## Model comparison

All values below come from training-only out-of-fold predictions. Each candidate receives its own selected threshold, so one model is not unfairly compared at 0.50 while another uses an optimized cutoff.

| Candidate | ROC-AUC | PR-AUC | Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Feature-engineered CatBoost | 0.8500 | 0.6688 | 0.33 | 0.5661 | 0.7391 | **0.6411** |
| Raw CatBoost | 0.8498 | 0.6677 | 0.38 | **0.6027** | 0.6809 | 0.6394 |
| **Hybrid ensemble** | **0.8505** | **0.6705** | **0.31** | 0.5525 | **0.7565** | 0.6386 |
| Raw Logistic Regression | 0.8459 | 0.6581 | 0.33 | 0.5582 | 0.7411 | 0.6368 |

![Final model comparison](imgs/fig-18-final-model-comparison.png)

### Why the hybrid ensemble is selected

The hybrid model is preferred from a business-risk perspective because it achieves the highest recall while maintaining nearly the same F1 score as the other finalists. It also has the highest ROC-AUC and PR-AUC in the final OOF comparison.

This is not a claim that the hybrid dominates every metric. Feature-engineered CatBoost has a slightly higher F1, and raw CatBoost has higher precision. The hybrid is selected because missing a real churner is assumed to be more costly than contacting some customers who would have stayed. The production choice must therefore be revisited when the real costs of false negatives, false positives, offer value, and campaign capacity are known.

## Final holdout performance

The frozen 30% Logistic / 70% CatBoost blend and 0.31 threshold produce:

| Metric | Holdout result | Business interpretation |
|---|---:|---|
| Accuracy | 76.44% | Overall label correctness |
| Precision | 53.99% | About 54 of every 100 contacted customers are actual churners |
| Recall | 75.94% | About 76 of every 100 actual churners are identified |
| F1 | 63.11% | Precision-recall balance at the selected action threshold |
| ROC-AUC | 84.65% | Strong ranking of churners above non-churners across thresholds |

The confusion matrix contains 793 true negatives, 242 false positives, 90 false negatives, and 284 true positives. The model catches 284 of 374 actual churners and creates a retention queue of 526 customers.

![Final holdout confusion matrix](imgs/fig-19-final-test-confusion-matrix.png)

## Conversational flow

1. The user describes any known customer details in ordinary language.
2. Deterministic parsing handles common shorthand, typos, numeric answers, and ordered multi-answer replies.
3. Ollama returns a structured patch constrained by the Pydantic schema.
4. Confirmed values are stored for the session.
5. Relationship rules automatically handle “No phone service” and “No internet service” dependencies.
6. The assistant asks only for missing, invalid, or conflicting fields.
7. Prediction starts only when all 19 model inputs validate.
8. The API returns the model score, class, threshold, business signals, and recommended action.

The profile signals are operational guidance derived from the validated profile. They are not causal explanations and should not be presented as proof that a specific offer will prevent churn.

## Bulk scoring

The optional bulk path accepts `.csv` or `.xlsx`, maps known header aliases, uses Ollama only for unfamiliar headers, validates each row independently, and scores valid rows in one vectorized model call.

The downloaded ZIP contains:

- A cloned Excel workbook with `Churn` and `Churn_Percentage` appended.
- A PDF containing the finding and suggested action for every source row.
- Invalid rows retained as `Error` rather than silently deleted.

Limits are 15 MB and 20,000 customer rows per upload.

## Run locally

### Prerequisites

- Python 3.9 or compatible environment used by the project.
- Node.js 22.13 or newer for the frontend.
- Ollama with `qwen3:1.7b` installed.

```powershell
ollama pull qwen3:1.7b
```

### 1. Start the API

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn src.Api:app --reload --host 127.0.0.1 --port 8000
```

Health check: `http://127.0.0.1:8000/health`  
Interactive API schema: `http://127.0.0.1:8000/docs`

### 2. Start the frontend

In another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:3000`.

### 3. Run backend tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Current verified result: **19 passed**.

## API surface

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | API and local Ollama readiness |
| POST | `/predict` | Structured single-customer prediction |
| POST | `/chat` | Stateful conversational collection and prediction |
| DELETE | `/chat/{session_id}` | Reset one conversation |
| POST | `/report` | Generate one customer assessment PDF |
| POST | `/bulk-predict` | Score CSV/Excel and return an Excel/PDF ZIP package |

## Project structure

```text
telco-churn-assistant/
|-- Data/                  Source telco churn dataset
|-- notebooks/             Executed analysis and training notebook
|-- models/                Logistic, CatBoost, ensemble, and feature artifacts
|-- src/                   FastAPI, chatbot, validation, inference, bulk, reports
|-- frontend/              ChurnSignal web interface
|-- tests/                 Backend API and chatbot tests
|-- imgs/                  Notebook figures, result figures, and UI evidence
|-- diagram/               Architecture PNG, SVG, and Mermaid source
|-- output/pdf/            Concise project report
|-- requirements.txt       Python environment dependencies
`-- README.md              Project guide
```

## Saved inference contract

| Artifact | Responsibility |
|---|---|
| `models/logistic_raw.joblib` | Raw-feature preprocessing and tuned Logistic Regression |
| `models/catboost_feature_engineered.cbm` | Tuned feature-engineered CatBoost |
| `models/ensemble_config.json` | 0.30/0.70 weights, positive class, and 0.31 threshold |
| `models/feature_config.json` | Exact raw, engineered, categorical, and numerical feature lists |

Serving must load all four artifacts. Saving only the classifiers would lose feature order, ensemble policy, or the chosen decision threshold.

## PoC boundaries

- The split is random because the dataset contains no event timestamp; production should validate on a later time period.
- OOF evaluation is appropriate for this PoC, while nested CV would provide a stricter model-selection estimate.
- Probability calibration is not measured, so the percentage should be treated as a model-estimated churn risk.
- Chat sessions are stored in process memory and reset when the API restarts.
- Production deployment would add authentication, persistent state, latency/error monitoring, data drift, prediction drift, and delayed-label performance monitoring.
- The current threshold must be re-optimized when actual campaign costs and retention capacity are available.

## Additional evidence

All 19 notebook and documentation figures are indexed in [`imgs/README.md`](imgs/README.md). The short submission report is generated as `output/pdf/telco-churn-poc-report.pdf`.

