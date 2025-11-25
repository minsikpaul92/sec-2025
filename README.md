# Job Posting Validity Checker

**Team name:** codeXperts  
**Project title:** Job Posting Validity Checker  
**Competitors:** Tan Dat, Khai Ngo, Minsik Kim

## How to run (judges)

### Prerequisites
- Python 3.11+ (for backend/model components)
- Node.js 18+ (for frontend)

### Backend (FastAPI)
From the repository root:
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn pandas
uvicorn main:app --reload
```
- Serves endpoints under `http://localhost:8000/`.
- Reads training samples from `dataset/raw_postings/` relative to repo root.

### Model (rule + logistic regression)
From the repository root:
```bash
cd model_textbase
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python classify_posting.py
```
- Interactive: enter a test index (1-10 in `dataset/test_postings/`) or a path to any `.txt` file.

### ML experiments (transparency models)
From the repository root:
```bash
cd models_ml
python3 -m venv venv
source venv/bin/activate
pip install scikit-learn pandas numpy scipy
python models.py            # trains RF on job_postings_dataset.csv and saves transparency_model.pkl
python models_field_base.py # validates required fields in the CSV
```

### Frontend (React + Vite)
From the repository root:
```bash
cd frontend
npm install
npm run dev      # or npm run build && npm run preview
```

## Project structure
- `backend/`: FastAPI service exposing sample endpoints to list raw postings and validate JSON payloads.
- `frontend/`: React + Vite + Tailwind starter UI (shadcn/ui button demo).
- `model_textbase/`: Rule-first classifier with logistic regression fallback; uses text files in `dataset/`.
- `models_ml/`: ML experiments on the CSV dataset (RandomForest transparency scoring, field validation).
- `dataset/`: Shared data (CSV and raw/test postings).

## Key code aspects
- **Rule + model hybrid (`model_textbase/classify_posting.py`):** Hard rules enforce required fields, salary range width, AI disclosure, vacancy type, and disallow explicit Canadian-experience requirements. Rule hits force `INVALID`; otherwise a balanced `LogisticRegression` on rule-derived features decides. CLI maps numeric input to sorted test files and reports reasons.
- **Backend endpoints (`backend/main.py`):** `/jobs-postings/` returns raw posting texts grouped by filename; `/jobs-postings/text-base` echoes payload; `/jobs-postings/field-base` checks for required JSON fields.
- **ML transparency (`models_ml/models.py`):** Builds TF-IDF + numeric features, trains a RandomForest, and saves `transparency_model.pkl`; reports compliance scores.
- **Field validation utility (`models_ml/models_field_base.py`):** Confirms presence of required fields across the CSV and prints counts.

## AI usage and citations
- OpenAI ChatGPT was used for iterative code refactoring, feature design, debugging assistance, and README drafting; all outputs were reviewed and edited manually.
- GitHub Copilot is cited in-code (see `backend/main.py` comment) for a helper that reads files and was wrapped in a FastAPI endpoint.
- Libraries/resources: scikit-learn, numpy, scipy, pandas, FastAPI, uvicorn, React, Vite, Tailwind CSS, shadcn/ui. Data comes from `dataset/job_postings_dataset.csv` and text samples in `dataset/raw_postings` and `dataset/test_postings`.
