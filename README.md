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

## AI Usage & Citation Statement

AI Tools Used

This project used OpenAI ChatGPT and GitHub Copilot to support development tasks such as code refactoring, feature brainstorming, logic troubleshooting, debugging assistance, and documentation drafting. All AI-generated suggestions were manually reviewed, validated, and integrated by the development team.

Purpose of AI Assistance

AI tools were used to:
	•	improve development efficiency and explore alternative solution approaches,
	•	brainstorm rule-based features aligned with Ontario’s 2026 Job Posting Transparency requirements,
	•	refine the logic and flow of the hybrid rule–plus–logistic-regression model,
	•	identify issues in early iterations of classify_posting.py,
	•	enhance regex patterns for extracting salary ranges and required fields,
	•	produce and revise explanatory documentation (README sections, comments),
	•	clarify edge cases and suggest additional test cases for dataset/test_postings.

AI tools did not make autonomous system design, architectural, or algorithmic decisions.
All final implementations, including rule sets, scoring logic, and model parameters, were directly authored and verified by the development team.

Human Oversight

All code, logic, and model behavior were written, validated, and tested by the codeXperts team.
AI assistance was treated strictly as optional input, and all outputs were significantly edited before integration.
No fully-generated code modules were used.

Specific AI Usage Notes
	•	ChatGPT assisted with conceptual explanations, reasoning tasks, and iterative improvement of transparency-validation logic.
	•	GitHub Copilot assisted minimally, specifically with a helper method in backend/main.py to list posting files (documented in-code).

Compliance Disclosure

In accordance with the Working for Workers Four Act, 2024 (S.O. 2024, c.3 – Bill 149), this project fully discloses the use of AI in its development.
AI tools were used only during development and documentation. They do not participate in automated decision-making within the job-posting evaluation workflow.

References
	•	Working for Workers Four Act, 2024 (S.O. 2024, c.3 – Bill 149).
	•	SEC 2025 – Problem Brief: Defines transparency requirements, required posting fields, and compliance expectations.
	•	SEC 2025 – Opening Briefing: System tasks, expectations, and challenge context.
	•	SEC 2025 – FAQ & Rules: Judging criteria, allowed tools, and AI-usage disclosure requirements.
	•	SEC 2025 Job Postings Dataset (dataset/job_postings_dataset.csv, dataset/raw_postings/, dataset/test_postings/).
	•	Python libraries: scikit-learn, numpy, pandas, FastAPI, uvicorn.
	•	Frontend libraries: React, Vite, Tailwind CSS.
