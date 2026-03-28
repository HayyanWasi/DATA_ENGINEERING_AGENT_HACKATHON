## AutoML Pipeline Agent

**Team:** Coloners
**Theme & Challenge:** Theme 5 - Agentic Data Engineering Pipeline
**Track:** Open Innovation

### Problem statement
Non-technical users struggle to go from a raw dataset to a production-ready ML model. This system lets anyone upload a CSV/XLSX, pick a target column, and receive cleaned data, a tuned model, evaluation charts, and downloadable artifacts — zero coding required.

### Why multi-agent?
An end-to-end ML pipeline spans wildly different reasoning domains — data profiling, cleaning strategy, feature engineering, model selection, hyperparameter tuning, evaluation, and artifact packaging. A single monolithic agent would hallucinate across domain boundaries and be impossible to debug. By decomposing into 14 specialized agents grouped under 10 stage orchestrators, each agent stays focused on one task, failures are isolated to a single stage, and stages can evolve independently without breaking the whole pipeline.

### Agent architecture
| Agent | Role |
|-------|------|
| Master Orchestrator | Enforces 10-stage sequential flow, handles blocking vs non-blocking errors |
| Data Cleaning Orchestrator | Runs Analyzer → Strategist → Executor to clean raw data |
| EDA Orchestrator | Generates exploratory statistics, distribution charts, and correlation analysis |
| Feature Engineering Orchestrator | Encodes categoricals, transforms numerics, writes engineered dataset |
| Feature Scaling Orchestrator | Selects and applies scaling methods (StandardScaler, MinMax, etc.) |
| Class Imbalance Orchestrator | Detects imbalance, applies SMOTE for classification tasks |
| Model Training Orchestrator | Trains 3 candidate models (Random Forest, Logistic Regression, XGBoost) |
| Model Selection Orchestrator | Compares candidates and selects the best model |
| Hyperparameter Tuning Orchestrator | Runs GridSearchCV with cross-validation on selected model |
| Model Evaluation Orchestrator | Generates confusion matrix, ROC curve, feature importance charts |
| Final Output Orchestrator | Packages results_manifest.json, final_model.joblib, and all artifacts |

Each orchestrator follows a 3-phase pattern internally: **Analyzer Agent** (profiles data) → **Strategist Agent** (plans approach) → **Executor Agent** (implements in sandboxed Python).

### How to run
1. **Prerequisites:** Python 3.13, Node.js, PostgreSQL running locally

2. **Backend**
   ```bash
   cd backend
   python -m venv venv && source venv/bin/activate
   pip install -e .          # or: uv sync
   # Create .env with:
   #   GOOGLE_API_KEY=<your_gemini_key>
   #   DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/hackathon_db
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Frontend**
   ```bash
   cd frontend
   npm install
   npm run dev               # opens http://localhost:3000
   ```

4. **Usage:** Upload a dataset → enter target column → pipeline runs automatically → download model, charts, and cleaned data from the results page.

### Demo
[Link to demo video]

### Tech stack
- **Frontend:** Next.js 16, React 19, TypeScript, Tailwind CSS 4, Motion
- **Backend:** FastAPI, Uvicorn, SQLAlchemy (async) + asyncpg, PostgreSQL
- **ML:** Pandas, scikit-learn, XGBoost, Joblib, SMOTE (imbalanced-learn)
- **AI:** Google ADK (Agent Development Kit), Gemini 2.0-flash / 2.5-flash
- **Other:** httpx, Lucide React
