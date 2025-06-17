# SciGenie 🧠 
AI-Powered Research Assistant

Turn **datasets → discoveries** in minutes.  
Upload your data ➜ automatic cleaning & EDA ➜ one-click AutoML (Random Forest, XGBoost, Auto-Sklearn) ➜ GPT-4 abstract ➜ downloadable PDF report ➜ Zotero‐ready references.

<img width="1246" alt="Screenshot 2025-06-17 at 6 04 00 PM" src="https://github.com/user-attachments/assets/c85524c6-0f2d-4430-8d61-2ce7dcc79703" />

---

## ✨ Features

| Module               | What it Does                                              |
|----------------------|-----------------------------------------------------------|
| **Upload**           | CSV / XLSX ingestion, schema & missing-value detection    |
| **EDA**              | Summary stats, histogram, correlation heat-map           |
| **AutoML**           | Random Forest • XGBoost • Auto-Sklearn (best model `.pkl`)|
| **LLM Abstract**     | 120-150 word research abstract via GPT-4 (fallback text) |
| **PDF Report**       | One-click PDF — EDA plots, model metrics, abstract       |
| **Zotero Export**    | Search arXiv → download `.bib` for Zotero / LaTeX        |

---

## 🏗️ Architecture

frontend/   # React + Tailwind (Vercel-ready)
backend/    # FastAPI – REST endpoints
│  └── routers/ (upload, eda, automl, abstract, report)
docker-compose.yml

---

## 🚀 Quick Start (local)

```bash
# 1. clone & move in
git clone https://github.com/<you>/ai-research-assistant.git
cd ai-research-assistant

# 2. spin up everything
docker compose up --build
# ➜ frontend: http://localhost:5173
# ➜ api:      http://localhost:8000/docs

<details>
<summary>Manual setup (no Docker)</summary>


# Backend
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend
cd ../frontend
npm i
npm run dev          # Vite or Next.js

</details>

#🗺️ Roadmap
	•	OAuth push straight to Zotero library
	•	SHAP feature-importance visualizations
	•	User accounts & team share links
	•	GPU-powered Auto-DL for images / text

⸻

#🤝 Contributing
	1.	Fork ✔︎  → 2. Create branch ✔︎  → 3. Commit ✔︎ → 4. PR
Feel free to open issues—bug reports & feature ideas welcome!

⸻

#📄 License

MIT © 2025 Md. Mehedi Hasan


