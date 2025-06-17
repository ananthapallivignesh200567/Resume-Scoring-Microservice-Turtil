# 🧠 Resume Scoring Microservice

[![Docker Image](https://img.shields.io/badge/docker-ready-blue)](https://hub.docker.com/)
[![FastAPI](https://img.shields.io/badge/fastapi-🔥-green)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![CI](https://img.shields.io/github/actions/workflow/status/your-org/resume-scorer/ci.yml?branch=main)](https://github.com/your-org/resume-scorer/actions)

> 🚀 A fully offline, ML-powered FastAPI microservice that evaluates resumes against specific goals (e.g., "Amazon SDE", "GATE CSE") and outputs match scores, skill gaps, and learning paths.

---

## ✨ Features

* ✅ **Offline-first**: No internet or cloud dependencies
* 🤖 **ML-Based Scoring**: TF-IDF + Logistic Regression per goal
* 🧠 **Skill Matching Engine**: Matched + missing skills + learning recommendations
* 🐳 **Fully Dockerized**: Containerized for fast and reproducible deployment
* ⚙️ **Configurable**: Controlled via `config.json`
* 📡 **API-first**: FastAPI-powered with schema validation and testing

---

## 📦 Installation

```bash
git clone https://github.com/your-org/resume-scorer.git
cd resume-scorer
docker build -t resume-scorer .
docker run -p 8000:8000 resume-scorer
```

---

## 🥪 API Usage

### 🔹 Endpoint: `POST /score`

```json
{
  "student_id": "stu_1084",
  "goal": "Amazon SDE",
  "resume_text": "Skilled in Java, Python, REST APIs, DSA"
}
```

### 🔹 Response

```json
{
  "score": 0.81,
  "matched_skills": ["Java", "DSA", "SQL"],
  "missing_skills": ["System Design"],
  "suggested_learning_path": [
    "Learn basic system design concepts",
    "Complete SQL joins and indexing course"
  ]
}
```

---

## ⚖️ Configuration

File: `config.json`

```json
{
  "version": "1.0.0",
  "minimum_score_to_pass": 0.6,
  "log_score_details": true,
  "model_goals_supported": ["Amazon SDE", "ML Internship"],
  "default_goal_model": "Amazon SDE"
}
```

---

## 📁 Project Structure

```
resume-scorer/
├── app/
│   ├── main.py             # FastAPI app
│   ├── scorer.py           # Core model logic
│   └── model/              # Pickled models per goal
├── data/                   # Training and goal definitions
│   ├── training_amazon_sde.json
│   └── goals.json
├── config.json
├── schema.json             # JSON schema definitions
├── Dockerfile
├── requirements.txt
├── README.md
└── tests/
    └── test_score.py       # Unit tests
```

---

## 🧠 Model Details

* **Vectorizer**: Shared TF-IDF
* **Classifier**: Logistic Regression (Binary)
* **Goal-specific models**: e.g., `amazon_sde_model.pkl`

### 🔹 Training Format

```json
[
  {
    "goal": "Amazon SDE",
    "resume_text": "Java, C++, REST APIs",
    "label": 1
  },
  {
    "goal": "Amazon SDE",
    "resume_text": "Mechanical CAD, Civil Design",
    "label": 0
  }
]
```

---

## 📊 Health & Monitoring

* `GET /health` → `{ "status": "ok" }`
* `GET /version` → returns model + config metadata

---

## ✅ Testing

```bash
pytest tests/
```

Includes:

* High/low scoring resumes
* Invalid/missing fields
* Malformed JSON
* Unknown goal fallback

---

## 🔒 Constraints

* ❌ No OpenAI/GPT APIs
* ✅ Must work offline only
* 📏 Input/output via structured JSON
* 📦 Fully Dockerized & isolated

---

## 🗓️ Timeline Summary

| Week | Milestone                         |
| ---- | --------------------------------- |
| 1    | Resume samples + goals definition |
| 2    | Train models per goal             |
| 3    | API + core logic                  |
| 4    | Skill matching + Dockerization    |
| 5    | Testing + Config enforcement      |
| 6    | QA, Documentation, CI/CD          |

---

## 🤝 Contributing

Contributions are welcome! Please submit a pull request or open an issue.
I mostly depended on AI LLMs like claude and chatgpt.

---



---

## 🔗 Links

* [FastAPI Docs](https://fastapi.tiangolo.com/)
* [Scikit-learn](https://scikit-learn.org/)
* [Docker Hub](https://hub.docker.com/)
