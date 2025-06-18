# 🧠 Resume Scoring Microservice

An AI-powered microservice that evaluates resumes for targeted goals (e.g., **Amazon SDE**, **ML Internship**), assigns a score based on skill match, and suggests personalized learning paths. Built using FastAPI, Logistic Regression, and TF-IDF — fully offline and Dockerized.

## ✅ Features
- 🚀 Offline ML scoring using TF-IDF + Logistic Regression (per goal)
- 📊 Returns score, matched/missing skills, and a skill-based learning path
- ⚙️ Controlled by `config.json` (thresholds, goals, defaults)
- 🧠 Rule-based skill logic from `goals.json`
- 🧪 Unit tested with fallback and edge case handling
- 🐳 Fully containerized with `Dockerfile`
- 🔌 FastAPI interface with Swagger documentation

## 📁 Project Structure
```
resume-scorer/
├── app/
│   ├── main.py                 # FastAPI server
│   ├── scorer.py               # Model scoring logic
│   └── model/
│       ├── amazon_sde_model.pkl
│       └── tfidf_vectorizer.pkl
├── data/
│   ├── training_amazon_sde.json
│   └── goals.json              # Static skill sets per goal
├── config.json                 # Global app settings
├── schema.json                 # Input/output schema reference
├── requirements.txt            # Dependencies
├── Dockerfile                  # Container config
├── README.md                   # Project guide
└── tests/
    └── test_score.py           # Unit tests
```

## 🧪 Functional Overview

### Input Format – POST `/score`
```json
{
  "student_id": "stu_1084",
  "goal": "Amazon SDE",
  "resume_text": "Final year student skilled in Java, Python, DSA, SQL, REST APIs..."
}
```

### Output Format
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

## ⚙️ Configuration – `config.json`
```json
{
  "version": "1.0.0",
  "minimum_score_to_pass": 0.6,
  "log_score_details": true,
  "model_goals_supported": ["Amazon SDE", "ML Internship"],
  "default_goal_model": "Amazon SDE"
}
```
- **Fail-safe**: App will terminate if config is missing or malformed.

## 📚 Goals & Skill Logic – `goals.json`
Example:
```json
{
  "Amazon SDE": ["Java", "Data Structures", "System Design", "SQL"],
  "ML Internship": ["Python", "Numpy", "Scikit-learn", "Linear Algebra"]
}
```

## 🧠 Model Architecture
- TF-IDF Vectorizer for resume text
- Logistic Regression (binary classifier) per goal
- Thresholding based on `config.json`
- Output: Score ∈ [0.0, 1.0]

## 🔍 Testing & Validation
- Run unit tests: `tests/test_score.py`
- Validate:
  - Response format
  - Accuracy for known inputs
  - Fail gracefully on:
    - Unknown goal
    - Malformed/empty input
    - Missing config

## 🧪 API Endpoints

| Method | Endpoint       | Description                        |
|--------|----------------|------------------------------------|
| POST   | `/score`       | Evaluate resume, return insights   |
| GET    | `/health`      | Status check (`{"status": "ok"}`) |
| GET    | `/version`     | Returns model + config metadata    |

## 🚀 Running Locally

### 🧪 Setup
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Visit: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## 🐳 Docker Deployment

### Build the image
```bash
docker build -t resume-scorer .
```

### Run the container
```bash
docker run -p 8000:8000 resume-scorer
```

### Test locally
```bash
curl -X POST "http://127.0.0.1:8000/score" \
     -H "Content-Type: application/json" \
     -d @sample_input.json
```

## 📌 Sample Input JSON
```json
{
  "student_id": "stu_0001",
  "goal": "Amazon SDE",
  "resume_text": "Skilled in Java, Data Structures, REST APIs, and SQL. Built multiple microservices in Spring Boot."
}
```

## 📄 Licensing & Attribution
- Built under Turtil Internship Program
- For academic and educational use only
- Credits to LLMs like ChatGPT, Claude for ideation support (final logic is fully original)

## 💬 Contact
📨 vigneshananthapalli@turtilintern.com
📧 vigneshananthapalli67@gmail.com
