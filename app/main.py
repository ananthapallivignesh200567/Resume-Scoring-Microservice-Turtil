import os
import json
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from scorer import ResumeScorer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("resume-scorer")

# Initialize FastAPI app
app = FastAPI(
    title="Resume Scoring Microservice",
    description="Evaluates resumes against job goals and provides skill-based insights",
    version="1.0.0"
)

# Define request & response models
class ScoreRequest(BaseModel):
    student_id: str = Field(..., description="Unique student identifier")
    goal: str = Field(..., description="Target position or domain (e.g., Amazon SDE)")
    resume_text: str = Field(..., description="Full plain-text resume content")

class ScoreResponse(BaseModel):
    score: float = Field(..., description="Match score between 0.0 and 1.0")
    matched_skills: list[str] = Field(..., description="Skills found in resume that match goal")
    missing_skills: list[str] = Field(..., description="Skills required for goal but not found in resume")
    suggested_learning_path: list[str] = Field(..., description="Recommended steps to improve match")

# Load configuration at startup
def load_config() -> Dict[str, Any]:
    """Load and validate config.json file."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            
        # Validate required fields
        required_fields = [
            "version", 
            "minimum_score_to_pass", 
            "log_score_details", 
            "model_goals_supported", 
            "default_goal_model"
        ]
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")
                
        return config
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        logger.critical(f"Failed to load config: {str(e)}")
        raise RuntimeError(f"Configuration error: {str(e)}")

# Load goals data
def load_goals() -> Dict[str, list]:
    """Load goals.json containing required skills per goal."""
    goals_path = os.path.join(os.path.dirname(__file__), "..", "data", "goals.json")
    
    try:
        with open(goals_path, "r") as f:
            goals = json.load(f)
        return goals
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.critical(f"Failed to load goals data: {str(e)}")
        raise RuntimeError(f"Goals data error: {str(e)}")

# Startup event
@app.on_event("startup")
async def startup_event():
    try:
        # Load config and goals
        config = load_config()
        goals = load_goals()
        
        # Initialize the resume scorer and attach to app state
        app.state.scorer = ResumeScorer(config, goals)
        app.state.config = config
        
        logger.info(f"Resume Scorer initialized with {len(goals)} goals and {len(config['model_goals_supported'])} supported models")
    except Exception as e:
        logger.critical(f"Failed to initialize application: {str(e)}")
        # Exit the application if initialization fails
        os._exit(1)

# Error handler for internal exceptions
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Return health status of the service."""
    return {"status": "ok"}

# Version endpoint
@app.get("/version")
async def version():
    """Return version and model metadata."""
    return {
        "version": app.state.config["version"],
        "goals_supported": app.state.config["model_goals_supported"],
        "default_goal": app.state.config["default_goal_model"]
    }

# Main scoring endpoint
@app.post("/score", response_model=ScoreResponse)
async def score_resume(request: ScoreRequest):
    """Score a resume against a goal and return insights."""
    config = app.state.config
    
    # Check if goal is supported
    if request.goal not in config["model_goals_supported"]:
        logger.warning(f"Unsupported goal requested: {request.goal}, falling back to default")
        goal = config["default_goal_model"]
    else:
        goal = request.goal
    
    try:
        # Score the resume
        result = app.state.scorer.score_resume(
            student_id=request.student_id,
            goal=goal,
            resume_text=request.resume_text
        )
        
        # Log details if enabled
        if config["log_score_details"]:
            logger.info(
                f"Scored resume for student {request.student_id}: "
                f"goal={goal}, score={result['score']:.2f}, "
                f"matched={len(result['matched_skills'])}, "
                f"missing={len(result['missing_skills'])}"
            )
            
        return result
    except Exception as e:
        logger.error(f"Error scoring resume: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to score resume: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)