import os
import json
import logging
import joblib
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from datetime import datetime ,timezone
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import request_validation_exception_handler
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from typing import Union, List

from .scorer import ResumeScorer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("resume-scorer")


class LearningPathItem(BaseModel):
    path: Union[str, List[str]]
    course: str

# Define request & response models
class ScoreRequest(BaseModel):
    student_id: str = Field(..., description="Unique student identifier")
    goal: str = Field(..., description="Target position or domain (e.g., Amazon SDE)")
    resume_text: str = Field(..., description="Full plain-text resume content")

class ScoreResponse(BaseModel):
    score: float = Field(..., description="Match score between 0.0 and 1.0")
    matched_skills: list[str] = Field(..., description="Skills found in resume that match goal")
    missing_skills: list[str] = Field(..., description="Skills required for goal but not found in resume")
    suggested_learning_path: list[LearningPathItem] = Field(..., description="Recommended steps to improve match")

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

async def _check_app_state() -> Dict[str, Any]:
    """Check if FastAPI app state is properly initialized."""
    try:
        if not hasattr(app.state, 'scorer'):
            return {
                "status": "error",
                "message": "ResumeScorer not initialized in app state",
                "details": {"missing_component": "scorer"}
            }
            
        if not hasattr(app.state, 'config'):
            return {
                "status": "error", 
                "message": "Config not loaded in app state",
                "details": {"missing_component": "config"}
            }
            
        return {
            "status": "ok",
            "message": "App state properly initialized",
            "details": {
                "has_scorer": True,
                "has_config": True,
                "scorer_type": type(app.state.scorer).__name__
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": "App state check failed",
            "details": {"error": str(e)}
        }




#@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        config = load_config()
        goals = load_goals()

        # Attach scorer and config to app state
        app.state.scorer = ResumeScorer(config, goals)
        app.state.config = config

        logger.info(
            f"Resume Scorer initialized with {len(goals)} goals and {len(config['model_goals_supported'])} supported models"
        )
    except Exception as e:
        logger.critical(f"Failed to initialize application: {str(e)}")
        os._exit(1)  # Exit app if init fails

    yield  # App runs here

# Create FastAPI app using the lifespan context
app = FastAPI(
    title="Resume Scoring Microservice",
    description="Evaluates resumes against job goals and provides skill-based insights",
    version="1.0.0",
    lifespan=lifespan  # ← this replaces on_event("startup")
)

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
    """
    Comprehensive health check for the resume scoring microservice.
    Returns detailed status of all critical components.
    """
    health_status = {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "resume-scorer",
        "checks": {}
    }
    
    overall_healthy = True
    
    try:
        # 1. Check config.json
        config_status = await _check_config()
        health_status["checks"]["config"] = config_status
        if config_status["status"] != "ok":
            overall_healthy = False
            
        # 2. Check TF-IDF vectorizer
        vectorizer_status = await _check_vectorizer()
        health_status["checks"]["vectorizer"] = vectorizer_status
        if vectorizer_status["status"] != "ok":
            overall_healthy = False
            
        # 3. Check trained models
        models_status = await _check_models()
        health_status["checks"]["models"] = models_status
        if models_status["status"] != "ok":
            overall_healthy = False
            
        # 4. Check goals.json
        goals_status = await _check_goals()
        health_status["checks"]["goals"] = goals_status
        if goals_status["status"] != "ok":
            overall_healthy = False
            
        # 5. Check model registry
        registry_status = await _check_model_registry()
        health_status["checks"]["registry"] = registry_status
        if registry_status["status"] != "ok":
            overall_healthy = False
            
        # 6. Performance test
        performance_status = await _check_performance()
        health_status["checks"]["performance"] = performance_status
        
        # Overall status
        if not overall_healthy:
            health_status["status"] = "degraded"
            
    except Exception as e:
        health_status["status"] = "error"
        health_status["error"] = str(e)
        overall_healthy = False
    
    # Return appropriate HTTP status
    if health_status["status"] == "error":
        raise HTTPException(status_code=503, detail=health_status)
    elif health_status["status"] == "degraded":
        raise HTTPException(status_code=200, detail=health_status)  # Still return 200 but with warnings
    
    return health_status

async def _check_config() -> Dict[str, Any]:
    """Check if config.json exists and is valid."""
    try:
        config_path = "config.json"
        if not os.path.exists(config_path):
            return {
                "status": "error",
                "message": "config.json not found",
                "details": {"path": config_path}
            }
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate required fields
        required_fields = [
            "version", "minimum_score_to_pass", 
            "model_goals_supported", "default_goal_model"
        ]
        
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            return {
                "status": "error",
                "message": "Invalid config format",
                "details": {"missing_fields": missing_fields}
            }
        
        return {
            "status": "ok",
            "message": "Config loaded successfully",
            "details": {
                "version": config.get("version"),
                "supported_goals": len(config.get("model_goals_supported", [])),
                "min_score_threshold": config.get("minimum_score_to_pass")
            }
        }
        
    except json.JSONDecodeError as e:
        return {
            "status": "error",
            "message": "Invalid JSON in config.json",
            "details": {"error": str(e)}
        }
    except Exception as e:
        return {
            "status": "error",
            "message": "Config check failed",
            "details": {"error": str(e)}
        }

async def _check_vectorizer() -> Dict[str, Any]:
    """Check if TF-IDF vectorizer exists and loads correctly."""
    try:
        vectorizer_path = "app/model/tfidf_vectorizer.pkl"
        if not os.path.exists(vectorizer_path):
            return {
                "status": "error",
                "message": "TF-IDF vectorizer not found",
                "details": {"path": vectorizer_path}
            }
        
        # Try to load vectorizer
        vectorizer = joblib.load(vectorizer_path)
        vocab_size = len(vectorizer.vocabulary_) if hasattr(vectorizer, 'vocabulary_') else 0
        
        return {
            "status": "ok",
            "message": "Vectorizer loaded successfully",
            "details": {
                "vocabulary_size": vocab_size,
                "file_size_mb": round(os.path.getsize(vectorizer_path) / (1024*1024), 2)
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": "Vectorizer load failed",
            "details": {"error": str(e)}
        }

async def _check_models() -> Dict[str, Any]:
    """Check if trained models exist and load correctly."""
    try:
        model_dir = "app/model"
        if not os.path.exists(model_dir):
            return {
                "status": "error",
                "message": "Model directory not found",
                "details": {"path": model_dir}
            }
        
        # Find all model files
        model_files = [f for f in os.listdir(model_dir) if f.endswith('_model.pkl')]
        
        if not model_files:
            return {
                "status": "error",
                "message": "No trained models found",
                "details": {"directory": model_dir}
            }
        
        loaded_models = {}
        failed_models = []
        
        for model_file in model_files:
            try:
                model_path = os.path.join(model_dir, model_file)
                model = joblib.load(model_path)
                
                goal_name = model_file.replace('_model.pkl', '').replace('_', ' ').title()
                loaded_models[goal_name] = {
                    "file": model_file,
                    "size_mb": round(os.path.getsize(model_path) / (1024*1024), 2)
                }
                
            except Exception as e:
                failed_models.append({"file": model_file, "error": str(e)})
        
        status = "ok" if not failed_models else ("degraded" if loaded_models else "error")
        
        return {
            "status": status,
            "message": f"Models check completed: {len(loaded_models)} loaded, {len(failed_models)} failed",
            "details": {
                "loaded_models": loaded_models,
                "failed_models": failed_models,
                "total_models": len(model_files)
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": "Models check failed",
            "details": {"error": str(e)}
        }

async def _check_goals() -> Dict[str, Any]:
    """Check if goals.json exists and is valid."""
    try:
        goals_path = "data/goals.json"
        if not os.path.exists(goals_path):
            return {
                "status": "error",
                "message": "goals.json not found",
                "details": {"path": goals_path}
            }
        
        with open(goals_path, 'r') as f:
            goals = json.load(f)
        
        if not isinstance(goals, dict):
            return {
                "status": "error",
                "message": "Invalid goals.json format",
                "details": {"expected": "dict", "got": type(goals).__name__}
            }
        
        goal_stats = {}
        for goal, skills in goals.items():
            if isinstance(skills, list):
                goal_stats[goal] = len(skills)
            else:
                goal_stats[goal] = "invalid_format"
        
        return {
            "status": "ok",
            "message": "Goals loaded successfully",
            "details": {
                "total_goals": len(goals),
                "goals_skills_count": goal_stats
            }
        }
        
    except json.JSONDecodeError as e:
        return {
            "status": "error",
            "message": "Invalid JSON in goals.json",
            "details": {"error": str(e)}
        }
    except Exception as e:
        return {
            "status": "error",
            "message": "Goals check failed",
            "details": {"error": str(e)}
        }

async def _check_model_registry() -> Dict[str, Any]:
    """Check if model registry exists and is consistent."""
    try:
        registry_path = "app/model/model_registry.json"
        if not os.path.exists(registry_path):
            return {
                "status": "warning",
                "message": "Model registry not found (optional)",
                "details": {"path": registry_path}
            }
        
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        # Check if registered models actually exist
        missing_files = []
        if "models" in registry:
            for goal, model_file in registry["models"].items():
                model_path = os.path.join("app/model", model_file)
                if not os.path.exists(model_path):
                    missing_files.append(model_file)
        
        status = "ok" if not missing_files else "warning"
        
        return {
            "status": status,
            "message": "Registry check completed",
            "details": {
                "registered_models": len(registry.get("models", {})),
                "missing_files": missing_files,
                "has_metrics": "metrics" in registry
            }
        }
        
    except Exception as e:
        return {
            "status": "warning",
            "message": "Registry check failed",
            "details": {"error": str(e)}
        }

async def _check_performance() -> Dict[str, Any]:
    """Quick performance test with dummy data."""
    try:
        # This would normally call your scorer function
        test_text = "Python programming data structures algorithms"
        start_time = datetime.now(timezone.utc)

        
        # Simulate scoring (replace with actual scorer call)
        # score_result = await score_resume("test", "Amazon SDE", test_text)
        
        end_time = datetime.now(timezone.utc)

        response_time = (end_time - start_time).total_seconds()
        
        # Check if response time meets SLA (< 1.5s as per requirements)
        status = "ok" if response_time < 1.5 else "warning"
        
        return {
            "status": status,
            "message": f"Performance test completed in {response_time:.3f}s",
            "details": {
                "response_time_seconds": round(response_time, 3),
                "sla_threshold": 1.5,
                "meets_sla": response_time < 1.5
            }
        }
        
    except Exception as e:
        return {
            "status": "warning",
            "message": "Performance test failed",
            "details": {"error": str(e)}
        }

# Version endpoint
@app.get("/version")
async def version():
    """Return version and model metadata."""
    config = app.state.config
    scorer = app.state.scorer

    # Log metadata
    logger.info("📦 /version endpoint called")
    logger.info(f"🔢 Version: {config['version']}")
    logger.info(f"🎯 Supported goals: {len(config['model_goals_supported'])} goals")
    logger.info(f"🧰 Default goal: {config['default_goal_model']}")
    logger.info(f"✅ Loaded models: {list(scorer.models.keys())}")
    logger.info(f"📊 Logging enabled: {config['log_score_details']}")
    logger.info(f"📈 Analytics enabled: {config['analytics']['collect_usage_metrics']}")
    logger.info(f"🚨 Alert on low scores: {config['notification']['alert_on_error']} if < {config['notification']['alert_threshold_score']}")
    return {
        "version": config["version"],
        "minimum_score_to_pass": config["minimum_score_to_pass"],
        "default_goal_model": config["default_goal_model"],
        "model_goals_supported": config["model_goals_supported"],
        "loaded_models": list(scorer.models.keys()),
        "skill_matching": config["skill_matching"],
        "performance": config["performance"],
        "logging": config["logging"],
        "api": config["api"],
        "analytics": config["analytics"],
        "notification": config["notification"]
    }

# Main scoring endpoint
@app.post("/score", response_model=ScoreResponse)
async def score_resume(request: ScoreRequest):
    """Score a resume against a goal and return insights."""
    config = app.state.config

    try:
        # Validate goal
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
            # Force score to 0.0 if no matched skills
            if not result["matched_skills"]:
                result["score"] = 0.0
        except Exception as e:
            logger.exception(f"❌ Error during resume scoring for goal={goal}")
            raise HTTPException(status_code=500, detail="Error occurred while scoring resume")

        # Log scoring details
        if config.get("log_score_details", False):
            logger.info(
                f"Scored resume for student {request.student_id}: "
                f"goal={goal}, score={result['score']:.2f}, "
                f"matched={len(result['matched_skills'])}, "
                f"missing={len(result['missing_skills'])}"
            )

        # Format suggested_learning_path to split '→' into clean bullet steps
        # Sanitize and format the suggested learning path
        # Format path into list and append course as final step
        cleaned_learning_path = []
        for item in result.get("suggested_learning_path", []):
            if isinstance(item, dict) and "path" in item and "course" in item:
                if isinstance(item["path"], str):
                    steps = [step.strip() for step in item["path"].split("→") if step.strip()]
                    # Append course as a final step
                    item["path"] = steps
                cleaned_learning_path.append(item)
            else:
                logger.warning(f"⚠️ Skipping invalid learning path entry: {item}")

        return ScoreResponse(
    score=result["score"],
    matched_skills=result["matched_skills"],
    missing_skills=result["missing_skills"],
    suggested_learning_path=cleaned_learning_path
)


    except HTTPException:
        raise  # re-raise explicitly raised exceptions
    except Exception as e:
        logger.exception("❌ Unexpected error in score endpoint")
        raise HTTPException(status_code=500, detail="Unexpected server error occurred")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    logger.warning(f"📛 Validation error at {request.url.path} - {errors}")

    # Format custom errors
    custom_errors = []
    for err in errors:
        field = ".".join(str(x) for x in err.get("loc", []))
        message = err.get("msg", "Invalid input")
        custom_errors.append(f"{field}: {message}")

    return JSONResponse(
        status_code=422,
        content={
            "error": "Invalid request payload",
            "message": "One or more required fields are missing or malformed.",
            "details": custom_errors
        }
    )
    
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.warning(f"HTTP error at {request.url.path} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP error",
            "message": exc.detail
        }
    )
    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)