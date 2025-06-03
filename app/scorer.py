import os
import json
import joblib
import logging
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeScorer:
    def __init__(self, model_dir: str = "model", data_dir: str = "../data", config_path: str = "config.json"):
        """
        Initialize the Resume Scorer with models, vectorizers, and configuration.
        
        Args:
            model_dir: Directory containing trained models and vectorizers
            data_dir: Directory containing goals.json and training data
            config_path: Path to configuration file
        """
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.config_path = Path(config_path)
        
        # Storage for loaded components
        self.models = {}
        self.vectorizers = {}
        self.goals_data = {}
        self.config = {}
        self.model_registry = {}
        
        # Load all components
        self._load_config()
        self._load_goals_data()
        self._load_model_registry()
        self._load_models_and_vectorizers()
        
        self.skill_synonyms = self._build_skill_synonym_map()
        self.reverse_skill_map = self._build_reverse_skill_map()

        
        logger.info(f"ResumeScorer initialized with {len(self.models)} models")
        
    def _build_skill_synonym_map(self) -> Dict[str, List[str]]:
        """Define synonyms for each canonical skill."""
        return {
            "Java": ["Java SE", "Java 8", "Core Java"],
            "Python": ["Python 3", "Python programming"],
            "C++": ["C plus plus", "CPP"],
            "JavaScript": ["JS", "ECMAScript"],
            "System Design": ["Architecture design", "Software design"],
            "Data Structures": ["DSA", "Data struct"],
            "Algorithms": ["Algo", "Problem solving"],
            "AWS": ["Amazon Web Services", "AWS Cloud"],
            "Git": ["Version control", "GitHub", "GitLab"],
            "SQL": ["Structured Query Language", "MySQL", "PostgreSQL"],
            "REST APIs": ["RESTful services", "API development"],
            "Microservices": ["Service-oriented architecture", "Microservice architecture"],
            "Object Oriented Programming": ["OOP", "OOPS"],
            "Distributed Systems": ["Scalable systems", "Distributed architecture"],
            "Docker": ["Containerization", "Docker containers"],
            "Linux": ["Unix", "Linux shell"]
        }

    def _build_reverse_skill_map(self) -> Dict[str, str]:
        """Flatten synonym map into alias → canonical skill map."""
        reverse_map = {}
        for canonical, aliases in self.skill_synonyms.items():
            reverse_map[canonical.lower()] = canonical
            for alias in aliases:
                reverse_map[alias.lower()] = canonical
        return reverse_map

    
    def _load_config(self):
        """Load configuration from config.json"""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            
            # Validate required config fields
            required_fields = ['minimum_score_to_pass', 'model_goals_supported']
            for field in required_fields:
                if field not in self.config:
                    raise ValueError(f"Missing required config field: {field}")
            
            logger.info("Configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _load_goals_data(self):
        """Load goals and their required skills from goals.json"""
        try:
            goals_file = self.data_dir / "goals.json"
            if not goals_file.exists():
                raise FileNotFoundError(f"Goals file not found: {goals_file}")
            
            with open(goals_file, 'r') as f:
                self.goals_data = json.load(f)
            
            logger.info(f"Loaded {len(self.goals_data)} goals from goals.json")
            
        except Exception as e:
            logger.error(f"Failed to load goals data: {e}")
            raise
    
    def _load_model_registry(self):
        """Load model registry containing model and vectorizer mappings"""
        try:
            registry_file = self.model_dir / "model_registry.json"
            if registry_file.exists():
                with open(registry_file, 'r') as f:
                    self.model_registry = json.load(f)
                logger.info("Model registry loaded successfully")
            else:
                logger.warning("Model registry not found, will attempt to load models directly")
                
        except Exception as e:
            logger.warning(f"Failed to load model registry: {e}")
    
    def _load_models_and_vectorizers(self):
        """Load all trained models and their corresponding vectorizers"""
        try:
            supported_goals = self.config.get('model_goals_supported', [])
            
            for goal in supported_goals:
                model_key = goal.lower().replace(' ', '_')
                
                # Try to load from registry first, then fallback to direct file names
                if self.model_registry and 'models' in self.model_registry:
                    model_file = self.model_registry['models'].get(goal)
                    vectorizer_file = self.model_registry['vectorizers'].get(goal)
                else:
                    model_file = f"{model_key}_model.pkl"
                    vectorizer_file = f"{model_key}_vectorizer.pkl"
                
                if model_file and vectorizer_file:
                    model_path = self.model_dir / model_file
                    vectorizer_path = self.model_dir / vectorizer_file
                    
                    if model_path.exists() and vectorizer_path.exists():
                        try:
                            self.models[goal] = joblib.load(model_path)
                            self.vectorizers[goal] = joblib.load(vectorizer_path)
                            logger.info(f"Loaded model and vectorizer for goal: {goal}")
                        except Exception as e:
                            logger.error(f"Failed to load model/vectorizer for {goal}: {e}")
                    else:
                        logger.warning(f"Model or vectorizer files not found for goal: {goal}")
                        if not model_path.exists():
                            logger.warning(f"Missing model file: {model_path}")
                        if not vectorizer_path.exists():
                            logger.warning(f"Missing vectorizer file: {vectorizer_path}")
            
            if not self.models:
                raise RuntimeError("No models were loaded successfully")
                
        except Exception as e:
            logger.error(f"Failed to load models and vectorizers: {e}")
            raise
    
    def _extract_skills_from_text(self, text: str) -> List[str]:
        """
        Extract canonical skills using synonym-aware keyword matching.
        """
        if not text:
            return []

        text_lower = text.lower()
        found_skills = set()

        for alias_lower, canonical in self.reverse_skill_map.items():
            pattern = r'\b' + re.escape(alias_lower) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.add(canonical)

        return list(found_skills)

    
    def _get_original_skill_name(self, skill_lower: str) -> Optional[str]:
        """Get the original case skill name from lowercase version"""
        for goal_skills in self.goals_data.values():
            for skill in goal_skills:
                if skill.lower() == skill_lower:
                    return skill
        return None
    
    def _generate_learning_path(self, missing_skills: List[str]) -> List[str]:
        """
        Generate learning path suggestions based on missing skills.
        
        Args:
            missing_skills: List of skills that are missing from resume
            
        Returns:
            List of learning suggestions
        """
        learning_suggestions = []
        
        # Hardcoded learning paths for common skills
        skill_learning_map = {
            "java": "Learn Java fundamentals and object-oriented programming",
            "python": "Master Python programming and its libraries",
            "data structures": "Study data structures and algorithms",
            "algorithms": "Practice algorithmic problem solving",
            "system design": "Learn basic system design concepts and patterns",
            "sql": "Complete SQL joins and indexing course",
            "javascript": "Learn JavaScript ES6+ and modern frameworks",
            "react": "Build projects with React and component-based architecture",
            "aws": "Get AWS Cloud Practitioner certification",
            "docker": "Learn containerization with Docker",
            "kubernetes": "Master container orchestration with Kubernetes",
            "machine learning": "Study machine learning algorithms and applications",
            "tensorflow": "Learn deep learning with TensorFlow",
            "scikit-learn": "Practice machine learning with scikit-learn",
            "pandas": "Master data manipulation with pandas",
            "numpy": "Learn numerical computing with NumPy",
            "git": "Master version control with Git",
            "linux": "Learn Linux command line and system administration",
            "networking": "Understand computer networking fundamentals",
            "database": "Study database design and management",
            "api": "Learn RESTful API design and development",
            "microservices": "Understand microservices architecture patterns"
        }
        
        for skill in missing_skills:
            skill_lower = skill.lower()
            if skill_lower in skill_learning_map:
                suggestion = skill_learning_map[skill_lower]
            else:
                # Generic suggestion for unknown skills
                suggestion = f"Learn {skill} through online courses and hands-on projects"
            
            if suggestion not in learning_suggestions:
                learning_suggestions.append(suggestion)
        
        return learning_suggestions[:5]  # Limit to top 5 suggestions
    
    def score_resume(self, goal: str, resume_text: str, student_id: str = None) -> Dict:
        """
        Score a resume against a specific goal.
        
        Args:
            goal: Target goal/position (e.g., "Amazon SDE")
            resume_text: Full resume text content
            student_id: Optional student identifier for logging
            
        Returns:
            Dictionary containing score, matched skills, missing skills, and learning path
        """
        try:
            # Validate inputs
            if not goal or not resume_text:
                raise ValueError("Goal and resume_text are required")
            
            if goal not in self.config.get('model_goals_supported', []):
                raise ValueError(f"Unsupported goal: {goal}. Supported goals: {self.config.get('model_goals_supported', [])}")
            
            if goal not in self.models:
                raise ValueError(f"Model not loaded for goal: {goal}")
            
            # Get model and vectorizer for this goal
            model = self.models[goal]
            vectorizer = self.vectorizers[goal]
            
            # Vectorize the resume text
            resume_vector = vectorizer.transform([resume_text])
            
            # Get prediction probability (score)
            prediction_proba = model.predict_proba(resume_vector)
            score = float(prediction_proba[0][1])  # Probability of positive class
            
            # Extract skills from resume
            resume_skills = self._extract_skills_from_text(resume_text)
            
            # Get required skills for this goal
            required_skills = self.goals_data.get(goal, [])
            
            # Determine matched and missing skills
            matched_skills = [skill for skill in required_skills if skill in resume_skills]
            missing_skills = [skill for skill in required_skills if skill not in resume_skills]
            
            # Generate learning path
            learning_path = self._generate_learning_path(missing_skills)
            
            # Log details if configured
            if self.config.get('log_score_details', False):
                logger.info(f"Scored resume for student {student_id}, goal {goal}: score={score:.3f}")
                logger.info(f"Matched skills: {matched_skills}")
                logger.info(f"Missing skills: {missing_skills}")
            
            # Prepare response
            result = {
                "score": round(score, 3),
                "matched_skills": matched_skills,
                "missing_skills": missing_skills,
                "suggested_learning_path": learning_path
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error scoring resume: {e}")
            raise
    
    def get_supported_goals(self) -> List[str]:
        """Get list of supported goals"""
        return self.config.get('model_goals_supported', [])
    
    def get_config_info(self) -> Dict:
        """Get configuration information"""
        return {
            "version": self.config.get("version", "unknown"),
            "minimum_score_to_pass": self.config.get("minimum_score_to_pass"),
            "supported_goals": self.get_supported_goals(),
            "models_loaded": len(self.models)
        }
    
    def health_check(self) -> Dict:
        """Perform health check on the scorer"""
        try:
            # Check if models are loaded
            models_loaded = len(self.models) > 0
            
            # Check if config is valid
            config_valid = bool(self.config and 'model_goals_supported' in self.config)
            
            # Check if goals data is loaded
            goals_loaded = len(self.goals_data) > 0
            
            status = "ok" if (models_loaded and config_valid and goals_loaded) else "error"
            
            return {
                "status": status,
                "models_loaded": models_loaded,
                "config_valid": config_valid,
                "goals_loaded": goals_loaded,
                "supported_goals_count": len(self.get_supported_goals())
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }