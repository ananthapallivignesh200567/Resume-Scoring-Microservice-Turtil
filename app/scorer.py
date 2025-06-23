import os
import re
import json
import logging
import joblib
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from .synonym_map import synonym_map
from .skill_recommendations import skill_recommendations

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("resume-scorer")

class ResumeScorer:
    """
    Main class for scoring resumes against various goals using ML models
    and skill-based analysis.
    """
    
    def __init__(self, config: Dict[str, Any], goals: Dict[str, List[str]]):
        """
        Initialize the ResumeScorer with configuration and goals data.
        
        Args:
            config: Configuration dictionary loaded from config.json
            goals: Dictionary of goals and their required skills
        """
        self.goals_data = goals
        
        self.config = config
        self.goals = goals
        self.models = {}
        self.shared_vectorizer = None
        
        # Load models and vectorizers for each supported goal
        self._load_models()
        
        logger.info(f"ResumeScorer initialized with {len(self.models)} models")
    
    def _load_models(self) -> None:
        """Load shared TF-IDF vectorizer and goal-specific models."""
        model_dir = os.path.join(os.path.dirname(__file__), "model")

        # Load shared vectorizer
        try:
            vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
            self.shared_vectorizer = joblib.load(vectorizer_path)
            logger.info(f"✅ Shared TF-IDF vectorizer loaded with {len(self.shared_vectorizer.vocabulary_)} features")
        except Exception as e:
            logger.error(f"❌ Failed to load shared TF-IDF vectorizer: {str(e)}")
            self.shared_vectorizer = None

        # Load goal-specific models
        for goal in self.config["model_goals_supported"]:
            try:
                goal_filename = goal.lower().replace(" ", "_")
                model_path = os.path.join(model_dir, f"{goal_filename}_model.pkl")
                self.models[goal] = joblib.load(model_path)
                logger.info(f"✅ Loaded model for goal: {goal}")
            except Exception as e:
                logger.error(f"❌ Failed to load model for goal {goal}: {str(e)}")

    
    def _extract_skills_from_resume(self, resume_text: str) -> List[str]:
        if isinstance(resume_text, list):
            resume_text = ", ".join(resume_text)
        
        all_skills = set()
        for skills in self.goals.values():
            all_skills.update(skills)

        resume_text_lower = resume_text.lower()
        normalized_resume = resume_text_lower

        # Fix: Replace all synonyms with canonical skills
        for canonical, synonyms in synonym_map.items():
            for synonym in synonyms:
                pattern = r'\b' + re.escape(synonym.lower()) + r'\b'
                normalized_resume = re.sub(pattern, canonical.lower(), normalized_resume)

        found_skills = []
        for skill in all_skills:
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, normalized_resume):
                found_skills.append(skill)

        return found_skills

    
    def _get_matched_missing_skills(self, found_skills: List[str], goal: str) -> tuple:
        """
        Compare found skills with required skills for the goal.
        
        Args:
            found_skills: List of skills found in the resume
            goal: The target goal
            
        Returns:
            Tuple of (matched_skills, missing_skills)
        """
        if goal not in self.goals:
            logger.warning(f"Goal '{goal}' not found in goals data, using default")
            goal = self.config["default_goal_model"]
            
        required_skills = self.goals.get(goal, [])
        
        # Find matched and missing skills
        matched_skills = [skill for skill in found_skills if skill in required_skills]
        missing_skills = [skill for skill in required_skills if skill not in found_skills]
        
        return matched_skills, missing_skills
    
    def _generate_learning_path(self, missing_skills: List[str]) -> List[str]:
        """
        Generate a personalized learning path based on missing skills.
        
        Args:
            missing_skills: List of skills missing from the resume
            
        Returns:
            List of suggested learning activities
        """
        learning_path = []
        

  
        # Generate recommendations for each missing skill
        for skill in missing_skills:
            if skill in skill_recommendations:
                learning_path.append(skill_recommendations[skill])
            else:
                # Generic recommendation if specific one isn't available
                learning_path.append({
        "path": [f"Develop proficiency in {skill} through online courses and projects"],
        "course": "General Skill Development – Coursera/Udemy/YouTube"
    })
                
        # Add general advice if there are missing skills
        if missing_skills:
            learning_path.append({
    "path": "Create portfolio projects that showcase your skills in these areas",
    "course": "Not applicable"
})

            
        return learning_path
    
    def score_resume(self, student_id: str, goal: str, resume_text: str) -> Dict[str, Any]:
        """
        Score a resume against a goal and provide detailed insights.
        
        Args:
            student_id: Unique student identifier
            goal: Target position or domain
            resume_text: Full plain-text resume content
            
        Returns:
            Dictionary with score, matched_skills, missing_skills, and suggested_learning_path
        """
        if not isinstance(resume_text, str):
            resume_text = ""
        resume_text = resume_text.strip()
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", resume_text)
        if not cleaned.strip():
            return {
                "score": 0.0,
                    "matched_skills": [],
                    "missing_skills": [],
                    "suggested_learning_path": []
                }
        if not isinstance(resume_text, str):
            resume_text = ""

        resume_text = resume_text.strip()
        if not resume_text:
            goal = goal if goal in self.goals_data else self.default_goal
            return {
                "score": 0.0,
                "matched_skills": [],
                "missing_skills": self.goals_data.get(goal, []),
                "suggested_learning_path": self._generate_learning_path(self.goals_data.get(goal, []))
            }

        # Check if the goal is supported, otherwise use default
        if goal not in self.config["model_goals_supported"]:
            logger.warning(f"Goal '{goal}' not supported, using default: {self.config['default_goal_model']}")
            goal = self.config["default_goal_model"]
            
        # Extract skills from resume
        found_skills = self._extract_skills_from_resume(resume_text)
        
        # Get matched and missing skills
        matched_skills, missing_skills = self._get_matched_missing_skills(found_skills, goal)
        
        # Generate ML model score if model exists for this goal
        if goal in self.models and self.shared_vectorizer:
            X = self.shared_vectorizer.transform([resume_text])

            
            # Get probability score from the model (positive class probability)
            score = float(self.models[goal].predict_proba(X)[0][1])
        else:
            # Fallback scoring based on skill match percentage if model unavailable
            logger.warning(f"No model available for goal '{goal}', using skill-based scoring")
            required_skills = self.goals.get(goal, [])
            score = len(matched_skills) / max(len(required_skills), 1) if required_skills else 0.0
            
        # Generate learning path
        learning_path = self._generate_learning_path(missing_skills)
        
        # Prepare and return the result
        result = {
            "score": score,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "suggested_learning_path": learning_path
        }
        
        return result
    
    def evaluate_passing(self, score: float) -> bool:
        """
        Determine if a score meets the minimum passing threshold.
        
        Args:
            score: The resume match score
            
passes the thresholdpip         Returns:
            Boolean indicating if the score install joblib
        """
        return score >= self.config["minimum_score_to_pass"]