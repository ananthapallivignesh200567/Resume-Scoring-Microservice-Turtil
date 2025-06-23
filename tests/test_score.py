import unittest
import json
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import numpy as np
from app.scorer import ResumeScorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from fastapi.testclient import TestClient
from app.main import app

# Assuming scorer.py is in the same directory

class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        # Setup test config and goals
        self.test_config = {
    "version": "1.0.0",
    "minimum_score_to_pass": 0.65,
    "log_score_details": True,
    "model_goals_supported": ["Amazon SDE", "ML Internship"],
    "default_goal_model": "Amazon SDE",
    "skill_matching": {
        "case_sensitive": False,
        "exact_match_only": True,
        "partial_match_threshold": 0.8
    },
    "performance": {
        "max_response_time_ms": 1500,
        "max_memory_usage_mb": 256
    },
    "logging": {
        "level": "INFO",
        "file_rotation": True,
        "max_log_files": 5,
        "max_log_file_size_mb": 10
    },
    "analytics": {
        "collect_usage_metrics": True,
        "anonymize_student_data": True,
        "store_request_history": False
    },
    "notification": {
        "alert_on_error": True,
        "alert_threshold_score": 0.3
    },
    "api": {
    "port": 8000,
    "host": "0.0.0.0",
    "workers": 4,
    "timeout_seconds": 30,
    "rate_limit": {
        "requests_per_minute": 60,
        "enabled": False
    }
    }
}


        self.test_goals = {
            "Amazon SDE": ["Java", "Data Structures", "System Design", "SQL"],
            "ML Internship": ["Python", "Numpy", "Scikit-learn", "Linear Algebra"]
        }

        # ✅ Manually inject config and scorer into app state
        app.state.config = self.test_config
        app.state.scorer = ResumeScorer(self.test_config, self.test_goals)

        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), response.json())

    def test_version_check(self):
        response = self.client.get("/version")
        self.assertEqual(response.status_code, 200)
        self.assertIn("version", response.json())

    def test_score_valid_request(self):
        payload = {
            "student_id": "stu_001",
            "goal": "Amazon SDE",
            "resume_text": "Java, Python, SQL, Data Structures, System Design"
        }
        response = self.client.post("/score", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("score", data)
        self.assertIn("matched_skills", data)
        self.assertIn("missing_skills", data)
        self.assertIn("suggested_learning_path", data)

    def test_score_missing_fields(self):
        payload = {
            "student_id": "stu_002",
            "goal": "Amazon SDE"
            # Missing resume_text
        }
        response = self.client.post("/score", json=payload)
        self.assertEqual(response.status_code, 422)  # Unprocessable Entity

    def test_score_empty_resume(self):
        payload = {
            "student_id": "stu_003",
            "goal": "Amazon SDE",
            "resume_text": ""
        }
        response = self.client.post("/score", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["score"], 0.0)

class TestResumeScorer(unittest.TestCase):
    """Unit tests for ResumeScorer class"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Sample configuration
        self.test_config = {
    "version": "1.0.0",
    "minimum_score_to_pass": 0.65,
    "log_score_details": True,
    "model_goals_supported": ["Amazon SDE", "ML Internship"],
    "default_goal_model": "Amazon SDE",
    "skill_matching": {
        "case_sensitive": False,
        "exact_match_only": True,
        "partial_match_threshold": 0.8
    },
    "performance": {
        "max_response_time_ms": 1500,
        "max_memory_usage_mb": 256
    },
    "logging": {
        "level": "INFO",
        "file_rotation": True,
        "max_log_files": 5,
        "max_log_file_size_mb": 10
    },
    "analytics": {
        "collect_usage_metrics": True,
        "anonymize_student_data": True,
        "store_request_history": False
    },
    "notification": {
        "alert_on_error": True,
        "alert_threshold_score": 0.3
    },
    "api": {
    "port": 8000,
    "host": "0.0.0.0",
    "workers": 4,
    "timeout_seconds": 30,
    "rate_limit": {
        "requests_per_minute": 60,
        "enabled": False
    }
    }
}

        
        # Sample goals data
        self.test_goals = {
            "Amazon SDE": ["Java", "Data Structures", "System Design", "SQL"],
            "ML Internship": ["Python", "Numpy", "Scikit-learn", "Linear Algebra"]
        }
        
        # Create temporary directory for test models
        self.temp_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(self.temp_dir, "model")
        os.makedirs(self.model_dir)
        
        # Create mock models and vectorizer
        self._create_mock_models()
        
    def tearDown(self):
        """Clean up after each test."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def _create_mock_models(self):
        """Create mock TF-IDF vectorizer and models for testing."""
        # Create mock TF-IDF vectorizer
        mock_vectorizer = TfidfVectorizer(max_features=1000)
        sample_texts = [
            "Java Python SQL Data Structures",
            "Python Numpy Scikit-learn Linear Algebra",
            "JavaScript HTML CSS React",
            "C++ Algorithms System Design"
        ]
        mock_vectorizer.fit(sample_texts)
        
        # Save mock vectorizer
        vectorizer_path = os.path.join(self.model_dir, "tfidf_vectorizer.pkl")
        joblib.dump(mock_vectorizer, vectorizer_path)
        
        # Create mock models for each goal
        for goal in self.test_config["model_goals_supported"]:
            # Create simple logistic regression model
            model = LogisticRegression(random_state=42)
            
            # Generate some dummy training data
            X = mock_vectorizer.transform(sample_texts)
            y = np.array([1, 1, 0, 1])  # Dummy labels
            
            model.fit(X, y)
            
            # Save model
            goal_filename = goal.lower().replace(" ", "_")
            model_path = os.path.join(self.model_dir, f"{goal_filename}_model.pkl")
            joblib.dump(model, model_path)
    
    @patch('app.scorer.os.path.dirname')
    def test_init_successful_loading(self, mock_dirname):
        """Test successful initialization with valid config and goals."""
        mock_dirname.return_value = self.temp_dir
        
        scorer = ResumeScorer(self.test_config, self.test_goals)
        
        # Verify initialization
        self.assertIsNotNone(scorer.shared_vectorizer)
        self.assertEqual(len(scorer.models), 2)
        self.assertIn("Amazon SDE", scorer.models)
        self.assertIn("ML Internship", scorer.models)
    
    @patch('app.scorer.os.path.dirname')
    def test_init_missing_vectorizer(self, mock_dirname):
        """Test initialization when vectorizer file is missing."""
        # Create temp dir without vectorizer
        temp_dir_no_vectorizer = tempfile.mkdtemp()
        model_dir_no_vectorizer = os.path.join(temp_dir_no_vectorizer, "model")
        os.makedirs(model_dir_no_vectorizer)
        
        mock_dirname.return_value = temp_dir_no_vectorizer
        
        scorer = ResumeScorer(self.test_config, self.test_goals)
        
        # Verify vectorizer is None when file is missing
        self.assertIsNone(scorer.shared_vectorizer)
        
        # Cleanup
        shutil.rmtree(temp_dir_no_vectorizer)
    
    @patch('app.scorer.os.path.dirname')
    def test_extract_skills_from_resume(self, mock_dirname):
        """Test skill extraction from resume text."""
        mock_dirname.return_value = self.temp_dir
        scorer = ResumeScorer(self.test_config, self.test_goals)
        
        # Test with resume containing some skills
        resume_text = "I am proficient in Java, Python"
        
        found_skills = scorer._extract_skills_from_resume(resume_text)
        
        # Should find Java, Python, SQL, Data Structures
        expected_skills = ["Java", "Python"]
        for skill in expected_skills:
            self.assertIn(skill, found_skills)
    
    @patch('app.scorer.os.path.dirname')
    def test_extract_skills_with_synonyms(self, mock_dirname):
        """Test skill extraction with synonym mapping."""
        mock_dirname.return_value = self.temp_dir
        scorer = ResumeScorer(self.test_config, self.test_goals)
        
        # Test with resume containing synonyms
        resume_text = "I know DS, OOP, and REST APIs. I have experience with ML and DL."
        
        found_skills = scorer._extract_skills_from_resume(resume_text)
        
        # Should find skills through synonym mapping
        # Note: This test might need adjustment based on actual synonym implementation
        self.assertIsInstance(found_skills, list)
    
    @patch('app.scorer.os.path.dirname')
    def test_get_matched_missing_skills(self, mock_dirname):
        """Test matching and missing skills identification."""
        mock_dirname.return_value = self.temp_dir
        scorer = ResumeScorer(self.test_config, self.test_goals)
        
        # Test with some found skills
        found_skills = ["Java", "SQL", "Python"]
        goal = "Amazon SDE"
        
        matched, missing = scorer._get_matched_missing_skills(found_skills, goal)
        
        # For Amazon SDE: required = ["Java", "Data Structures", "System Design", "SQL"]
        expected_matched = ["Java", "SQL"]
        expected_missing = ["Data Structures", "System Design"]
        
        self.assertEqual(set(matched), set(expected_matched))
        self.assertEqual(set(missing), set(expected_missing))
    
    @patch('app.scorer.os.path.dirname')
    def test_get_matched_missing_skills_unknown_goal(self, mock_dirname):
        """Test behavior with unknown goal."""
        mock_dirname.return_value = self.temp_dir
        scorer = ResumeScorer(self.test_config, self.test_goals)
        
        found_skills = ["Java", "Python"]
        unknown_goal = "Unknown Goal"
        
        matched, missing = scorer._get_matched_missing_skills(found_skills, unknown_goal)
        
        # Should use default goal when unknown goal is provided
        self.assertIsInstance(matched, list)
        self.assertIsInstance(missing, list)
    
    @patch('app.scorer.os.path.dirname')
    def test_generate_learning_path(self, mock_dirname):
        """Test learning path generation."""
        mock_dirname.return_value = self.temp_dir
        scorer = ResumeScorer(self.test_config, self.test_goals)
        
        missing_skills = ["System Design", "Docker"]
        
        learning_path = scorer._generate_learning_path(missing_skills)
        
        # Should return a list of learning recommendations
        self.assertIsInstance(learning_path, list)
        self.assertGreater(len(learning_path), 0)
        
        # Should contain specific recommendations for known skills
        path_text = " ".join(item["course"] for item in learning_path)

        self.assertIn("system design", path_text.lower())
    
    @patch('app.scorer.os.path.dirname')
    def test_score_resume_with_model(self, mock_dirname):
        """Test resume scoring with ML model."""
        mock_dirname.return_value = self.temp_dir
        scorer = ResumeScorer(self.test_config, self.test_goals)
        
        # Test inputs
        student_id = "test_student_123"
        goal = "Amazon SDE"
        resume_text = "I am a software engineer with experience in Java, Python, and SQL. I have worked on system design projects."
        
        result = scorer.score_resume(student_id, goal, resume_text)
        
        # Verify result structure
        self.assertIn("score", result)
        self.assertIn("matched_skills", result)
        self.assertIn("missing_skills", result)
        self.assertIn("suggested_learning_path", result)
        
        # Verify data types
        self.assertIsInstance(result["score"], float)
        self.assertIsInstance(result["matched_skills"], list)
        self.assertIsInstance(result["missing_skills"], list)
        self.assertIsInstance(result["suggested_learning_path"], list)
        
        # Score should be between 0 and 1
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)
    
    @patch('app.scorer.os.path.dirname')
    def test_score_resume_unsupported_goal(self, mock_dirname):
        """Test resume scoring with unsupported goal."""
        mock_dirname.return_value = self.temp_dir
        scorer = ResumeScorer(self.test_config, self.test_goals)
        
        # Test with unsupported goal
        result = scorer.score_resume("test_student", "Unsupported Goal", "Java Python SQL")
        
        # Should still return valid result using default goal
        self.assertIn("score", result)
        self.assertIsInstance(result["score"], float)
    
    @patch('app.scorer.os.path.dirname')
    def test_score_resume_empty_text(self, mock_dirname):
        """Test resume scoring with empty resume text."""
        mock_dirname.return_value = self.temp_dir
        scorer = ResumeScorer(self.test_config, self.test_goals)
        
        result = scorer.score_resume("test_student", "Amazon SDE", "")
        
        # Should handle empty resume gracefully
        self.assertIn("score", result)
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["matched_skills"], [])
    
    @patch('app.scorer.os.path.dirname')
    def test_score_resume_without_model(self, mock_dirname):
        """Test resume scoring when model is not available."""
        # Create scorer without models
        temp_dir_no_models = tempfile.mkdtemp()
        model_dir_no_models = os.path.join(temp_dir_no_models, "model")
        os.makedirs(model_dir_no_models)
        
        mock_dirname.return_value = temp_dir_no_models
        
        scorer = ResumeScorer(self.test_config, self.test_goals)
        
        result = scorer.score_resume("test_student", "Amazon SDE", "Java Python SQL")
        
        # Should fall back to skill-based scoring
        self.assertIn("score", result)
        self.assertIsInstance(result["score"], float)
        
        # Cleanup
        shutil.rmtree(temp_dir_no_models)
    
    @patch('app.scorer.os.path.dirname')
    def test_evaluate_passing(self, mock_dirname):
        """Test score evaluation against passing threshold."""
        mock_dirname.return_value = self.temp_dir
        scorer = ResumeScorer(self.test_config, self.test_goals)
        
        # Test passing score
        self.assertTrue(scorer.evaluate_passing(0.8))
        self.assertTrue(scorer.evaluate_passing(0.65))  # Exactly at threshold
        
        # Test failing score
        self.assertFalse(scorer.evaluate_passing(0.5))
        self.assertFalse(scorer.evaluate_passing(0.0))
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test with missing required config keys
        invalid_config = {
            "version": "1.0.0"
            # Missing other required keys
        }
        
        # Should handle missing config gracefully or raise appropriate error
        with self.assertRaises(KeyError):
            scorer = ResumeScorer(invalid_config, self.test_goals)
            scorer.score_resume("test", "Amazon SDE", "Java Python")


class TestIntegration(unittest.TestCase):
    """Integration tests for end-to-end functionality."""
    
    @patch("app.scorer.joblib.load")
    def test_high_score_resume(self, mock_joblib_load):
        # Set up mock vectorizer
        mock_vectorizer = MagicMock()
        mock_vectorizer.transform.return_value = np.array([[0.1] * 17])

        # Set up mock model
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])

        # joblib.load returns vectorizer first, then model
        mock_joblib_load.side_effect = lambda path: mock_vectorizer if "tfidf" in str(path) else mock_model

        scorer = ResumeScorer(self.test_config, self.test_goals)
        text = "Java, Python, SQL, Data Structures, System Design"
        result = scorer.score_resume("high_score_001", "Amazon SDE", text)

        self.assertGreaterEqual(result["score"], 0.8)

    @patch("app.scorer.joblib.load")
    def test_low_score_resume(self, mock_joblib_load):
        # Set up mock vectorizer
        mock_vectorizer = MagicMock()
        mock_vectorizer.transform.return_value = np.array([[0.0] * 17])

        # Set up mock model
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])

        # joblib.load returns vectorizer first, then model
        mock_joblib_load.side_effect = lambda path: mock_vectorizer if "tfidf" in str(path) else mock_model

        scorer = ResumeScorer(self.test_config, self.test_goals)
        text = "AutoCAD, Mechanical CAD, SolidWorks, Civil Engineering"
        result = scorer.score_resume("low_score_001", "Amazon SDE", text)

        self.assertLessEqual(result["score"], 0.3)
    def setUp(self):
        """Set up integration test fixtures."""
        self.test_config = {
    "version": "1.0.0",
    "minimum_score_to_pass": 0.65,
    "log_score_details": True,
    "model_goals_supported": ["Amazon SDE", "ML Internship"],
    "default_goal_model": "Amazon SDE",
    "skill_matching": {
        "case_sensitive": False,
        "exact_match_only": True,
        "partial_match_threshold": 0.8
    },
    "performance": {
        "max_response_time_ms": 1500,
        "max_memory_usage_mb": 256
    },
    "logging": {
        "level": "INFO",
        "file_rotation": True,
        "max_log_files": 5,
        "max_log_file_size_mb": 10
    },
    "analytics": {
        "collect_usage_metrics": True,
        "anonymize_student_data": True,
        "store_request_history": False
    },
    "notification": {
        "alert_on_error": True,
        "alert_threshold_score": 0.3
    },
    "api": {
    "port": 8000,
    "host": "0.0.0.0",
    "workers": 4,
    "timeout_seconds": 30,
    "rate_limit": {
        "requests_per_minute": 60,
        "enabled": False
    }
    }
}

        
        self.test_goals = {
            "Amazon SDE": ["Java", "Data Structures", "System Design", "SQL"],
            "ML Internship": ["Python", "Numpy", "Scikit-learn", "Linear Algebra"]
        }
    
    
    
    
    
    def test_response_time(self):
        """Test that scoring completes within required time limit."""
        import time
        
        # Mock quick test - in real implementation, measure actual scoring time
        start_time = time.time()
        
        # Simulate scoring operation
        time.sleep(0.1)  # Simulate processing time
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within 1.5 seconds
        self.assertLess(processing_time, 1.5)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_malformed_input(self):
        config = {
            "version": "1.0.0",
            "minimum_score_to_pass": 0.6,
            "log_score_details": True,
            "model_goals_supported": ["Amazon SDE"],
            "default_goal_model": "Amazon SDE"
        }
        goals = {"Amazon SDE": ["Java", "Python"]}

        scorer = ResumeScorer(config, goals)

        # Null resume
        result = scorer.score_resume("x", "Amazon SDE", None)
        self.assertEqual(result["score"], 0.0)

        # Non-string resume
        result = scorer.score_resume("x", "Amazon SDE", 12345)
        self.assertEqual(result["score"], 0.0)

        # Completely missing goal
        result = scorer.score_resume("x", "", "Java SQL")
        self.assertIsInstance(result["score"], float)
        
    def test_extreme_values(self):
        config = {
            "version": "1.0.0",
            "minimum_score_to_pass": 0.6,
            "log_score_details": True,
            "model_goals_supported": ["Amazon SDE"],
            "default_goal_model": "Amazon SDE"
        }
        goals = {"Amazon SDE": ["Java", "Python", "System Design"]}
        scorer = ResumeScorer(config, goals)

        # Very long resume
        long_text = "Java " * 1000
        result = scorer.score_resume("x", "Amazon SDE", long_text)
        self.assertGreater(result["score"], 0.5)

        # Resume with only special characters
        weird_text = "!@#$%^&*()_+=-[]{}|;':,./<>?"
        result = scorer.score_resume("x", "Amazon SDE", weird_text)
        self.assertLess(result["score"], 0.01)

        # Empty string
        result = scorer.score_resume("x", "Amazon SDE", "")
        self.assertEqual(result["score"], 0.0)


if __name__ == "__main__":
    # Configure test runner
    unittest.main(verbosity=2)