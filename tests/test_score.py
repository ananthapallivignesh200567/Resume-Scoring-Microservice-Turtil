import pytest
import json
import time
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os

# Add the app directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from main import app
from scorer import ResumeScorer

client = TestClient(app)

class TestResumeScorer:
    """Test suite for the Resume Scoring Microservice"""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing"""
        return {
            "version": "1.0.0",
            "minimum_score_to_pass": 0.6,
            "log_score_details": True,
            "model_goals_supported": ["Amazon SDE", "ML Internship"],
            "default_goal_model": "Amazon SDE"
        }
    
    @pytest.fixture
    def sample_goals(self):
        """Sample goals configuration"""
        return {
            "Amazon SDE": ["Java", "Data Structures", "System Design", "SQL", "Python", "REST APIs"],
            "ML Internship": ["Python", "Numpy", "Scikit-learn", "Linear Algebra", "TensorFlow", "Pandas"]
        }
    
    @pytest.fixture
    def high_score_resume(self):
        """Resume that should get a high score for Amazon SDE"""
        return {
            "student_id": "stu_1001",
            "goal": "Amazon SDE",
            "resume_text": "Experienced software engineer with 3 years at tech companies. "
                          "Proficient in Java, Python, Data Structures and Algorithms. "
                          "Built REST APIs and microservices. Strong knowledge of SQL databases "
                          "and system design principles. Completed projects using Spring Boot, "
                          "designed scalable systems, and optimized database queries."
        }
    
    @pytest.fixture
    def low_score_resume(self):
        """Resume that should get a low score for Amazon SDE"""
        return {
            "student_id": "stu_1002",
            "goal": "Amazon SDE",
            "resume_text": "Mechanical engineer with experience in AutoCAD and SolidWorks. "
                          "Designed mechanical components and worked on manufacturing processes. "
                          "Knowledge of material science and thermodynamics. "
                          "Proficient in Excel and PowerPoint for technical documentation."
        }
    
    # Health Check Tests
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    
    def test_version_endpoint(self):
        """Test the version endpoint"""
        response = client.get("/version")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "model_goals_supported" in data
    
    # Score Endpoint Tests - Valid Requests
    def test_score_endpoint_valid_request(self, high_score_resume):
        """Test scoring with a valid high-scoring resume"""
        response = client.post("/score", json=high_score_resume)
        assert response.status_code == 200
        
        data = response.json()
        self._validate_score_response_structure(data)
        
        # High score resume should get a decent score
        assert data["score"] >= 0.5
        assert len(data["matched_skills"]) > 0
    
    def test_score_endpoint_low_score_resume(self, low_score_resume):
        """Test scoring with a low-scoring resume"""
        response = client.post("/score", json=low_score_resume)
        assert response.status_code == 200
        
        data = response.json()
        self._validate_score_response_structure(data)
        
        # Low score resume should get a lower score
        assert data["score"] <= 0.7
        assert len(data["missing_skills"]) > 0
    
    def test_score_response_time(self, high_score_resume):
        """Test that response time is under 1.5 seconds"""
        start_time = time.time()
        response = client.post("/score", json=high_score_resume)
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 1.5
    
    def test_ml_internship_goal(self):
        """Test scoring for ML Internship goal"""
        ml_resume = {
            "student_id": "stu_2001",
            "goal": "ML Internship",
            "resume_text": "Data science student with experience in Python, NumPy, Pandas, "
                          "and scikit-learn. Built machine learning models for classification "
                          "and regression. Strong background in linear algebra and statistics. "
                          "Completed projects using TensorFlow and Keras for deep learning."
        }
        
        response = client.post("/score", json=ml_resume)
        assert response.status_code == 200
        
        data = response.json()
        self._validate_score_response_structure(data)
        assert data["score"] >= 0.6
    
    # Score Endpoint Tests - Invalid Requests
    def test_missing_required_fields(self):
        """Test request with missing required fields"""
        # Missing student_id
        incomplete_request = {
            "goal": "Amazon SDE",
            "resume_text": "Some resume text"
        }
        response = client.post("/score", json=incomplete_request)
        assert response.status_code == 422
        
        # Missing goal
        incomplete_request = {
            "student_id": "stu_001",
            "resume_text": "Some resume text"
        }
        response = client.post("/score", json=incomplete_request)
        assert response.status_code == 422
        
        # Missing resume_text
        incomplete_request = {
            "student_id": "stu_001",
            "goal": "Amazon SDE"
        }
        response = client.post("/score", json=incomplete_request)
        assert response.status_code == 422
    
    def test_empty_resume_text(self):
        """Test request with empty resume text"""
        empty_resume = {
            "student_id": "stu_empty",
            "goal": "Amazon SDE",
            "resume_text": ""
        }
        response = client.post("/score", json=empty_resume)
        assert response.status_code == 200
        
        data = response.json()
        self._validate_score_response_structure(data)
        assert data["score"] == 0.0
        assert len(data["matched_skills"]) == 0
        assert len(data["missing_skills"]) > 0
    
    def test_unknown_goal(self):
        """Test request with unsupported goal"""
        unknown_goal_request = {
            "student_id": "stu_unknown",
            "goal": "Unsupported Goal",
            "resume_text": "Some resume content"
        }
        response = client.post("/score", json=unknown_goal_request)
        assert response.status_code == 400
        
        error_data = response.json()
        assert "error" in error_data
        assert "not supported" in error_data["error"].lower()
    
    def test_malformed_json(self):
        """Test request with malformed JSON"""
        response = client.post("/score", 
                             data="{'invalid': 'json'}", 
                             headers={"Content-Type": "application/json"})
        assert response.status_code == 422
    
    def test_invalid_data_types(self):
        """Test request with invalid data types"""
        invalid_request = {
            "student_id": 123,  # Should be string
            "goal": "Amazon SDE",
            "resume_text": "Some text"
        }
        response = client.post("/score", json=invalid_request)
        assert response.status_code == 422
    
    # Edge Cases
    def test_very_long_resume(self):
        """Test with very long resume text"""
        long_text = "Java Python SQL " * 1000  # Very long repetitive text
        long_resume = {
            "student_id": "stu_long",
            "goal": "Amazon SDE",
            "resume_text": long_text
        }
        response = client.post("/score", json=long_resume)
        assert response.status_code == 200
        
        data = response.json()
        self._validate_score_response_structure(data)
    
    def test_special_characters_in_resume(self):
        """Test resume with special characters and unicode"""
        special_resume = {
            "student_id": "stu_special",
            "goal": "Amazon SDE",
            "resume_text": "Software engineer with expertise in Java, Python, and SQL. "
                          "Experience with APIs & microservices. Résumé includes work at "
                          "companies like Über and Café Corp. Skills: C++, JavaScript, "
                          "databases (MySQL, PostgreSQL), cloud computing ☁️"
        }
        response = client.post("/score", json=special_resume)
        assert response.status_code == 200
        
        data = response.json()
        self._validate_score_response_structure(data)
    
    def test_case_insensitive_skill_matching(self):
        """Test that skill matching is case insensitive"""
        mixed_case_resume = {
            "student_id": "stu_case",
            "goal": "Amazon SDE",
            "resume_text": "Experienced in JAVA, python, data structures, and sql databases"
        }
        response = client.post("/score", json=mixed_case_resume)
        assert response.status_code == 200
        
        data = response.json()
        self._validate_score_response_structure(data)
        assert len(data["matched_skills"]) > 0
    
    # Skill Analysis Tests
    def test_matched_skills_accuracy(self):
        """Test accuracy of matched skills detection"""
        skill_resume = {
            "student_id": "stu_skills",
            "goal": "Amazon SDE",
            "resume_text": "Proficient in Java, Python, and SQL. Strong understanding of "
                          "data structures and algorithms. Experience with REST APIs."
        }
        response = client.post("/score", json=skill_resume)
        assert response.status_code == 200
        
        data = response.json()
        matched_skills_lower = [skill.lower() for skill in data["matched_skills"]]
        
        # Should match these skills (case insensitive)
        expected_matches = ["java", "python", "sql", "data structures", "rest apis"]
        for skill in expected_matches:
            assert any(skill in matched.lower() for matched in data["matched_skills"]), \
                   f"Expected skill '{skill}' not found in matched_skills"
    
    def test_missing_skills_detection(self):
        """Test detection of missing skills"""
        partial_resume = {
            "student_id": "stu_partial",
            "goal": "Amazon SDE",
            "resume_text": "Basic programming knowledge in Java. Some database experience."
        }
        response = client.post("/score", json=partial_resume)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["missing_skills"]) > 0
        
        # Should be missing system design as it's not mentioned
        missing_skills_lower = [skill.lower() for skill in data["missing_skills"]]
        assert any("system design" in skill.lower() for skill in data["missing_skills"])
    
    def test_learning_path_generation(self):
        """Test generation of learning path suggestions"""
        beginner_resume = {
            "student_id": "stu_beginner",
            "goal": "Amazon SDE",
            "resume_text": "Computer science student with basic programming knowledge"
        }
        response = client.post("/score", json=beginner_resume)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["suggested_learning_path"]) > 0
        assert all(isinstance(suggestion, str) for suggestion in data["suggested_learning_path"])
    
    # Performance and Stress Tests
    def test_concurrent_requests(self):
        """Test handling multiple concurrent requests"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request():
            test_resume = {
                "student_id": f"stu_concurrent_{threading.current_thread().ident}",
                "goal": "Amazon SDE",
                "resume_text": "Java Python SQL experience"
            }
            response = client.post("/score", json=test_resume)
            results.put(response.status_code)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check all requests succeeded
        while not results.empty():
            assert results.get() == 200
    
    # Configuration Tests
    def test_unsupported_goal_based_on_config(self):
        """Test that only configured goals are supported"""
        # This would require mocking the config, but demonstrates the test concept
        unsupported_goal = {
            "student_id": "stu_config",
            "goal": "Unsupported Career Path",
            "resume_text": "Some experience"
        }
        response = client.post("/score", json=unsupported_goal)
        assert response.status_code == 400
    
    # Helper Methods
    def _validate_score_response_structure(self, data):
        """Validate the structure of a score response"""
        required_fields = ["score", "matched_skills", "missing_skills", "suggested_learning_path"]
        
        for field in required_fields:
            assert field in data, f"Required field '{field}' missing from response"
        
        # Validate data types
        assert isinstance(data["score"], (int, float)), "Score should be numeric"
        assert 0.0 <= data["score"] <= 1.0, "Score should be between 0.0 and 1.0"
        
        assert isinstance(data["matched_skills"], list), "matched_skills should be a list"
        assert isinstance(data["missing_skills"], list), "missing_skills should be a list"
        assert isinstance(data["suggested_learning_path"], list), "suggested_learning_path should be a list"
        
        # Validate list contents
        for skill in data["matched_skills"]:
            assert isinstance(skill, str), "Each matched skill should be a string"
        
        for skill in data["missing_skills"]:
            assert isinstance(skill, str), "Each missing skill should be a string"
        
        for suggestion in data["suggested_learning_path"]:
            assert isinstance(suggestion, str), "Each learning path suggestion should be a string"

# Integration Tests
class TestIntegration:
    """Integration tests for the complete workflow"""
    
    def test_end_to_end_high_score_workflow(self):
        """Test complete workflow for a high-scoring resume"""
        high_score_request = {
            "student_id": "integration_high",
            "goal": "Amazon SDE",
            "resume_text": "Senior Software Engineer with 5+ years experience. Expert in Java, "
                          "Python, and system design. Built scalable REST APIs and microservices. "
                          "Strong SQL and database optimization skills. Led teams in agile development."
        }
        
        response = client.post("/score", json=high_score_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["score"] >= 0.7
        assert len(data["matched_skills"]) >= 3
        assert len(data["suggested_learning_path"]) >= 0  # May be empty for high scores
    
    def test_end_to_end_low_score_workflow(self):
        """Test complete workflow for a low-scoring resume"""
        low_score_request = {
            "student_id": "integration_low",
            "goal": "ML Internship",
            "resume_text": "Accounting major with experience in Excel and financial modeling. "
                          "Basic mathematics background. Completed courses in statistics."
        }
        
        response = client.post("/score", json=low_score_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["score"] <= 0.4
        assert len(data["missing_skills"]) >= 3
        assert len(data["suggested_learning_path"]) >= 1

# Pytest Configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v"])