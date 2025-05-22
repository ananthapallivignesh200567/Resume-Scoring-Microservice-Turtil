import joblib

# Load the vectorizer
vectorizer = joblib.load('app/model/tfidf_vectorizer.pkl')

# Test with sample resume
test_resume = "Python developer with machine learning experience"
features = vectorizer.transform([test_resume])

print(f"Feature shape: {features.shape}")
print(f"Non-zero features: {features.nnz}")
print("✅ Vectorizer working correctly!")