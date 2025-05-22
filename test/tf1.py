import joblib
import numpy as np

def quick_feature_check(text, model_path="app/model/tfidf_vectorizer.pkl"):
    """
    Quick function to see non-zero features for any text.
    """
    # Load vectorizer
    vectorizer = joblib.load(model_path)
    
    # Transform text
    tfidf_matrix = vectorizer.transform([text])
    
    # Get feature names and non-zero indices
    feature_names = vectorizer.get_feature_names_out()
    non_zero_indices = tfidf_matrix.nonzero()[1]
    non_zero_scores = tfidf_matrix.data
    
    # Create sorted list of features with scores
    features_with_scores = []
    for idx, score in zip(non_zero_indices, non_zero_scores):
        features_with_scores.append((feature_names[idx], score))
    
    # Sort by score (highest first)
    features_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Non-zero features ({len(features_with_scores)} total):")
    print("-" * 40)
    for feature, score in features_with_scores:
        print(f"{feature:<25} | {score:.4f}")
    
    return features_with_scores

# Example usage:
if __name__ == "__main__":
    sample_resume = """
    Senior Data Scientist with expertise in machine learning and Python programming.
    Experience with TensorFlow, scikit-learn, and data visualization.
    Strong analytical skills and experience with statistical modeling.
    """
    
    features = quick_feature_check(sample_resume)