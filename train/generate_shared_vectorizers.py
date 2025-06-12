import os
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def generate_shared_vectorizer(data_dir="../data", model_dir="../app/model"):
    """
    Generate a single shared TF-IDF vectorizer trained on all resume texts
    from all goal training files in the data directory.
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # Collect all resume texts from all training files
    all_texts = []
    processed_files = []
    
    print("Collecting resume texts from all training files...")
    
    for filename in os.listdir(data_dir):
        if filename.startswith("training_") and filename.endswith(".json"):
            goal_name = filename[len("training_"):-len(".json")]
            goal_file = os.path.join(data_dir, filename)
            
            print(f"Processing {filename} (goal: {goal_name})")
            
            try:
                with open(goal_file, 'r') as f:
                    data = json.load(f)
                
                # Extract resume texts from this file
                texts = [item['resume_text'] for item in data if 'resume_text' in item]
                
                if texts:
                    all_texts.extend(texts)
                    processed_files.append(filename)
                    print(f"  ✅ Added {len(texts)} resume texts from {filename}")
                else:
                    print(f"  ⚠️  No resume texts found in {filename}")
                    
            except Exception as e:
                print(f"  ❌ Error processing {filename}: {e}")
    
    if not all_texts:
        print("❌ No resume texts found in any training files!")
        return
    
    print(f"\n📊 Total resume texts collected: {len(all_texts)}")
    print(f"📁 Files processed: {len(processed_files)}")
    print(f"📝 Files: {', '.join(processed_files)}")
    
    # Train shared TF-IDF vectorizer on all texts
    print("\n🔧 Training shared TF-IDF vectorizer...")
    
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.8,
        lowercase=True,
        token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b',
        sublinear_tf=True
    )
    
    vectorizer.fit(all_texts)
    
    # Save the shared vectorizer
    vectorizer_filename = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    joblib.dump(vectorizer, vectorizer_filename)
    
    print(f"✅ Shared TF-IDF vectorizer saved to {vectorizer_filename}")
    print(f"📈 Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"🎯 Feature matrix shape would be: ({len(all_texts)}, {len(vectorizer.vocabulary_)})")

if __name__ == "__main__":
    generate_shared_vectorizer()