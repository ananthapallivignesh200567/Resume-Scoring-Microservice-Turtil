import os
import json
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def train_goal_models(data_dir="../data", model_dir="../app/model"):
    """
    Train separate logistic regression models for each goal using individual TF-IDF vectorizers.
    """
    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Find all training files
    training_files = []
    for filename in os.listdir(data_dir):
        if filename.startswith("training_") and filename.endswith(".json"):
            training_files.append(filename)

    if not training_files:
        print(f"❌ No training files found in {data_dir}")
        print("Expected files with pattern: training_*.json")
        return

    print(f"\n🎯 Found {(training_files)} training files:")
    for f in training_files:
        print(f"  📄 {f}")

    # Train a model for each goal
    trained_models = {}
    model_metrics = {}
    vectorizers = {}

    for filename in training_files:
        
        goal_file = os.path.join(data_dir, filename)
        with open(goal_file, 'r') as f:
            data = json.load(f)

        goal_name = data[0].get("goal", "Unknown Goal").strip()  # Use canonical name from file
        

        

        print(f"\n🔧 Training model for goal: {goal_name}")
        print(f"📂 Processing: {filename}")

        try:
            # Load training data
            with open(goal_file, 'r') as f:
                data = json.load(f)

            if not data:
                print(f"  ⚠️  No data found in {filename}")
                continue

            # Extract features and labels
            texts = []
            labels = []

            for item in data:
                if 'resume_text' in item and 'label' in item:
                    texts.append(item['resume_text'])
                    labels.append(int(item['label']))
                else:
                    print(f"  ⚠️  Skipping invalid item: {item}")

            if len(texts) < 10:
                print(f"  ❌ Not enough samples ({len(texts)}) for {goal_name}. Need at least 10.")
                continue

            print(f"  📊 Loaded {len(texts)} samples")
            print(f"  ✅ Positive samples: {sum(labels)}")
            print(f"  ❌ Negative samples: {len(labels) - sum(labels)}")

            # Vectorize the texts using goal-specific vectorizer
            print(f"  🔄 Vectorizing texts...")

            y = np.array(labels)

            # Split into train/test
            texts_train, texts_test, y_train, y_test = train_test_split(texts, y, test_size=0.4, random_state=42, stratify=y)
            print(f"Train Labels: {np.bincount(y_train)}")
            print(f"Test Labels: {np.bincount(y_test)}")

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
            X_train = vectorizer.fit_transform(texts_train)
            X_test = vectorizer.transform(texts_test)
            print(f"  📈 Feature matrix shape: train {X_train.shape}, test {X_test.shape}")

            vectorizer_filename = f"{goal_name.lower().replace(' ', '_')}_vectorizer.pkl"
            vectorizer_path = os.path.join(model_dir, vectorizer_filename)
            joblib.dump(vectorizer, vectorizer_path)
            vectorizers[goal_name] = vectorizer_filename

            # Train logistic regression model
            print(f"  🤖 Training logistic regression model...")

            model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced',  # Handle class imbalance
                solver='lbfgs',
                n_jobs=-1
            )

            model.fit(X_train, y_train)

            # Evaluate model
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)

            print(f"  ✅ Model trained successfully!")
            print(f"  📊 Accuracy: {accuracy:.3f}")

            if len(set(y_test)) > 1:  # Only if we have both classes in test set
                print(f"  📋 Classification Report:")
                report = classification_report(y_test, y_pred, output_dict=True)
                print(f"     Precision: {report['1']['precision']:.3f}")
                print(f"     Recall: {report['1']['recall']:.3f}")
                print(f"     F1-score: {report['1']['f1-score']:.3f}")

                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                print(f"  🎯 Confusion Matrix: [[{cm[0,0]}, {cm[0,1]}], [{cm[1,0]}, {cm[1,1]}]]")

            # Save the model
            model_filename = f"{goal_name.lower().replace(' ', '_')}_model.pkl"
            model_path = os.path.join(model_dir, model_filename)
            joblib.dump(model, model_path)
            print(f"  💾 Model saved to: {model_path}")

            # Store model info
            trained_models[goal_name] = model_path
            model_metrics[goal_name] = {
                'accuracy': accuracy,
                'samples': len(texts),
                'positive_samples': sum(labels),
                'negative_samples': len(labels) - sum(labels),
                'feature_count': X_train.shape[1]
            }

        except Exception as e:
            print(f"  ❌ Error training model for {goal_name}: {e}")
            continue

    # Print summary
    print(f"\n📋 TRAINING SUMMARY")
    print(f"{'='*50}")
    print(f"✅ Successfully trained models: {len(trained_models)}")

    if trained_models:
        print(f"\n🎯 Trained Models:")
        for goal, model_path in trained_models.items():
            metrics = model_metrics[goal]
            print(f"  📄 {goal}:")
            print(f"     📁 File: {os.path.basename(model_path)}")
            print(f"     📊 Accuracy: {metrics['accuracy']:.3f}")
            print(f"     📈 Samples: {metrics['samples']} ({metrics['positive_samples']}+ / {metrics['negative_samples']}-)")
            print(f"     🔧 Features: {metrics['feature_count']}")

        # Save model registry
        registry = {
            'models': {goal: os.path.basename(path) for goal, path in trained_models.items()},
            'vectorizers': vectorizers,
            'metrics': model_metrics
        }

        registry_path = os.path.join(model_dir, "model_registry.json")
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)

        print(f"\n💾 Model registry saved to: {registry_path}")

    else:
        print("❌ No models were successfully trained!")
        print("Please check your training data format and try again.")

def test_trained_models(model_dir="../app/model"):
    """
    Test the trained models with sample inputs.
    """
    print("\n🧪 TESTING TRAINED MODELS")
    print("="*50)

    # Load model registry
    registry_path = os.path.join(model_dir, "model_registry.json")
    if not os.path.exists(registry_path):
        print("❌ Model registry not found. Train models first.")
        return

    with open(registry_path, 'r') as f:
        registry = json.load(f)

    # Test samples
    test_samples = [
        "Experienced software engineer with Java, Python, data structures, algorithms, system design, and AWS experience",
        "Machine learning engineer skilled in Python, TensorFlow, scikit-learn, pandas, numpy, deep learning",
        "Mechanical engineer with AutoCAD, SolidWorks, manufacturing, and materials science background",
        "Fresh graduate with basic programming knowledge in C++ and some web development"
    ]

    print("🎯 Testing with sample resumes:")

    for goal, model_file in registry['models'].items():
        print(f"\n📄 Goal: {goal}")
        print("-" * 30)

        # Load vectorizer specific to this goal
        vectorizer_file = registry['vectorizers'][goal]
        vectorizer_path = os.path.join(model_dir, vectorizer_file)
        vectorizer = joblib.load(vectorizer_path)

        # Load model
        model_path = os.path.join(model_dir, model_file)
        model = joblib.load(model_path)

        for i, sample in enumerate(test_samples, 1):
            X = vectorizer.transform([sample])
            score = model.predict_proba(X)[0, 1]
            prediction = "MATCH" if score > 0.5 else "NO MATCH"
            print(f"  {i}. Score: {score:.3f} ({prediction})")
            print(f"     Text: {sample[:60]}...")

if __name__ == "__main__":
    train_goal_models()
    test_trained_models()
