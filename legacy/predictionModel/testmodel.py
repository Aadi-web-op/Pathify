import pickle
import json
import pandas as pd
import sys

# Install scikit-learn if not already installed
try:
    import sklearn
except ImportError:
    print("Installing scikit-learn...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    import sklearn

# Load the model with error handling
print("Loading model...")
try:
    with open('career_prediction_model_v80_20251130_131358.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\nTrying alternative loading method...")
    try:
        import joblib
        model = joblib.load('career_prediction_model_v80_20251130_131358.pkl')
        print("✅ Model loaded successfully with joblib!")
    except ImportError:
        print("Installing joblib...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])
        import joblib
        model = joblib.load('career_prediction_model_v80_20251130_131358.pkl')
        print("✅ Model loaded successfully with joblib!")
    except Exception as e2:
        print(f"❌ Still failed: {e2}")
        print("\n💡 The model might have been created with different Python/library versions.")
        sys.exit(1)

# Load encoders
print("Loading encoders...")
try:
    # Import LabelEncoder before loading
    from sklearn.preprocessing import LabelEncoder
    
    import joblib
    encoders = joblib.load('model_encoders_20251130_131358.pkl')
    print("✅ Encoders loaded successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not load encoders: {e}")
    print("Continuing without encoders...")
    encoders = None

# Load metadata to see what features the model expects
print("\nLoading metadata...")
try:
    with open('model_metadata_20251130_13135', 'r') as f:
        metadata = json.load(f)
    print("✅ Metadata loaded successfully!")
except Exception as e:
    print(f"⚠️ Could not load metadata: {e}")
    print("Will inspect model directly instead...")
    metadata = {}

print("\n" + "="*50)
print("MODEL INFORMATION")
print("="*50)

# Try to get info from model directly
try:
    if hasattr(model, 'feature_names_in_'):
        print(f"Features expected: {model.feature_names_in_}")
        print(f"Number of features: {len(model.feature_names_in_)}")
    elif hasattr(model, 'n_features_in_'):
        print(f"Number of features: {model.n_features_in_}")
    else:
        print("Could not determine features from model")
    
    if hasattr(model, 'classes_'):
        print(f"Target classes: {model.classes_}")
        print(f"Number of classes: {len(model.classes_)}")
    
    print(f"Model type: {type(model).__name__}")
    
except Exception as e:
    print(f"Could not inspect model: {e}")

# If we have encoders, show them
if encoders:
    print("\n📊 Encoders available:")
    if isinstance(encoders, dict):
        for key in encoders.keys():
            print(f"   - {key}")
    
print("="*50)

# Example: Test with sample data
# ⚠️ MODIFY THIS based on your actual features
print("\n🧪 Testing with sample data...\n")

# Option 1: If you know the exact feature names
sample_data = {
    'feature1': [85],  # Replace with actual feature names
    'feature2': [90],
    'feature3': [75],
    # Add more features as needed
}

# Create DataFrame
df = pd.DataFrame(sample_data)

# Make prediction
try:
    prediction = model.predict(df)
    print(f"✅ Prediction: {prediction[0]}")
    
    # If model supports probability prediction
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(df)
        print(f"\n📊 Confidence scores:")
        classes = metadata.get('classes', [])
        for i, prob in enumerate(probabilities[0]):
            career = classes[i] if i < len(classes) else f"Class {i}"
            print(f"   {career}: {prob*100:.2f}%")
            
except Exception as e:
    print(f"❌ Error during prediction: {e}")
    print("\n💡 You need to provide the correct features.")
    print("Check the metadata above to see what features are expected.")

print("\n" + "="*50)
print("INTERACTIVE MODE")
print("="*50)
print("Enter values for each feature (or type 'quit' to exit):")

# Interactive testing loop
while True:
    try:
        print("\nEnter feature values:")
        # ⚠️ MODIFY based on your actual features
        val1 = input("Feature 1: ")
        if val1.lower() == 'quit':
            break
        val2 = input("Feature 2: ")
        val3 = input("Feature 3: ")
        
        # Create input data
        test_data = pd.DataFrame({
            'feature1': [float(val1)],
            'feature2': [float(val2)],
            'feature3': [float(val3)],
        })
        
        # Predict
        prediction = model.predict(test_data)
        print(f"\n🎯 Predicted Career: {prediction[0]}")
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(test_data)
            print(f"Confidence: {max(probabilities[0])*100:.2f}%")
        
    except ValueError:
        print("❌ Invalid input. Please enter numbers.")
    except KeyboardInterrupt:
        print("\n\nExiting...")
        break
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n👋 Testing complete!")