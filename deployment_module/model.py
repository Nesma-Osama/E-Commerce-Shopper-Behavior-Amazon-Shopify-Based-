import json
import pickle
from datetime import datetime

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configuration
CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "model_path": "model.pkl",
    "scaler_path": "scaler.pkl",
    "metadata_path": "model_metadata.json",
}

# Hyperparameter grid for tuning
PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "bootstrap": [True, False],
}


def load_and_preprocess_data():
    """Load dataset and perform preprocessing."""
    print("=" * 50)
    print("STEP 1: Loading and Preprocessing Data")
    print("=" * 50)

    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names
    target_names = data.target_names

    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Target classes: {target_names}")
    print(f"Class distribution: {np.bincount(y)}")

    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"],
        stratify=y,  # Maintain class distribution
    )

    print(f"\nTrain set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    return X_train, X_test, y_train, y_test, feature_names, target_names


def scale_features(X_train, X_test):
    """Apply feature scaling using StandardScaler."""
    print("\n" + "=" * 50)
    print("STEP 2: Feature Scaling")
    print("=" * 50)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train mean (before): {X_train.mean(axis=0).round(3)}")
    print(f"Train std (before): {X_train.std(axis=0).round(3)}")
    print(f"Train mean (after): {X_train_scaled.mean(axis=0).round(3)}")
    print(f"Train std (after): {X_train_scaled.std(axis=0).round(3)}")

    # Save scaler for inference
    with open(CONFIG["scaler_path"], "wb") as f:
        pickle.dump(scaler, f)
    print(f"\nScaler saved to {CONFIG['scaler_path']}")

    return X_train_scaled, X_test_scaled, scaler


def perform_hyperparameter_tuning(X_train, y_train, quick_search=True):
    """Perform hyperparameter tuning using GridSearchCV."""
    print("\n" + "=" * 50)
    print("STEP 3: Hyperparameter Tuning")
    print("=" * 50)

    # Use smaller grid for quick testing
    if quick_search:
        param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [None, 10],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        }
        print("Using quick search grid (subset of parameters)")
    else:
        param_grid = PARAM_GRID
        print("Using full search grid")

    base_model = RandomForestClassifier(random_state=CONFIG["random_state"])

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=CONFIG["cv_folds"],
        scoring="accuracy",
        n_jobs=-1,  # Use all available cores
        verbose=1,
        return_train_score=True,
    )

    print(
        f"\nSearching {len(param_grid)} parameters with {CONFIG['cv_folds']}-fold CV..."
    )
    grid_search.fit(X_train, y_train)

    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")

    return (
        grid_search.best_estimator_,
        grid_search.best_params_,
        grid_search.best_score_,
    )


def cross_validate_model(model, X_train, y_train):
    """Perform cross-validation on the model."""
    print("\n" + "=" * 50)
    print("STEP 4: Cross-Validation")
    print("=" * 50)

    cv_scores = cross_val_score(
        model, X_train, y_train, cv=CONFIG["cv_folds"], scoring="accuracy"
    )

    print(f"CV Scores: {cv_scores.round(4)}")
    print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    return cv_scores


def evaluate_model(model, X_test, y_test, target_names):
    """Evaluate model performance on test set."""
    print("\n" + "=" * 50)
    print("STEP 5: Model Evaluation")
    print("=" * 50)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("\nTest Set Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
    }

    return metrics


def get_feature_importance(model, feature_names):
    """Extract and display feature importances."""
    print("\n" + "=" * 50)
    print("STEP 6: Feature Importance")
    print("=" * 50)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\nFeature ranking:")
    for i, idx in enumerate(indices):
        print(f"  {i + 1}. {feature_names[idx]}: {importances[idx]:.4f}")

    return dict(zip(feature_names, importances.tolist()))


def save_model_and_metadata(
    model, best_params, cv_score, metrics, feature_importance
):
    """Save model and metadata for deployment."""
    print("\n" + "=" * 50)
    print("STEP 7: Saving Model and Metadata")
    print("=" * 50)

    # Save model
    with open(CONFIG["model_path"], "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {CONFIG['model_path']}")

    # Save metadata
    metadata = {
        "model_type": "RandomForestClassifier",
        "training_date": datetime.now().isoformat(),
        "config": CONFIG,
        "best_params": best_params,
        "cv_score": float(cv_score),
        "test_metrics": metrics,
        "feature_importance": feature_importance,
        "sklearn_version": "latest",
    }

    with open(CONFIG["metadata_path"], "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {CONFIG['metadata_path']}")

    return metadata


def train_model():
    """Full training pipeline."""
    print("\n" + "#" * 60)
    print("#" + " " * 20 + "TRAINING PIPELINE" + " " * 21 + "#")
    print("#" * 60)

    # Step 1: Load and preprocess
    X_train, X_test, y_train, y_test, feature_names, target_names = (
        load_and_preprocess_data()
    )

    # Step 2: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Step 3: Hyperparameter tuning
    best_model, best_params, cv_score = perform_hyperparameter_tuning(
        X_train_scaled, y_train, quick_search=True
    )

    # Step 4: Cross-validation
    cv_scores = cross_validate_model(best_model, X_train_scaled, y_train)

    # Step 5: Evaluate on test set
    metrics = evaluate_model(best_model, X_test_scaled, y_test, target_names)

    # Step 6: Feature importance
    feature_importance = get_feature_importance(best_model, feature_names)

    # Step 7: Save everything
    metadata = save_model_and_metadata(
        best_model, best_params, cv_score, metrics, feature_importance
    )

    print("\n" + "#" * 60)
    print("#" + " " * 18 + "TRAINING COMPLETE!" + " " * 20 + "#")
    print("#" * 60)

    return best_model, scaler, metadata


def predict(input_data=None):
    """Make predictions using saved model and scaler."""
    print("\n" + "#" * 60)
    print("#" + " " * 18 + "INFERENCE PIPELINE" + " " * 20 + "#")
    print("#" * 60)

    # Load model
    with open(CONFIG["model_path"], "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {CONFIG['model_path']}")

    # Load scaler
    with open(CONFIG["scaler_path"], "rb") as f:
        scaler = pickle.load(f)
    print(f"Scaler loaded from {CONFIG['scaler_path']}")

    # Load metadata
    with open(CONFIG["metadata_path"], "r") as f:
        metadata = json.load(f)
    print(f"Model trained on: {metadata['training_date']}")

    # Default test samples
    if input_data is None:
        input_data = [
            [5.1, 3.5, 1.4, 0.2],  # Likely setosa
            [7.0, 3.2, 4.7, 1.4],  # Likely versicolor
            [6.3, 3.3, 6.0, 2.5],  # Likely virginica
        ]

    # Ensure 2D array
    input_array = np.array(input_data)
    if input_array.ndim == 1:
        input_array = input_array.reshape(1, -1)

    # Scale input
    input_scaled = scaler.transform(input_array)

    # Predict
    predictions = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)

    # Map to class names
    target_names = ["setosa", "versicolor", "virginica"]

    print("\n" + "=" * 50)
    print("Predictions:")
    print("=" * 50)
    for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
        print(f"\nSample {i + 1}: {input_array[i].tolist()}")
        print(f"  Predicted class: {pred} ({target_names[pred]})")
        print(f"  Probabilities: {dict(zip(target_names, proba.round(4).tolist()))}")

    return predictions, probabilities


if __name__ == "__main__":
    train_model()
    predict()
