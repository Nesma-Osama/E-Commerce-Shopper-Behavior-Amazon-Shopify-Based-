import os
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
)

java17_home = Path("/usr/lib/jvm/java-17-openjdk-amd64")
if java17_home.exists():
    os.environ["JAVA_HOME"] = str(java17_home)
    os.environ["PATH"] = f"{java17_home / 'bin'}:{os.environ.get('PATH', '')}"

try:
    from pyspark import StorageLevel
    from pyspark.sql import SparkSession
except ImportError as exc:
    raise ImportError(
        "Install pyspark first. If you use the project venv, run: .venv/bin/pip install pyspark"
    ) from exc


def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1), dtype=np.float32), X])


def sigmoid(z):
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def partition_stats_binary(partition, weights, threshold):
    grad = np.zeros_like(weights, dtype=np.float32)
    loss = 0.0
    count = 0

    for feature_block, label_block in partition:
        X_block = np.asarray(feature_block, dtype=np.float32)
        y_block = (np.asarray(label_block, dtype=np.int64) >= threshold).astype(np.float32)

        probs = sigmoid(X_block @ weights)
        grad += ((probs - y_block)[:, None] * X_block).sum(axis=0)
        loss += float(
            -(y_block * np.log(probs + 1e-12) + (1.0 - y_block) * np.log(1.0 - probs + 1e-12)).sum()
        )
        count += int(X_block.shape[0])

    if count > 0:
        yield grad, loss, count


def partition_stats_linear(partition, weights):
    grad = np.zeros_like(weights, dtype=np.float32)
    mse_sum = 0.0
    count = 0

    for feature_block, label_block in partition:
        X_block = np.asarray(feature_block, dtype=np.float32)
        y_block = np.asarray(label_block, dtype=np.float32)

        preds = X_block @ weights
        errors = preds - y_block
        grad += (errors[:, None] * X_block).sum(axis=0)
        mse_sum += float((errors ** 2).sum())
        count += int(X_block.shape[0])

    if count > 0:
        yield grad, mse_sum, count


def train_threshold_models(sc, train_rdd, num_features, thresholds=(1, 2), epochs=25, lr=0.8, reg=0.0001):
    all_weights = []

    for threshold in thresholds:
        print(f"\nTraining threshold y >= {threshold}")
        weights = np.zeros(num_features, dtype=np.float32)

        for epoch in range(1, epochs + 1):
            weights_bc = sc.broadcast(weights)

            grad_sum, loss_sum, count = train_rdd.mapPartitions(
                lambda partition: partition_stats_binary(partition, weights_bc.value, threshold)
            ).reduce(
                lambda left, right: (left[0] + right[0], left[1] + right[1], left[2] + right[2])
            )

            weights_bc.unpersist()
            gradient = (grad_sum / count) + reg * weights
            weights -= lr * gradient

            if epoch in {1, 2, 5, 10, 15, 20, 25}:
                print(f"  epoch {epoch:02d} | loss = {loss_sum / count:.4f}")

        all_weights.append(weights)

    return np.vstack(all_weights)


def train_linear_model(sc, train_rdd, num_features, epochs=25, lr=0.1, reg=0.0001):
    weights = np.zeros(num_features, dtype=np.float32)

    for epoch in range(1, epochs + 1):
        weights_bc = sc.broadcast(weights)

        grad_sum, mse_sum, count = train_rdd.mapPartitions(
            lambda partition: partition_stats_linear(partition, weights_bc.value)
        ).reduce(
            lambda left, right: (left[0] + right[0], left[1] + right[1], left[2] + right[2])
        )

        weights_bc.unpersist()
        gradient = (grad_sum / count) + reg * weights
        weights -= lr * gradient

        if epoch in {1, 2, 5, 10, 15, 20, 25}:
            print(f"  epoch {epoch:02d} | mse = {mse_sum / count:.4f}")

    return weights


def predict_linear_values(X, linear_weights):
    return X @ linear_weights


def evaluate_linear_split(name, X, y, linear_weights):
    preds = predict_linear_values(X, linear_weights)
    mse = mean_squared_error(y, preds)
    print(f"\n{name} MSE: {mse:.6f}")
    print("Sample predictions (pred, true):")
    for pred, true_val in zip(preds[:5], y[:5]):
        print(f"  ({pred:.4f}, {float(true_val):.4f})")


def predict_ordinal(X, threshold_weights):
    p_ge_1 = sigmoid(X @ threshold_weights[0])
    p_ge_2 = sigmoid(X @ threshold_weights[1])

    p_ge_2 = np.minimum(p_ge_1, p_ge_2)
    probs = np.column_stack([1.0 - p_ge_1, p_ge_1 - p_ge_2, p_ge_2])
    return np.argmax(probs, axis=1)


def evaluate_split(name, X, y, threshold_weights):
    preds = predict_ordinal(X, threshold_weights)
    print(f"\n{name} accuracy: {accuracy_score(y, preds):.4f}")
    print(classification_report(y, preds, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y, preds))