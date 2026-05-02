import json
import pickle
from typing import Any
from typing import Dict

import numpy as np
import pandas as pd


EDUCATION_LEVEL_MAPPING = {
    "High School": 0,
    "Associate Degree": 1,
    "Bachelor": 2,
    "Master": 3,
    "PhD": 4,
}

BUDGETING_STYLE_MAPPING = {
    "Loose": 0,
    "Moderate": 1,
    "Strict": 2,
}


class ModelService:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self._artifacts: Dict[str, Any] = {}
        self._preprocessor = None
        self._target_scaler = None
        self._selected_features = None
        self._raw_feature_count = None
        self._transformed_feature_count = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        with open(self.config_path, "r") as f:
            cfg = json.load(f)
        models = cfg.get("models", {})
        for name, path in models.items():
            with open(path, "rb") as f:
                self._artifacts[name] = pickle.load(f)

        preprocessing_artifact = self._artifacts.get("preprocessing_pipeline.pkl")

        if isinstance(preprocessing_artifact, dict):
            self._preprocessor = (
                preprocessing_artifact.get("feature_preprocessor")
                or preprocessing_artifact.get("preprocessor")
                or preprocessing_artifact.get("pipeline")
            )
            self._target_scaler = preprocessing_artifact.get("target_scaler")
            self._selected_features = preprocessing_artifact.get("selected_features")
        else:
            self._preprocessor = preprocessing_artifact

        if self._target_scaler is None:
            self._target_scaler = self._artifacts.get("target_scaling.pkl")

        if not self._selected_features:
            feature_artifact = self._artifacts.get("total_remaining_features.pkl")
            if isinstance(feature_artifact, list) and feature_artifact:
                self._selected_features = feature_artifact

        if isinstance(self._selected_features, list) and self._selected_features:
            self._raw_feature_count = len(self._selected_features)
        elif self._preprocessor is not None:
            self._raw_feature_count = getattr(self._preprocessor, "n_features_in_", None)

        knn_model = self._artifacts.get("knn_model")
        if knn_model is not None:
            self._transformed_feature_count = getattr(knn_model, "n_features_in_", None)

        if self._transformed_feature_count is None:
            weights = self._artifacts.get("linear_regression_weights")
            if weights is not None and len(weights) > 1:
                self._transformed_feature_count = int(len(weights) - 1)

        self._loaded = True

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _add_bias(X):
        return np.hstack([np.ones((X.shape[0], 1), dtype=np.float32), X])

    def _prepare_tabular_input(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = df.copy()

        if self._selected_features:
            if all(col in prepared.columns for col in self._selected_features):
                prepared = prepared[self._selected_features].copy()

        if "education_level" in prepared.columns and not pd.api.types.is_numeric_dtype(prepared["education_level"]):
            mapped = prepared["education_level"].map(EDUCATION_LEVEL_MAPPING)
            if mapped.isna().any():
                raise ValueError("Unknown values in 'education_level' for ordinal mapping")
            prepared["education_level"] = mapped

        if "budgeting_style" in prepared.columns and not pd.api.types.is_numeric_dtype(prepared["budgeting_style"]):
            mapped = prepared["budgeting_style"].map(BUDGETING_STYLE_MAPPING)
            if mapped.isna().any():
                raise ValueError("Unknown values in 'budgeting_style' for ordinal mapping")
            prepared["budgeting_style"] = mapped

        return prepared

    def _extract_ordinal_features(self, source):
        ordinal_columns = ["education_level", "budgeting_style"]
        if not ordinal_columns:
            return None

        if isinstance(source, pd.DataFrame):
            if not all(column_name in source.columns for column_name in ordinal_columns):
                return None

            encoded_columns = []
            for column_name in ordinal_columns:
                values = source[column_name]
                if pd.api.types.is_numeric_dtype(values):
                    encoded = values.to_numpy(dtype=np.float32)
                else:
                    mapping = (
                        EDUCATION_LEVEL_MAPPING
                        if column_name == "education_level"
                        else BUDGETING_STYLE_MAPPING
                    )
                    encoded = values.map(mapping)
                    if encoded.isna().any():
                        raise ValueError(f"Unknown values in '{column_name}' for ordinal mapping")
                    encoded = encoded.to_numpy(dtype=np.float32)
                encoded_columns.append(encoded.reshape(-1, 1))

            return np.hstack(encoded_columns)

        if isinstance(source, np.ndarray) and source.ndim == 2 and self._selected_features:
            ordinal_indices = []
            for column_name in ordinal_columns:
                try:
                    ordinal_indices.append(self._selected_features.index(column_name))
                except ValueError:
                    return None
            return np.asarray(source[:, ordinal_indices], dtype=np.float32)

        return None

    def _prepare_input(self, input_data):
        if not self._loaded:
            self.load()

        if isinstance(input_data, pd.DataFrame):
            tabular_df = input_data
        elif (
            isinstance(input_data, list)
            and input_data
            and isinstance(input_data[0], dict)
        ):
            tabular_df = pd.DataFrame(input_data)
        else:
            tabular_df = None

        if tabular_df is not None:
            ordinal_features = self._extract_ordinal_features(tabular_df)
            prepared_df = self._prepare_tabular_input(tabular_df)

            if self._preprocessor is not None:
                X = self._preprocessor.transform(prepared_df)
            else:
                X = prepared_df.to_numpy()

            if ordinal_features is not None:
                X = np.hstack([X, ordinal_features])

            return np.asarray(X, dtype=np.float32)

        X = np.array(input_data)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        raw_input = X

        if self._preprocessor is not None:
            if self._raw_feature_count is not None and X.shape[1] == self._raw_feature_count:
                X = self._preprocessor.transform(X)
                ordinal_features = self._extract_ordinal_features(raw_input)
                if ordinal_features is not None:
                    X = np.hstack([X, ordinal_features])
            elif self._transformed_feature_count is not None and X.shape[1] == self._transformed_feature_count:
                pass
            elif self._raw_feature_count is None:
                X = self._preprocessor.transform(X)
            else:
                raise ValueError(
                    f"Input has {X.shape[1]} features, but expected either "
                    f"{self._raw_feature_count} raw features or "
                    f"{self._transformed_feature_count} transformed features."
                )

        return np.asarray(X, dtype=np.float32)

    def predict_classifier(self, model_name, input_data):
        X = self._prepare_input(input_data)

        if model_name in {"knn", "knn_model"}:
            model = self._artifacts.get("knn_model")
            if model is None:
                raise ValueError("Classifier 'knn_model' is not available")
            return model.predict(X)

        if model_name in {"logistic_regression", "ordinal_logistic", "ordinal_logistic_threshold_weights"}:
            threshold_weights = self._artifacts.get("ordinal_logistic_threshold_weights")
            if threshold_weights is None:
                raise ValueError("Classifier 'ordinal_logistic_threshold_weights' is not available")
            X_bias = self._add_bias(X)
            p_ge_1 = self._sigmoid(X_bias @ threshold_weights[0])
            p_ge_2 = self._sigmoid(X_bias @ threshold_weights[1])
            p_ge_2 = np.minimum(p_ge_1, p_ge_2)
            probs = np.column_stack([1.0 - p_ge_1, p_ge_1 - p_ge_2, p_ge_2])
            return np.argmax(probs, axis=1)

        raise ValueError(f"Unknown classifier model: {model_name}")

    def predict_regressor(self, model_name, input_data):
        X = self._prepare_input(input_data)

        if model_name in {"linear_regression", "linear_regression_weights"}:
            weights = self._artifacts.get("linear_regression_weights")
            if weights is None:
                raise ValueError("Regressor 'linear_regression_weights' is not available")
            preds_scaled = self._add_bias(X) @ weights
            preds_scaled = preds_scaled.reshape(-1, 1)
            if self._target_scaler is not None:
                preds = self._target_scaler.inverse_transform(preds_scaled).ravel()
            else:
                preds = preds_scaled.ravel()
            return preds

        raise ValueError(f"Unknown regressor model: {model_name}")


service = ModelService()
