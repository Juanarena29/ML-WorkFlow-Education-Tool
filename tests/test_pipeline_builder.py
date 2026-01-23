"""
Tests para src/ml/pipeline_builder.py

Cubre:
- Construcción de preprocesador
- Construcción de pipeline completo
"""

import pytest
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression

from src.ml.pipeline_builder import build_preprocessing_pipeline, create_full_pipeline


class TestBuildPreprocessingPipeline:
    """Tests para build_preprocessing_pipeline."""

    def test_creates_numeric_transformer(self):
        """Crea transformer para features numéricas."""
        preprocessor = build_preprocessing_pipeline(
            numeric_features=["num1", "num2"],
            categorical_features=[],
        )

        assert preprocessor is not None
        transformer_names = [name for name, _, _ in preprocessor.transformers]
        assert "num" in transformer_names

    def test_creates_categorical_transformer(self):
        """Crea transformer para features categóricas."""
        preprocessor = build_preprocessing_pipeline(
            numeric_features=[],
            categorical_features=["cat1", "cat2"],
        )

        assert preprocessor is not None
        transformer_names = [name for name, _, _ in preprocessor.transformers]
        assert "cat" in transformer_names

    def test_creates_both_transformers(self):
        """Crea transformers para ambos tipos."""
        preprocessor = build_preprocessing_pipeline(
            numeric_features=["num1"],
            categorical_features=["cat1"],
        )

        transformer_names = [name for name, _, _ in preprocessor.transformers]
        assert "num" in transformer_names
        assert "cat" in transformer_names

    def test_raises_if_no_features(self):
        """Lanza error si no hay features."""
        with pytest.raises(ValueError, match="No hay features"):
            build_preprocessing_pipeline(
                numeric_features=[],
                categorical_features=[],
            )


class TestCreateFullPipeline:
    """Tests para create_full_pipeline."""

    def test_creates_pipeline_with_model(self):
        """Crea pipeline con preprocesador y modelo."""
        model = LogisticRegression()

        pipeline = create_full_pipeline(
            model=model,
            numeric_features=["num1"],
            categorical_features=["cat1"],
        )

        assert pipeline is not None
        assert "preprocessor" in pipeline.named_steps
        assert "model" in pipeline.named_steps

    def test_pipeline_can_fit_and_predict(self, sample_df_classification):
        """Pipeline puede entrenar y predecir."""
        model = LogisticRegression(max_iter=200)

        pipeline = create_full_pipeline(
            model=model,
            numeric_features=["feature_num1", "feature_num2"],
            categorical_features=["feature_cat"],
        )

        X = sample_df_classification.drop(columns=["target", "id_col"])
        y = sample_df_classification["target"]

        pipeline.fit(X, y)
        predictions = pipeline.predict(X)

        assert len(predictions) == len(y)

    def test_pipeline_handles_missing_values(self):
        """Pipeline maneja valores faltantes."""
        model = LogisticRegression(max_iter=200)

        pipeline = create_full_pipeline(
            model=model,
            numeric_features=["num"],
            categorical_features=["cat"],
        )

        # DataFrame con NaNs
        X = pd.DataFrame({
            "num": [1.0, np.nan, 3.0, 4.0],
            "cat": ["a", "b", None, "a"],
        })
        y = pd.Series([0, 1, 0, 1])

        # No debería lanzar error
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)

        assert len(predictions) == 4
