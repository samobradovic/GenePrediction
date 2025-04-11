from pathlib import Path
from typing import NamedTuple
from unittest.mock import MagicMock, call, patch

import numpy as np
import numpy.testing as np_testing
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from src.model_training.xgboost_gene_predictor import (
    get_training_and_validation_data,
    train,
)


class TestTrain:
    class TrainingTestCase(NamedTuple):
        description: str
        paths: dict
        xgb_params: dict
        mock_data: dict
        expected_model_file: Path

    TRAINING_TEST_CASES = [
        TrainingTestCase(
            description="Valid training and validation data",
            paths={
                "training_vectorized_file": Path("training_vectorized.pkl"),
                "training_csv_file": Path("training.csv"),
                "validation_vectorized_file": Path("validation_vectorized.pkl"),
                "validation_csv_file": Path("validation.csv"),
                "model_file": Path("xgb_model.pkl"),
            },
            xgb_params={
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "eval_metric": "logloss",
                "early_stopping_rounds": 10,
            },
            mock_data={
                "X_train": [[1, 2], [3, 4]],
                "y_train": [0, 1],
                "X_val": [[5, 6]],
                "y_val": [1],
            },
            expected_model_file=Path("xgb_model.pkl"),
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        TRAINING_TEST_CASES,
        ids=[case.description for case in TRAINING_TEST_CASES],
    )
    @patch("src.model_training.xgboost_gene_predictor.save_pkl")
    @patch("src.model_training.xgboost_gene_predictor.XGBClassifier")
    @patch("src.model_training.xgboost_gene_predictor.get_training_and_validation_data")
    def test_train(
        self,
        mock_get_training_and_validation_data: MagicMock,
        mock_xgb_classifier: MagicMock,
        mock_save_pkl: MagicMock,
        test_case: TrainingTestCase,
    ):
        # Arrange
        mock_get_training_and_validation_data.return_value = (
            test_case.mock_data["X_train"],
            test_case.mock_data["y_train"],
            test_case.mock_data["X_val"],
            test_case.mock_data["y_val"],
        )
        mock_classifier_instance = MagicMock()
        mock_xgb_classifier.return_value = mock_classifier_instance

        # Act
        train(paths=test_case.paths, xgb_params=test_case.xgb_params)

        # Assert
        mock_get_training_and_validation_data.assert_called_once_with(test_case.paths)
        mock_xgb_classifier.assert_called_once_with(**test_case.xgb_params)
        mock_classifier_instance.fit.assert_called_once_with(
            test_case.mock_data["X_train"],
            test_case.mock_data["y_train"],
            eval_set=[(test_case.mock_data["X_val"], test_case.mock_data["y_val"])],
            verbose=True,
        )
        mock_save_pkl.assert_called_once_with(
            mock_classifier_instance, test_case.paths["model_file"]
        )


class TestGetTrainingAndValidationData:
    class GetTrainingAndValidationDataTestCase(NamedTuple):
        description: str
        paths: dict
        mock_data: dict
        expected_x_train: csr_matrix
        expected_y_train: np.ndarray
        expected_x_val: csr_matrix
        expected_y_val: np.ndarray

    TEST_CASES = [
        GetTrainingAndValidationDataTestCase(
            description="Valid training and validation data",
            paths={
                "training_vectorized_file": Path("training_vectorized.pkl"),
                "training_csv_file": Path("training.csv"),
                "validation_vectorized_file": Path("validation_vectorized.pkl"),
                "validation_csv_file": Path("validation.csv"),
            },
            mock_data={
                "X_train": csr_matrix([[1, 2], [3, 4]]),
                "y_train": pd.DataFrame({"IsGene": [0, 1]}),
                "X_val": csr_matrix([[5, 6]]),
                "y_val": pd.DataFrame({"IsGene": [1]}),
            },
            expected_x_train=csr_matrix([[1, 2], [3, 4]]),
            expected_y_train=np.array([0, 1]),
            expected_x_val=csr_matrix([[5, 6]]),
            expected_y_val=np.array([1]),
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        TEST_CASES,
        ids=[case.description for case in TEST_CASES],
    )
    @patch("src.model_training.xgboost_gene_predictor.dill.load")
    @patch("src.model_training.xgboost_gene_predictor.pd.read_csv")
    @patch("builtins.open", new_callable=MagicMock)
    def test_get_training_and_validation_data(
        self,
        mock_open: MagicMock,
        mock_read_csv: MagicMock,
        mock_dill_load: MagicMock,
        test_case: GetTrainingAndValidationDataTestCase,
    ):
        # Arrange
        mock_file_train = MagicMock()
        mock_file_val = MagicMock()
        mock_open.return_value.__enter__.side_effect = [mock_file_train, mock_file_val]

        mock_dill_load.side_effect = [
            test_case.mock_data["X_train"],
            test_case.mock_data["X_val"],
        ]
        mock_read_csv.side_effect = [
            test_case.mock_data["y_train"],
            test_case.mock_data["y_val"],
        ]

        # Act
        X_train, y_train, X_val, y_val = get_training_and_validation_data(
            test_case.paths
        )

        # Assert
        assert (X_train - test_case.expected_x_train).nnz == 0
        assert (X_val - test_case.expected_x_val).nnz == 0

        np_testing.assert_array_equal(
            y_train, test_case.expected_y_train, err_msg="y_train mismatch"
        )
        np_testing.assert_array_equal(
            y_val, test_case.expected_y_val, err_msg="y_val mismatch"
        )

        dill_calls = [
            call(mock_file_train),
            call(mock_file_val),
        ]
        mock_dill_load.assert_has_calls(dill_calls, any_order=True)

        read_csv_calls = [
            call(test_case.paths["training_csv_file"]),
            call(test_case.paths["validation_csv_file"]),
        ]
        mock_read_csv.assert_has_calls(read_csv_calls, any_order=True)

    class GetTrainingAndValidationDataErrorTestCase(NamedTuple):
        description: str
        paths: dict
        exception: type[Exception]

    ERROR_TEST_CASES = [
        GetTrainingAndValidationDataErrorTestCase(
            description="FileNotFoundError",
            paths={
                "training_vectorized_file": Path("non_existent_file.pkl"),
                "training_csv_file": Path("non_existent_file.csv"),
                "validation_vectorized_file": Path("non_existent_file.pkl"),
                "validation_csv_file": Path("non_existent_file.csv"),
            },
            exception=FileNotFoundError,
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        ERROR_TEST_CASES,
        ids=[case.description for case in ERROR_TEST_CASES],
    )
    @patch("builtins.open", new_callable=MagicMock)
    def test_get_training_and_validation_data_error(
        self,
        mock_open: MagicMock,
        test_case: GetTrainingAndValidationDataErrorTestCase,
    ):
        # Arrange
        mock_open.side_effect = test_case.exception

        # Act & Assert
        with pytest.raises(test_case.exception):
            get_training_and_validation_data(test_case.paths)
