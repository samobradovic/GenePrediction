from pathlib import Path
from typing import NamedTuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.evaluation import evaluation


class TestGenerateROCCurve:
    class TestCase(NamedTuple):
        description: str
        y_true: np.ndarray
        y_score: np.ndarray
        false_positive_rates: np.ndarray
        true_positive_rates: np.ndarray
        thresholds: np.ndarray
        area_under_curve: float
        path: Path

    TEST_CASES = [
        TestCase(
            description="Random values, valid ROC curve",
            y_true=np.random.randint(0, 2, 10),
            y_score=np.random.randint(0, 2, 10),
            false_positive_rates=np.random.random(10),
            true_positive_rates=np.random.random(10),
            thresholds=np.random.random(10),
            area_under_curve=0.8,
            path=Path("roc_curve.png"),
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        TEST_CASES,
        ids=[case.description for case in TEST_CASES],
    )
    @patch("src.evaluation.evaluation.auc")
    @patch("src.evaluation.evaluation.roc_curve")
    @patch("src.evaluation.evaluation.plt.savefig")
    def test_generate_roc_curve(
        self,
        mock_savefig: MagicMock,
        mock_roc_curve: MagicMock,
        mock_auc: MagicMock,
        test_case: TestCase,
    ):
        # Arrange
        mock_roc_curve.return_value = (
            test_case.false_positive_rates,
            test_case.true_positive_rates,
            test_case.thresholds,
        )
        mock_auc.return_value = test_case.area_under_curve

        # Act
        evaluation.generate_roc_curve(
            test_case.y_true, test_case.y_score, test_case.path
        )

        # Assert
        mock_roc_curve.assert_called_once_with(test_case.y_true, test_case.y_score)
        mock_auc.assert_called_once_with(
            mock_roc_curve.return_value[0], mock_roc_curve.return_value[1]
        )
        mock_savefig.assert_called_once_with(test_case.path)


class TestGeneratePrecisionRecallCurve:
    class TestCase(NamedTuple):
        description: str
        y_true: np.ndarray
        y_score: np.ndarray
        precision: np.ndarray
        recall: np.ndarray
        path: Path

    TEST_CASES = [
        TestCase(
            description="Random values, valid Precision-Recall curve",
            y_true=np.random.randint(0, 2, 10),
            y_score=np.random.randint(0, 2, 10),
            precision=np.random.random(10),
            recall=np.random.random(10),
            path=Path("precision_recall_curve.png"),
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        TEST_CASES,
        ids=[case.description for case in TEST_CASES],
    )
    @patch("src.evaluation.evaluation.precision_recall_curve")
    @patch("src.evaluation.evaluation.plt.savefig")
    def test_generate_roc_curve(
        self,
        mock_savefig: MagicMock,
        mock_precision_recall: MagicMock,
        test_case: TestCase,
    ):
        # Arrange
        mock_precision_recall.return_value = (
            test_case.precision,
            test_case.recall,
        )

        # Act
        evaluation.generate_precision_recall_curve(
            test_case.y_true, test_case.y_score, test_case.path
        )

        # Assert
        mock_precision_recall.assert_called_once_with(
            test_case.y_true, test_case.y_score
        )
        mock_savefig.assert_called_once_with(test_case.path)


class TestGenerateConfusionMatrix:
    class TestCase(NamedTuple):
        description: str
        y_true: np.ndarray
        y_score: np.ndarray
        precision: np.ndarray
        recall: np.ndarray
        path: Path

    TEST_CASES = [
        TestCase(
            description="Random values, valid Precision-Recall curve",
            y_true=np.random.randint(0, 2, 10),
            y_score=np.random.randint(0, 2, 10),
            precision=np.random.random(10),
            recall=np.random.random(10),
            path=Path("precision_recall_curve.png"),
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        TEST_CASES,
        ids=[case.description for case in TEST_CASES],
    )
    @patch("src.evaluation.evaluation.precision_recall_curve")
    @patch("src.evaluation.evaluation.plt.savefig")
    def test_generate_roc_curve(
        self,
        mock_savefig: MagicMock,
        mock_precision_recall: MagicMock,
        test_case: TestCase,
    ):
        # Arrange
        mock_precision_recall.return_value = (
            test_case.precision,
            test_case.recall,
        )

        # Act
        evaluation.generate_precision_recall_curve(
            test_case.y_true, test_case.y_score, test_case.path
        )

        # Assert
        mock_precision_recall.assert_called_once_with(
            test_case.y_true, test_case.y_score
        )
        mock_savefig.assert_called_once_with(test_case.path)
