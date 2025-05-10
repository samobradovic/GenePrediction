from pathlib import Path
from typing import NamedTuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.evaluation import evaluation


class TestGenerateROCCurve:
    class CaseData(NamedTuple):
        description: str
        y_true: np.ndarray
        y_score: np.ndarray
        false_positive_rates: np.ndarray
        true_positive_rates: np.ndarray
        thresholds: np.ndarray
        area_under_curve: float
        path: Path

    TEST_CASES = [
        CaseData(
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
        test_case: CaseData,
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
    class CaseData(NamedTuple):
        description: str
        y_true: np.ndarray
        y_probs: np.ndarray
        precision: np.ndarray
        recall: np.ndarray
        thresholds: np.ndarray
        path: Path

    TEST_CASES = [
        CaseData(
            description="Random values, valid Precision-Recall curve",
            y_true=np.random.randint(0, 2, 10),
            y_probs=np.random.random(10),
            precision=np.random.random(10),
            recall=np.random.random(10),
            thresholds=np.random.random(10),
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
    def test_generate_precision_recall_curve(
        self,
        mock_savefig: MagicMock,
        mock_precision_recall: MagicMock,
        test_case: CaseData,
    ):
        # Arrange
        mock_precision_recall.return_value = (
            test_case.precision,
            test_case.recall,
            test_case.thresholds,
        )

        # Act
        evaluation.generate_precision_recall_curve(
            test_case.y_true, test_case.y_probs, test_case.path
        )

        # Assert
        mock_precision_recall.assert_called_once_with(
            test_case.y_true, test_case.y_probs
        )
        mock_savefig.assert_called_once_with(test_case.path)


class TestGenerateConfusionMatrix:
    class CaseData(NamedTuple):
        description: str
        y_true: np.ndarray
        y_score: np.ndarray
        confusion_matrix: np.ndarray
        path: Path

    TEST_CASES = [
        CaseData(
            description="Random values, valid confusion matrix",
            y_true=np.random.randint(0, 2, 10),
            y_score=np.random.randint(0, 2, 10),
            confusion_matrix=np.random.randint(0, 10, (2, 2)),
            path=Path("confusion_matrix.png"),
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        TEST_CASES,
        ids=[case.description for case in TEST_CASES],
    )
    @patch("src.evaluation.evaluation.ConfusionMatrixDisplay")
    @patch("src.evaluation.evaluation.confusion_matrix")
    @patch("src.evaluation.evaluation.plt.savefig")
    def test_generate_confusion_matrix(
        self,
        mock_savefig: MagicMock,
        mock_confusion_matrix: MagicMock,
        mock_matrix_display: MagicMock,
        test_case: CaseData,
    ):
        # Arrange
        mock_confusion_matrix.return_value = test_case.confusion_matrix
        mock_disp_instance = MagicMock()
        mock_matrix_display.return_value = mock_disp_instance

        # Act
        evaluation.generate_confusion_matrix(
            test_case.y_true, test_case.y_score, test_case.path
        )

        # Assert
        mock_confusion_matrix.assert_called_once_with(
            test_case.y_true, test_case.y_score
        )
        mock_savefig.assert_called_once_with(test_case.path)


class TestGenerateClassificationReport:
    class CaseData(NamedTuple):
        description: str
        y_true: np.ndarray
        y_score: np.ndarray
        classification_report: dict
        path: Path

    TEST_CASES = [
        CaseData(
            description="Random values, valid classification report",
            y_true=np.random.randint(0, 2, 10),
            y_score=np.random.randint(0, 2, 10),
            classification_report={
                "precision": np.random.random(10),
                "recall": np.random.random(10),
                "f1-score": np.random.random(10),
                "support": np.random.randint(0, 10, 10),
            },
            path=Path("classification_report.csv"),
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        TEST_CASES,
        ids=[case.description for case in TEST_CASES],
    )
    @patch("src.evaluation.evaluation.classification_report")
    @patch("pandas.DataFrame.to_csv")
    def test_generate_classification_report(
        self,
        mock_to_csv: MagicMock,
        mock_classification_report: MagicMock,
        test_case: CaseData,
    ):
        # Arrange
        mock_classification_report.return_value = test_case.classification_report

        # Act
        evaluation.generate_classification_report(
            test_case.y_true, test_case.y_score, test_case.path
        )

        # Assert
        mock_classification_report.assert_called_once_with(
            test_case.y_true, test_case.y_score, output_dict=True
        )
        mock_to_csv.assert_called_once_with(test_case.path, index=True)


class TestSaveAccuracy:
    class CaseData(NamedTuple):
        description: str
        y_true: np.ndarray
        y_score: np.ndarray
        accuracy: float
        path: Path

    TEST_CASES = [
        CaseData(
            description="Random values, valid accuracy score",
            y_true=np.random.randint(0, 2, 10),
            y_score=np.random.randint(0, 2, 10),
            accuracy=0.8,
            path=Path("accuracy.txt"),
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        TEST_CASES,
        ids=[case.description for case in TEST_CASES],
    )
    @patch("src.evaluation.evaluation.accuracy_score")
    @patch("builtins.open", new_callable=MagicMock)
    def test_generate_save_accuracy(
        self,
        mock_open: MagicMock,
        mock_accuracy: MagicMock,
        test_case: CaseData,
    ):
        # Arrange
        mock_accuracy.return_value = test_case.accuracy

        # Act
        evaluation.save_accuracy(test_case.y_true, test_case.y_score, test_case.path)

        # Assert
        mock_accuracy.assert_called_once_with(test_case.y_true, test_case.y_score)
        mock_open.assert_called_once_with(test_case.path, "w")
        mock_open.return_value.__enter__.return_value.write.assert_called_once_with(
            f"Model Accuracy: {test_case.accuracy:.2%}\n"
        )
