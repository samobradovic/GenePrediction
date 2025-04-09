from pathlib import Path
from typing import NamedTuple
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from src.data_preprocessing.train_test_splitter import (
    save_train_test_validation_csvs,
    train_test_validation_split,
)


@pytest.fixture
def mock_read_csv():
    with patch("pandas.read_csv") as mock:
        yield mock


@pytest.fixture
def mock_to_csv():
    with patch("pandas.DataFrame.to_csv") as mock:
        yield mock


@pytest.fixture
def mock_train_test_split():
    with patch("src.data_preprocessing.train_test_splitter.train_test_split") as mock:
        yield mock


class TestTrainTestValidationSplit:
    class TrainTestValidationSplitTestCase(NamedTuple):
        description: str
        csv_file_path: Path
        test_size: float
        validation_size: float
        first_split: float
        second_split: float
        destination_train_csv: Path
        destination_validation_csv: Path
        destination_test_csv: Path
        input_data: pd.DataFrame
        X_train: list
        y_train: list
        X_temp: list
        y_temp: list
        X_val: list
        y_val: list
        X_test: list
        y_test: list
        expected_to_csv_calls: list

    class TrainTestValidationSplitExceptionTestCase(NamedTuple):
        description: str
        csv_file_path: Path
        test_size: float
        validation_size: float
        destination_train_csv: Path
        destination_validation_csv: Path
        destination_test_csv: Path
        input_data: pd.DataFrame
        expected_exception: type[Exception]
        expected_message: str

    TRAIN_TEST_VALIDATION_SPLIT_CASES = [
        TrainTestValidationSplitTestCase(
            description="Valid data splitting",
            csv_file_path=Path("test_input.csv"),
            test_size=0.2,
            validation_size=0.2,
            first_split=0.4,
            second_split=0.5,
            destination_train_csv=Path("train.csv"),
            destination_validation_csv=Path("validation.csv"),
            destination_test_csv=Path("test.csv"),
            input_data=pd.DataFrame(
                {
                    "Sequence": [
                        "ATCG",
                        "GCTA",
                        "TACG",
                        "CGTA",
                        "GTAC",
                        "AGTC",
                        "TGCA",
                        "CATG",
                        "ACGT",
                        "CTAG",
                    ],
                    "IsGene": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                }
            ),
            X_train=["ATCG", "GCTA", "TACG", "CGTA", "GTAC", "AGTC"],
            y_train=[1, 0, 1, 0, 1, 0],
            X_temp=["TGCA", "CATG", "ACGT", "CTAG"],
            y_temp=[1, 0, 1, 0],
            X_val=["TGCA", "CATG"],
            y_val=[1, 0],
            X_test=["ACGT", "CTAG"],
            y_test=[1, 0],
            expected_to_csv_calls=[
                (Path("train.csv"), False),
                (Path("validation.csv"), False),
                (Path("test.csv"), False),
            ],
        ),
        TrainTestValidationSplitTestCase(
            description="Valid data splitting with 5 sequences",
            csv_file_path=Path("test_input.csv"),
            test_size=0.2,
            validation_size=0.2,
            first_split=0.4,
            second_split=0.5,
            destination_train_csv=Path("train.csv"),
            destination_validation_csv=Path("validation.csv"),
            destination_test_csv=Path("test.csv"),
            input_data=pd.DataFrame(
                {
                    "Sequence": ["ATCG", "GCTA", "TACG", "CGTA", "GTAC"],
                    "IsGene": [1, 0, 1, 0, 1],
                }
            ),
            X_train=["ATCG", "GCTA"],
            y_train=[1, 0],
            X_temp=["TACG", "CGTA", "GTAC"],
            y_temp=[1, 0, 1],
            X_val=["TACG"],
            y_val=[1],
            X_test=["CGTA", "GTAC"],
            y_test=[0, 1],
            expected_to_csv_calls=[
                (Path("train.csv"), False),
                (Path("validation.csv"), False),
                (Path("test.csv"), False),
            ],
        ),
        TrainTestValidationSplitTestCase(
            description="Valid data splitting with 50 sequences",
            csv_file_path=Path("test_input.csv"),
            test_size=0.1,
            validation_size=0.5,
            first_split=0.6,
            second_split=0.16666666666666669,
            destination_train_csv=Path("train.csv"),
            destination_validation_csv=Path("validation.csv"),
            destination_test_csv=Path("test.csv"),
            input_data=pd.DataFrame(
                {
                    "Sequence": [f"Seq{i}" for i in range(50)],
                    "IsGene": [i % 2 for i in range(50)],
                }
            ),
            X_train=[f"Seq{i}" for i in range(20)],
            y_train=[i % 2 for i in range(20)],
            X_temp=[f"Seq{i}" for i in range(20, 50)],
            y_temp=[i % 2 for i in range(20, 50)],
            X_val=[f"Seq{i}" for i in range(20, 30)],
            y_val=[i % 2 for i in range(20, 30)],
            X_test=[f"Seq{i}" for i in range(30, 50)],
            y_test=[i % 2 for i in range(30, 50)],
            expected_to_csv_calls=[
                (Path("train.csv"), False),
                (Path("validation.csv"), False),
                (Path("test.csv"), False),
            ],
        ),
    ]

    TRAIN_TEST_VALIDATION_SPLIT_EXCEPTION_CASES = [
        TrainTestValidationSplitExceptionTestCase(
            description="Invalid CSV format, no 'IsGene' column",
            csv_file_path=Path("invalid_input.csv"),
            test_size=0.2,
            validation_size=0.2,
            destination_train_csv=Path("train.csv"),
            destination_validation_csv=Path("validation.csv"),
            destination_test_csv=Path("test.csv"),
            input_data=pd.DataFrame(
                {
                    "Sequence": [f"Seq{i}" for i in range(5)],
                }
            ),
            expected_exception=ValueError,
            expected_message="Input CSV must contain 'Sequence' and 'IsGene' columns.",
        ),
        TrainTestValidationSplitExceptionTestCase(
            description="Invalid CSV format, no 'Sequence' column",
            csv_file_path=Path("invalid_input.csv"),
            test_size=0.2,
            validation_size=0.2,
            destination_train_csv=Path("train.csv"),
            destination_validation_csv=Path("validation.csv"),
            destination_test_csv=Path("test.csv"),
            input_data=pd.DataFrame(
                {
                    "IsGene": [i % 2 for i in range(5)],
                }
            ),
            expected_exception=ValueError,
            expected_message="Input CSV must contain 'Sequence' and 'IsGene' columns.",
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        TRAIN_TEST_VALIDATION_SPLIT_CASES,
        ids=[case.description for case in TRAIN_TEST_VALIDATION_SPLIT_CASES],
    )
    @patch("src.data_preprocessing.train_test_splitter.save_train_test_validation_csvs")
    def test_train_test_validation_split(
        self,
        mock_save_train_test_validation_csvs: MagicMock,
        mock_read_csv: MagicMock,
        mock_train_test_split: MagicMock,
        test_case: TrainTestValidationSplitTestCase,
    ):
        # Arrange
        mock_read_csv.return_value = test_case.input_data
        mock_train_test_split.side_effect = [
            (test_case.X_train, test_case.X_temp, test_case.y_train, test_case.y_temp),
            (test_case.X_val, test_case.X_test, test_case.y_val, test_case.y_test),
        ]

        # Act & Assert
        train_test_validation_split(
            csv_file_path=test_case.csv_file_path,
            test_size=test_case.test_size,
            validation_size=test_case.validation_size,
            destination_train_csv=test_case.destination_train_csv,
            destination_validation_csv=test_case.destination_validation_csv,
            destination_test_csv=test_case.destination_test_csv,
        )

        # Assert

        calls = [
            call(
                test_case.input_data["Sequence"].tolist(),
                test_case.input_data["IsGene"].values,
                test_size=test_case.first_split,
                random_state=42,
            ),
            call(
                test_case.X_temp,
                test_case.y_temp,
                test_size=test_case.second_split,
                random_state=42,
            ),
        ]
        mock_train_test_split.assert_has_calls(calls, any_order=False)
        mock_read_csv.assert_called_once_with(test_case.csv_file_path)
        mock_save_train_test_validation_csvs.assert_called_once_with(
            data_splits={
                "X_train": test_case.X_train,
                "y_train": test_case.y_train,
                "X_val": test_case.X_val,
                "y_val": test_case.y_val,
                "X_test": test_case.X_test,
                "y_test": test_case.y_test,
            },
            destination_train_csv=test_case.destination_train_csv,
            destination_validation_csv=test_case.destination_validation_csv,
            destination_test_csv=test_case.destination_test_csv,
        )

    @pytest.mark.parametrize(
        "test_case",
        TRAIN_TEST_VALIDATION_SPLIT_EXCEPTION_CASES,
        ids=[case.description for case in TRAIN_TEST_VALIDATION_SPLIT_EXCEPTION_CASES],
    )
    def test_train_test_validation_split_exception(
        self,
        mock_read_csv: MagicMock,
        test_case: TrainTestValidationSplitExceptionTestCase,
    ):
        # Arrange
        mock_read_csv.return_value = test_case.input_data

        # Act & Assert
        with pytest.raises(
            test_case.expected_exception, match=test_case.expected_message
        ):
            train_test_validation_split(
                csv_file_path=test_case.csv_file_path,
                test_size=test_case.test_size,
                validation_size=test_case.validation_size,
                destination_train_csv=test_case.destination_train_csv,
                destination_validation_csv=test_case.destination_validation_csv,
                destination_test_csv=test_case.destination_test_csv,
            )


class TestSaveTrainTestValidationCsvs:
    class SaveTrainTestValidationCsvsTestCase(NamedTuple):
        description: str
        data_splits: dict
        destination_train_csv: Path
        destination_validation_csv: Path
        destination_test_csv: Path
        expected_dataframe_calls: list

    SAVE_TRAIN_TEST_VALIDATION_CSVS_CASES = [
        SaveTrainTestValidationCsvsTestCase(
            description="Valid data saving",
            data_splits={
                "X_train": ["ATCG", "GCTA"],
                "y_train": [1, 0],
                "X_val": ["TACG"],
                "y_val": [1],
                "X_test": ["CGTA"],
                "y_test": [0],
            },
            destination_train_csv=Path("train.csv"),
            destination_validation_csv=Path("validation.csv"),
            destination_test_csv=Path("test.csv"),
            expected_dataframe_calls=[
                call(
                    {
                        "Sequence": ["ATCG", "GCTA"],
                        "IsGene": [1, 0],
                    },
                ),
                call().to_csv(Path("train.csv"), index=False),
                call(
                    {
                        "Sequence": ["TACG"],
                        "IsGene": [1],
                    },
                ),
                call().to_csv(Path("validation.csv"), index=False),
                call(
                    {
                        "Sequence": ["CGTA"],
                        "IsGene": [0],
                    },
                ),
                call().to_csv(Path("test.csv"), index=False),
            ],
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        SAVE_TRAIN_TEST_VALIDATION_CSVS_CASES,
        ids=[case.description for case in SAVE_TRAIN_TEST_VALIDATION_CSVS_CASES],
    )
    @patch("src.data_preprocessing.train_test_splitter.pd.DataFrame")
    def test_save_train_test_validation_csvs(
        self,
        mock_dataframe_constructor: MagicMock,
        test_case: SaveTrainTestValidationCsvsTestCase,
    ):
        # Act
        save_train_test_validation_csvs(
            data_splits=test_case.data_splits,
            destination_train_csv=test_case.destination_train_csv,
            destination_validation_csv=test_case.destination_validation_csv,
            destination_test_csv=test_case.destination_test_csv,
        )

        # Assert
        mock_dataframe_constructor.assert_has_calls(
            test_case.expected_dataframe_calls, any_order=False
        )
