from pathlib import Path
from typing import List, NamedTuple
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data_preprocessing.data_cleanup import clean_data


@pytest.fixture
def mock_read_csv():
    with patch("pandas.read_csv") as mock:
        yield mock


@pytest.fixture
def mock_to_csv():
    with patch("pandas.DataFrame.to_csv") as mock:
        yield mock


class TestCleanData:
    class CleanDataTestCase(NamedTuple):
        description: str
        source_path: Path
        destination_path: Path
        input_data: pd.DataFrame
        min_length: int
        max_length: int
        expected_output: pd.DataFrame

    class CleanDataExceptionTestCase(NamedTuple):
        description: str
        source_path: Path
        destination_path: Path
        input_data: pd.DataFrame | None
        min_length: int
        max_length: int
        expected_exception: type[Exception]
        expected_message: str

    CLEAN_DATA_VALID_CASES: List[CleanDataTestCase] = [
        CleanDataTestCase(
            description="Valid data with duplicate sequences removed",
            source_path=Path("test_source.csv"),
            destination_path=Path("test_destination.csv"),
            input_data=pd.DataFrame({"Sequence": ["ATCG", "GCTA", "ATCG", "TACG"]}),
            min_length=3,
            max_length=5,
            expected_output=pd.DataFrame({"Sequence": ["ATCG", "GCTA", "TACG"]}),
        ),
        CleanDataTestCase(
            description="Valid data with no duplicates and within length range",
            source_path=Path("test_source.csv"),
            destination_path=Path("test_destination.csv"),
            input_data=pd.DataFrame({"Sequence": ["ATCG", "GCTA"]}),
            min_length=3,
            max_length=5,
            expected_output=pd.DataFrame({"Sequence": ["ATCG", "GCTA"]}),
        ),
        CleanDataTestCase(
            description="Valid data with only long sequences",
            source_path=Path("test_source.csv"),
            destination_path=Path("test_destination.csv"),
            input_data=pd.DataFrame({"Sequence": ["A" * 50, "G" * 50]}),
            min_length=3,
            max_length=10,
            expected_output=pd.DataFrame(),
        ),
        CleanDataTestCase(
            description="Valid data with only short sequences",
            source_path=Path("test_source.csv"),
            destination_path=Path("test_destination.csv"),
            input_data=pd.DataFrame({"Sequence": ["A", "G"]}),
            min_length=3,
            max_length=5,
            expected_output=pd.DataFrame(),
        ),
        CleanDataTestCase(
            description="Valid data with some sequences too short and some too long",
            source_path=Path("test_source.csv"),
            destination_path=Path("test_destination.csv"),
            input_data=pd.DataFrame(
                {"Sequence": ["ATCGATATA", "ATCG", "GCTA", "A", "G"]}
            ),
            min_length=3,
            max_length=5,
            expected_output=pd.DataFrame({"Sequence": ["ATCG", "GCTA"]}),
        ),
        CleanDataTestCase(
            description="Valid data for length filtering and deduplication",
            source_path=Path("test_source.csv"),
            destination_path=Path("test_destination.csv"),
            input_data=pd.DataFrame(
                {"Sequence": ["AATTTAAAA", "A", "A", "ATCG", "TACG", "ATCG", "GGGG"]}
            ),
            min_length=3,
            max_length=5,
            expected_output=pd.DataFrame({"Sequence": ["ATCG", "TACG", "GGGG"]}),
        ),
        CleanDataTestCase(
            description="Valid data with unusual characters",
            source_path=Path("test_source.csv"),
            destination_path=Path("test_destination.csv"),
            input_data=pd.DataFrame({"Sequence": ["ATCG123", "GCTA!@#", "TACG$%^"]}),
            min_length=3,
            max_length=10,
            expected_output=pd.DataFrame(
                {"Sequence": ["ATCG123", "GCTA!@#", "TACG$%^"]}
            ),
        ),
    ]

    CLEAN_DATA_EXCEPTION_CASES: List[CleanDataExceptionTestCase] = [
        CleanDataExceptionTestCase(
            description="Missing 'Sequence' column",
            source_path=Path("test_source.csv"),
            destination_path=Path("test_destination.csv"),
            input_data=pd.DataFrame({"InvalidColumn": ["ATCG"]}),
            min_length=3,
            max_length=5,
            expected_exception=ValueError,
            expected_message="The input CSV must contain a 'Sequence' column.",
        ),
        CleanDataExceptionTestCase(
            description="Empty 'Sequence' column",
            source_path=Path("test_source.csv"),
            destination_path=Path("test_destination.csv"),
            input_data=pd.DataFrame({"Sequence": []}),
            min_length=3,
            max_length=5,
            expected_exception=ValueError,
            expected_message="The 'Sequence' column is empty.",
        ),
        CleanDataExceptionTestCase(
            description="min_length greater than max_length",
            source_path=Path("test_source.csv"),
            destination_path=Path("test_destination.csv"),
            input_data=None,
            min_length=10,
            max_length=5,
            expected_exception=ValueError,
            expected_message="min_length cannot be greater than max_length.",
        ),
        CleanDataExceptionTestCase(
            description="File not found",
            source_path=Path("non_existent_file.csv"),
            destination_path=Path("test_destination.csv"),
            input_data=None,
            min_length=3,
            max_length=5,
            expected_exception=FileNotFoundError,
            expected_message="No such file or directory",
        ),
        CleanDataExceptionTestCase(
            description="Empty data file",
            source_path=Path("empty_file.csv"),
            destination_path=Path("test_destination.csv"),
            input_data=None,
            min_length=3,
            max_length=5,
            expected_exception=pd.errors.EmptyDataError,
            expected_message="No columns to parse from file",
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        CLEAN_DATA_VALID_CASES,
        ids=[case.description for case in CLEAN_DATA_VALID_CASES],
    )
    def test_clean_data_valid(
        self,
        mock_read_csv: MagicMock,
        mock_to_csv: MagicMock,
        test_case: CleanDataTestCase,
    ):
        # Arrange
        mock_read_csv.return_value = test_case.input_data

        # Act
        result = clean_data(
            source_path=test_case.source_path,
            destination_path=test_case.destination_path,
            min_length=test_case.min_length,
            max_length=test_case.max_length,
        )

        # Assert
        mock_read_csv.assert_called_once_with(test_case.source_path)
        mock_to_csv.assert_called_once_with(test_case.destination_path, index=False)

        if test_case.expected_output.empty:
            assert result.empty
        else:
            pd.testing.assert_frame_equal(result, test_case.expected_output)

    @pytest.mark.parametrize(
        "test_case",
        CLEAN_DATA_EXCEPTION_CASES,
        ids=[case.description for case in CLEAN_DATA_EXCEPTION_CASES],
    )
    def test_clean_data_exceptions(
        self, mock_read_csv: MagicMock, test_case: CleanDataExceptionTestCase
    ):
        # Arrange
        if test_case.input_data is not None:
            mock_read_csv.return_value = test_case.input_data
        elif test_case.expected_exception is FileNotFoundError:
            mock_read_csv.side_effect = FileNotFoundError(test_case.expected_message)
        elif test_case.expected_exception is pd.errors.EmptyDataError:
            mock_read_csv.side_effect = pd.errors.EmptyDataError(
                test_case.expected_message
            )

        # Act & Assert
        with pytest.raises(
            test_case.expected_exception, match=test_case.expected_message
        ):
            clean_data(
                source_path=test_case.source_path,
                destination_path=test_case.destination_path,
                min_length=test_case.min_length,
                max_length=test_case.max_length,
            )
