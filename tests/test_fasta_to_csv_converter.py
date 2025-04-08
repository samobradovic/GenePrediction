from pathlib import Path
from typing import List, NamedTuple
from unittest.mock import MagicMock, patch

import pytest

from src.data_preprocessing.fasta_to_csv_converter import convert_fasta_folder_into_csv


@pytest.fixture
def mock_get_string_sequences_from_directory():
    with patch(
        "src.data_preprocessing.fasta_to_csv_converter.get_string_sequences_from_directory"
    ) as mock:
        yield mock


@pytest.fixture
def mock_write_string_sequences_to_csv():
    with patch(
        "src.data_preprocessing.fasta_to_csv_converter.write_string_sequences_to_csv"
    ) as mock:
        yield mock


class TestConvertFastaFolderIntoCsv:
    class ConversionTestCase(NamedTuple):
        description: str
        sequences: List[str]
        source_directory: Path
        csv_destination_path: Path

    class ExceptionTestCase(NamedTuple):
        description: str
        mock_function: str
        side_effect: Exception
        expected_exception: type[Exception]
        expected_message: str

    CONVERSION_VALID_CASES: List[ConversionTestCase] = [
        ConversionTestCase(
            description="Valid conversion with multiple sequences",
            sequences=["ATCG", "GCTA", "GCTA", "GCTA", "GCTA", "GCTA"],
            source_directory=Path("test_source"),
            csv_destination_path=Path("test_destination.csv"),
        ),
        ConversionTestCase(
            description="Valid conversion with single sequence",
            sequences=["ATCG"],
            source_directory=Path("test_source"),
            csv_destination_path=Path("test_destination.csv"),
        ),
        ConversionTestCase(
            description="Valid conversion with empty sequence list",
            sequences=[],
            source_directory=Path("test_source"),
            csv_destination_path=Path("test_destination.csv"),
        ),
    ]

    EXCEPTION_TEST_CASES: List[ExceptionTestCase] = [
        ExceptionTestCase(
            description="get_string_sequences_from_directory raises Exception",
            mock_function="mock_get_string_sequences_from_directory",
            side_effect=Exception("Test exception"),
            expected_exception=Exception,
            expected_message="Test exception",
        ),
        ExceptionTestCase(
            description="write_string_sequences_to_csv raises Exception",
            mock_function="mock_write_string_sequences_to_csv",
            side_effect=Exception("Test exception"),
            expected_exception=Exception,
            expected_message="Test exception",
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        CONVERSION_VALID_CASES,
        ids=[case.description for case in CONVERSION_VALID_CASES],
    )
    def test_convert_fasta_folder_into_csv_valid(
        self,
        mock_get_string_sequences_from_directory: MagicMock,
        mock_write_string_sequences_to_csv: MagicMock,
        test_case: ConversionTestCase,
    ):
        # Arrange
        mock_get_string_sequences_from_directory.return_value = test_case.sequences
        source_directory = test_case.source_directory
        csv_destination_path = test_case.csv_destination_path

        # Act
        convert_fasta_folder_into_csv(source_directory, csv_destination_path)

        # Assert
        mock_get_string_sequences_from_directory.assert_called_once_with(
            source_directory
        )
        mock_write_string_sequences_to_csv.assert_called_once_with(
            sequences=test_case.sequences,
            csv_destination_path=csv_destination_path,
            write_header=True,
            is_real_gene=True,
        )

    @pytest.mark.parametrize(
        "test_case",
        [
            case
            for case in EXCEPTION_TEST_CASES
            if case.mock_function == "mock_get_string_sequences_from_directory"
        ],
        ids=[
            case.description
            for case in EXCEPTION_TEST_CASES
            if case.mock_function == "mock_get_string_sequences_from_directory"
        ],
    )
    def test_convert_fasta_folder_into_csv_fasta_helper_exception(
        self,
        mock_get_string_sequences_from_directory: MagicMock,
        mock_write_string_sequences_to_csv: MagicMock,
        test_case: ExceptionTestCase,
    ):
        # Arrange
        mock_get_string_sequences_from_directory.side_effect = test_case.side_effect

        source_directory = Path("test_source")
        csv_destination_path = Path("test_destination.csv")

        # Act & Assert
        with pytest.raises(
            test_case.expected_exception, match=test_case.expected_message
        ):
            convert_fasta_folder_into_csv(source_directory, csv_destination_path)

        mock_get_string_sequences_from_directory.assert_called_once_with(
            source_directory
        )
        mock_write_string_sequences_to_csv.assert_not_called()

    @pytest.mark.parametrize(
        "test_case",
        [
            case
            for case in EXCEPTION_TEST_CASES
            if case.mock_function == "mock_write_string_sequences_to_csv"
        ],
        ids=[
            case.description
            for case in EXCEPTION_TEST_CASES
            if case.mock_function == "mock_write_string_sequences_to_csv"
        ],
    )
    def test_convert_fasta_folder_into_csv_csv_helper_exception(
        self,
        mock_get_string_sequences_from_directory: MagicMock,
        mock_write_string_sequences_to_csv: MagicMock,
        test_case: ExceptionTestCase,
    ):
        # Arrange
        sequences = ["seq1", "seq2"]
        mock_write_string_sequences_to_csv.side_effect = test_case.side_effect
        mock_get_string_sequences_from_directory.return_value = sequences

        source_directory = Path("test_source")
        csv_destination_path = Path("test_destination.csv")

        # Act & Assert
        with pytest.raises(
            test_case.expected_exception, match=test_case.expected_message
        ):
            convert_fasta_folder_into_csv(source_directory, csv_destination_path)

        mock_get_string_sequences_from_directory.assert_called_once_with(
            source_directory
        )
        mock_write_string_sequences_to_csv.assert_called_once_with(
            sequences=sequences,
            csv_destination_path=csv_destination_path,
            write_header=True,
            is_real_gene=True,
        )
