from pathlib import Path
from typing import List, NamedTuple
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.helpers.csv_helper import (
    copy_csv_file,
    get_iterable_series_sequence_lengths_from_csv,
    write_string_sequences_to_csv,
)


@pytest.fixture
def mock_open():
    with patch("builtins.open", new_callable=MagicMock) as mock:
        yield mock


@pytest.fixture
def mock_read_csv():
    with patch("pandas.read_csv") as mock:
        yield mock


@pytest.fixture
def mock_to_csv():
    with patch("pandas.DataFrame.to_csv") as mock:
        yield mock


class TestWriteStringSequencesToCsv:
    class WriteStringSequencesTestCase(NamedTuple):
        description: str
        sequences: List[str]
        csv_path: Path
        write_header: bool
        is_real_gene: bool
        expected_calls: List[str]

    TEST_CASES = [
        WriteStringSequencesTestCase(
            description="Write sequences with header",
            sequences=["ATCG", "GCTA"],
            csv_path=Path("test.csv"),
            write_header=True,
            is_real_gene=True,
            expected_calls=["IsGene,Sequence\n", "1,ATCG\n", "1,GCTA\n"],
        ),
        WriteStringSequencesTestCase(
            description="Append sequences without header",
            sequences=["ATCG", "GCTA"],
            csv_path=Path("test.csv"),
            write_header=False,
            is_real_gene=False,
            expected_calls=["0,ATCG\n", "0,GCTA\n"],
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        TEST_CASES,
        ids=[case.description for case in TEST_CASES],
    )
    def test_write_string_sequences_to_csv(self, mock_open, test_case):
        # Act
        write_string_sequences_to_csv(
            test_case.sequences,
            test_case.csv_path,
            test_case.write_header,
            test_case.is_real_gene,
        )

        # Assert
        mode = "w" if test_case.write_header else "a"
        mock_open.assert_called_once_with(test_case.csv_path, mode)
        for call in test_case.expected_calls:
            mock_open().write.assert_any_call(call)


class TestCopyCsvFile:
    class CopyCsvFileTestCase(NamedTuple):
        description: str
        source_csv: Path
        destination_csv: Path
        input_data: pd.DataFrame

    TEST_CASES = [
        CopyCsvFileTestCase(
            description="Copy valid CSV file",
            source_csv=Path("source.csv"),
            destination_csv=Path("destination.csv"),
            input_data=pd.DataFrame({"Column": [1, 2, 3]}),
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        TEST_CASES,
        ids=[case.description for case in TEST_CASES],
    )
    def test_copy_csv_file(self, mock_read_csv, mock_to_csv, test_case):
        # Arrange
        mock_read_csv.return_value = test_case.input_data

        # Act
        copy_csv_file(test_case.source_csv, test_case.destination_csv)

        # Assert
        mock_read_csv.assert_called_once_with(test_case.source_csv)
        mock_to_csv.assert_called_once_with(test_case.destination_csv, index=False)


class TestGetIterableSeriesSequenceLengthsFromCsv:
    class GetLengthsTestCase(NamedTuple):
        description: str
        csv_file: Path
        input_data: pd.DataFrame
        expected_output: List[int]
        raises: type[Exception] | None

    TEST_CASES = [
        GetLengthsTestCase(
            description="Valid CSV with sequences",
            csv_file=Path("test.csv"),
            input_data=pd.DataFrame({"Sequence": ["ATCG", "GCTA"]}),
            expected_output=[4, 4],
            raises=None,
        ),
        GetLengthsTestCase(
            description="CSV missing 'Sequence' column",
            csv_file=Path("test.csv"),
            input_data=pd.DataFrame({"OtherColumn": ["ATCG", "GCTA"]}),
            expected_output=[],
            raises=ValueError,
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        TEST_CASES,
        ids=[case.description for case in TEST_CASES],
    )
    def test_get_iterable_series_sequence_lengths_from_csv(
        self, mock_read_csv, test_case
    ):
        # Arrange
        mock_read_csv.return_value = test_case.input_data

        # Act & Assert
        if test_case.raises:
            with pytest.raises(test_case.raises):
                get_iterable_series_sequence_lengths_from_csv(test_case.csv_file)
        else:
            result = get_iterable_series_sequence_lengths_from_csv(test_case.csv_file)
            assert result.tolist() == test_case.expected_output
