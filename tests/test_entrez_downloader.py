import os
from pathlib import Path
from socket import timeout
from typing import List, NamedTuple
from unittest.mock import MagicMock, call, mock_open, patch
from urllib.error import HTTPError, URLError

import pytest

from src.data_loading.entrez_downloader import (
    download_batch,
    download_data,
    get_ids_from_file,
)


@pytest.fixture
def mock_entrez_read():
    with patch("src.data_loading.entrez_downloader.Entrez.read") as mock_read:
        yield mock_read


@pytest.fixture
def mock_entrez_esearch():
    with patch("src.data_loading.entrez_downloader.Entrez.esearch") as mock_esearch:
        yield mock_esearch


@pytest.fixture
def mock_entrez_efetch():
    with patch("src.data_loading.entrez_downloader.Entrez.efetch") as mock_efetch:
        yield mock_efetch


@pytest.fixture
def mock_get_folder_size():
    with patch("src.data_loading.entrez_downloader.get_folder_size") as mock_size:
        yield mock_size


@pytest.fixture
def mock_sleep():
    with patch("time.sleep", return_value=None) as mock:
        yield mock


@pytest.fixture
def mock_file_exists():
    with patch("src.data_loading.entrez_downloader.os.path.exists") as mock:
        yield mock


@pytest.fixture
def mock_open_file():
    with patch("builtins.open", new_callable=mock_open) as mock:
        yield mock


class TestGetIdsFromFile:
    class CaseData(NamedTuple):
        description: str
        file_content: str | None
        file_path: Path
        expected_ids: List[str] | None = None
        expected_exception: type[Exception] | None = None
        open_side_effect: Exception | None = None

    TEST_CASES = [
        CaseData(
            description="File with multiple IDs (normal case)",
            file_content="id1\nid2\nid3\n",
            file_path=Path("test_ids.txt"),
            expected_ids=["id1", "id2", "id3"],
        ),
        CaseData(
            description="File with single ID",
            file_content="id1\n",
            file_path=Path("test_ids.txt"),
            expected_ids=["id1"],
        ),
        CaseData(
            description="File with empty lines and whitespace",
            file_content="id1\n\n  \nid2\n\t\nid3\n",
            file_path=Path("test_ids.txt"),
            expected_ids=["id1", "", "  ", "id2", "\t", "id3"],
        ),
        CaseData(
            description="Empty file",
            file_content="",
            file_path=Path("empty.txt"),
            expected_ids=[],
        ),
    ]

    ERROR_TEST_CASES = [
        CaseData(
            description="File not found",
            file_content=None,
            file_path=Path("notfound.txt"),
            expected_exception=FileNotFoundError,
            open_side_effect=FileNotFoundError("File not found"),
        ),
        CaseData(
            description="Permission denied error",
            file_content=None,
            file_path=Path("forbidden.txt"),
            expected_exception=PermissionError,
            open_side_effect=PermissionError("Permission denied"),
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        TEST_CASES,
        ids=[case.description for case in TEST_CASES],
    )
    def test_get_ids_from_file_success(
        self,
        mock_open_file: MagicMock,
        test_case: CaseData,
    ):
        # Arrange
        mock_open_file().read.return_value = test_case.file_content

        # Act
        result = get_ids_from_file(test_case.file_path)

        # Assert
        assert result == test_case.expected_ids

    @pytest.mark.parametrize(
        "test_case",
        ERROR_TEST_CASES,
        ids=[case.description for case in ERROR_TEST_CASES],
    )
    def test_get_ids_from_file_errors(
        self,
        mock_open_file: MagicMock,
        test_case: CaseData,
    ):
        # Arrange
        mock_open_file.side_effect = test_case.open_side_effect

        # Act & Assert
        assert test_case.expected_exception is not None
        with pytest.raises(test_case.expected_exception):
            get_ids_from_file(test_case.file_path)


class TestDownloadData:
    class CaseData(NamedTuple):
        description: str
        folder_path: Path
        max_size_gb: int
        batch_size: int
        file_format: str
        batch_retry_count: int
        batch_delay_seconds: int
        ids: List[str]
        folder_size_gb: float
        expected_batch_count: int

    TEST_CASES: List[CaseData] = [
        CaseData(
            description="Small dataset, folder size within limit",
            folder_path=Path("test_folder"),
            max_size_gb=1,
            batch_size=5,
            file_format="testfileformat",
            batch_retry_count=3,
            batch_delay_seconds=1,
            ids=["id1", "id2", "id3", "id4", "id5", "id6", "id7", "id8", "id9", "id10"],
            folder_size_gb=0.5,
            expected_batch_count=2,
        ),
        CaseData(
            description="Folder size exceeds limit",
            folder_path=Path("test_folder"),
            max_size_gb=1,
            batch_size=5,
            file_format="testfileformat",
            batch_retry_count=3,
            batch_delay_seconds=1,
            ids=["id1", "id2", "id3", "id4", "id5", "id6", "id7", "id8", "id9", "id10"],
            folder_size_gb=1.5,
            expected_batch_count=0,
        ),
        CaseData(
            description="Empty ID list",
            folder_path=Path("test_folder"),
            max_size_gb=1,
            batch_size=5,
            file_format="testfileformat",
            batch_retry_count=3,
            batch_delay_seconds=1,
            ids=[],
            folder_size_gb=0.5,
            expected_batch_count=0,
        ),
        CaseData(
            description="Batch size larger than ID list",
            folder_path=Path("test_folder"),
            max_size_gb=1,
            batch_size=20,
            file_format="testfileformat",
            batch_retry_count=3,
            batch_delay_seconds=1,
            ids=["id1", "id2", "id3"],
            folder_size_gb=0.5,
            expected_batch_count=1,
        ),
        CaseData(
            description="Batch size of 1 (should download each ID separately)",
            folder_path=Path("test_folder"),
            max_size_gb=1,
            batch_size=1,
            file_format="testfileformat",
            batch_retry_count=3,
            batch_delay_seconds=1,
            ids=["id1", "id2", "id3"],
            folder_size_gb=0.5,
            expected_batch_count=3,
        ),
        CaseData(
            description="Batch size not divisible by number of IDs",
            folder_path=Path("test_folder"),
            max_size_gb=1,
            batch_size=2,
            file_format="testfileformat",
            batch_retry_count=3,
            batch_delay_seconds=1,
            ids=["id1", "id2", "id3"],
            folder_size_gb=0.5,
            expected_batch_count=2,
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        TEST_CASES,
        ids=[case.description for case in TEST_CASES],
    )
    @patch("src.data_loading.entrez_downloader.download_batch")
    @patch("os.makedirs")
    def test_download_data(
        self,
        mock_makedirs: MagicMock,
        mock_download_batch: MagicMock,
        mock_get_folder_size: MagicMock,
        test_case: CaseData,
    ):
        # Arrange
        mock_get_folder_size.return_value = test_case.folder_size_gb

        # Act
        download_data(
            ids=test_case.ids,
            folder_path=test_case.folder_path,
            max_size_gb=test_case.max_size_gb,
            batch_size=test_case.batch_size,
            file_format=test_case.file_format,
            batch_retry_count=test_case.batch_retry_count,
            batch_delay_seconds=test_case.batch_delay_seconds,
        )

        # Assert
        mock_makedirs.assert_called_once_with(test_case.folder_path, exist_ok=True)
        assert mock_download_batch.call_count == test_case.expected_batch_count

        if test_case.expected_batch_count > 0:
            calls = []
            for i in range(test_case.expected_batch_count):
                batch_ids = test_case.ids[
                    i * test_case.batch_size : (i + 1) * test_case.batch_size
                ]
                batch_number = i + 1
                file_path = os.path.join(
                    test_case.folder_path,
                    f"batch_{batch_number}.{test_case.file_format}",
                )
                calls.append(
                    call(
                        file_path,
                        batch_ids,
                        batch_number,
                        test_case.batch_retry_count,
                        test_case.batch_delay_seconds,
                        test_case.file_format,
                    )
                )
            mock_download_batch.assert_has_calls(calls, any_order=False)
        else:
            mock_download_batch.assert_not_called()


class TestDownloadBatch:
    class CaseData(NamedTuple):
        description: str
        batch_ids: List[str]
        retry_count: int
        delay_seconds: int
        file_exists: bool = False
        expected_efetch_calls: int = 0
        expected_write_calls: int = 0
        expected_sleep_calls: int = 0
        efetch_side_effect: Exception | None = None
        efetch_side_effects: list | None = None
        expected_exception: type[Exception] | None = None

    TEST_CASES: List[CaseData] = [
        CaseData(
            description="File already exists, no calls to efetch or write",
            file_exists=True,
            batch_ids=["id1", "id2", "id3"],
            retry_count=1,
            delay_seconds=1,
            expected_efetch_calls=0,
            expected_write_calls=0,
        ),
        CaseData(
            description="Successful download with no retries",
            file_exists=False,
            batch_ids=["id1", "id2", "id3"],
            retry_count=1,
            delay_seconds=1,
            expected_efetch_calls=1,
            expected_write_calls=1,
        ),
        CaseData(
            description="Empty batch IDs, no calls to efetch or write",
            file_exists=False,
            batch_ids=[],
            retry_count=1,
            delay_seconds=1,
            expected_efetch_calls=0,
            expected_write_calls=0,
        ),
    ]

    ERROR_TEST_CASES: List[CaseData] = [
        CaseData(
            description="Retry on network error",
            batch_ids=["id1", "id2", "id3"],
            retry_count=10,
            delay_seconds=0,
            efetch_side_effect=URLError("Network unreachable"),
            expected_efetch_calls=10,
            expected_sleep_calls=9,
            expected_exception=URLError,
        ),
        CaseData(
            description="Retry on HTTP error",
            batch_ids=["id1", "id2", "id3"],
            retry_count=3,
            delay_seconds=0,
            efetch_side_effect=HTTPError(
                "http://test.com", 500, "Internal Server Error", MagicMock, None
            ),
            expected_efetch_calls=3,
            expected_sleep_calls=2,
            expected_exception=HTTPError,
        ),
        CaseData(
            description="Retry on timeout",
            batch_ids=["id1", "id2", "id3"],
            retry_count=3,
            delay_seconds=0,
            efetch_side_effect=timeout(),
            expected_efetch_calls=3,
            expected_sleep_calls=2,
            expected_exception=timeout,
        ),
        CaseData(
            description="No retries",
            batch_ids=["id1", "id2", "id3"],
            retry_count=1,
            delay_seconds=0,
            efetch_side_effect=timeout(),
            expected_efetch_calls=1,
            expected_sleep_calls=0,
            expected_exception=timeout,
        ),
    ]

    TEST_CASES_RETRY: List[CaseData] = [
        CaseData(
            description="Retry on network error 2 times and then succeed",
            batch_ids=["id1", "id2", "id3"],
            retry_count=3,
            delay_seconds=0,
            efetch_side_effects=[
                URLError("Network unreachable"),
                URLError("Network unreachable"),
                MagicMock(read=MagicMock(return_value="test_data")),
            ],
            expected_efetch_calls=3,
            expected_sleep_calls=2,
        ),
        CaseData(
            description="Retry on HTTP error 3 times and then succeed",
            batch_ids=["id1", "id2", "id3"],
            retry_count=10,
            delay_seconds=0,
            efetch_side_effects=[
                HTTPError(
                    "http://test.com",
                    500,
                    "Internal Server Error",
                    MagicMock,
                    None,
                ),
                HTTPError(
                    "http://test.com",
                    500,
                    "Internal Server Error",
                    MagicMock,
                    None,
                ),
                HTTPError(
                    "http://test.com",
                    500,
                    "Internal Server Error",
                    MagicMock,
                    None,
                ),
                MagicMock(read=MagicMock(return_value="test_data")),
            ],
            expected_efetch_calls=4,
            expected_sleep_calls=3,
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        TEST_CASES,
        ids=[case.description for case in TEST_CASES],
    )
    def test_download_batch_no_retries(
        self,
        mock_file_exists,
        mock_open_file,
        mock_entrez_efetch,
        test_case: CaseData,
    ):
        # Arrange
        mock_file_exists.return_value = test_case.file_exists
        mock_entrez_efetch.return_value = MagicMock()
        mock_entrez_efetch.return_value.read.return_value = "test_data"

        file_path = "test_file_path.fasta"
        batch_number = 1
        file_format = "fasta"

        # Act
        download_batch(
            file_path,
            test_case.batch_ids,
            batch_number,
            test_case.retry_count,
            test_case.delay_seconds,
            file_format,
        )

        # Assert
        assert mock_entrez_efetch.call_count == test_case.expected_efetch_calls
        if test_case.expected_efetch_calls == 1:
            mock_entrez_efetch.assert_called_once_with(
                db="nucleotide",
                id=",".join(test_case.batch_ids),
                rettype=file_format,
                retmode="text",
            )
        if test_case.expected_write_calls == 1:
            mock_open_file.assert_called_once_with(file_path, "w")
            mock_open_file().write.assert_called_once_with("test_data")
        else:
            mock_open_file.assert_not_called()

    @pytest.mark.parametrize(
        "test_case",
        ERROR_TEST_CASES,
        ids=[case.description for case in ERROR_TEST_CASES],
    )
    def test_download_batch_retries_on_exceptions(
        self,
        mock_file_exists,
        mock_sleep,
        mock_open_file,
        mock_entrez_efetch,
        test_case: CaseData,
    ):
        # Arrange
        mock_file_exists.return_value = False
        mock_entrez_efetch.side_effect = test_case.efetch_side_effect

        file_path = "test_file_path"
        file_format = "fasta"
        batch_number = 1

        # Act & Assert
        assert test_case.expected_exception is not None
        with pytest.raises(test_case.expected_exception):
            download_batch(
                file_path,
                test_case.batch_ids,
                batch_number,
                test_case.retry_count,
                test_case.delay_seconds,
                file_format,
            )

        # Assert
        assert mock_entrez_efetch.call_count == test_case.expected_efetch_calls
        assert mock_sleep.call_count == test_case.expected_sleep_calls
        if test_case.expected_efetch_calls > 0:
            expected_efetch_calls = [
                call(
                    db="nucleotide",
                    id=",".join(test_case.batch_ids),
                    rettype=file_format,
                    retmode="text",
                )
            ] * (test_case.expected_efetch_calls)
            mock_entrez_efetch.assert_has_calls(expected_efetch_calls, any_order=True)
        if test_case.expected_sleep_calls > 0:
            expected_sleep_calls = [call(test_case.delay_seconds)] * (
                test_case.expected_sleep_calls
            )
            mock_sleep.assert_has_calls(expected_sleep_calls, any_order=True)

        mock_open_file.assert_not_called()

    @pytest.mark.parametrize(
        "test_case",
        TEST_CASES_RETRY,
        ids=[case.description for case in TEST_CASES_RETRY],
    )
    def test_download_batch_retries_and_succeeds(
        self,
        mock_open_file,
        mock_file_exists,
        mock_sleep,
        mock_entrez_efetch,
        test_case: CaseData,
    ):
        # Arrange
        mock_file_exists.return_value = False
        mock_entrez_efetch.side_effect = test_case.efetch_side_effects

        file_path = "test_file_path"
        file_format = "fasta"
        batch_number = 1

        # Act
        download_batch(
            file_path,
            test_case.batch_ids,
            batch_number,
            test_case.retry_count,
            test_case.delay_seconds,
            file_format,
        )

        # Assert
        assert mock_entrez_efetch.call_count == test_case.expected_efetch_calls
        assert mock_sleep.call_count == test_case.expected_sleep_calls
        if test_case.expected_efetch_calls > 0:
            expected_efetch_calls = [
                call(
                    db="nucleotide",
                    id=",".join(test_case.batch_ids),
                    rettype=file_format,
                    retmode="text",
                )
            ] * (test_case.expected_efetch_calls)
            mock_entrez_efetch.assert_has_calls(expected_efetch_calls, any_order=False)
        if test_case.expected_sleep_calls > 0:
            expected_sleep_calls = [call(test_case.delay_seconds)] * (
                test_case.expected_sleep_calls
            )
            mock_sleep.assert_has_calls(expected_sleep_calls, any_order=False)
        mock_open_file.assert_called_once_with(file_path, "w")
        mock_open_file().write.assert_called_once_with("test_data")
