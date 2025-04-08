import os
from pathlib import Path
from socket import timeout
from typing import List, NamedTuple, Tuple
from unittest.mock import MagicMock, call, mock_open, patch
from urllib.error import HTTPError, URLError

import pytest

from src.data_loading.entrez_downloader import (
    download_batch,
    download_data,
    get_ids,
)


class GetIdsTestCase(NamedTuple):
    description: str
    max_results: int
    search_query: str
    mock_response: dict
    expected_ids: List[str]


class DownloadDataTestCase(NamedTuple):
    description: str
    number_of_datapoints: int
    search_query: str
    folder_path: Path
    max_size_gb: int
    batch_size: int
    file_format: str
    batch_retry_count: int
    batch_delay_seconds: int
    ids: List[str]
    folder_size_gb: float
    expected_batch_count: int


GET_IDS_VALID_CASES: List[GetIdsTestCase] = [
    GetIdsTestCase(
        description="Valid response with one ID",
        max_results=10,
        search_query="test_query",
        mock_response={"IdList": ["NM_001301717"]},
        expected_ids=["NM_001301717"],
    ),
    GetIdsTestCase(
        description="Valid full response with IDs",
        max_results=10,
        search_query="test_query",
        mock_response={
            "Count": "3",
            "RetMax": "10",
            "RetStart": "0",
            "IdList": ["NM_001301717", "NM_001301718", "NM_001301719"],
            "TranslationSet": [],
            "TranslationStack": [{"Term": "test", "Field": "All Fields", "Count": "3"}],
            "QueryTranslation": "test",
        },
        expected_ids=["NM_001301717", "NM_001301718", "NM_001301719"],
    ),
    GetIdsTestCase(
        description="Valid response with 50 IDs",
        max_results=10,
        search_query="test_query",
        mock_response={"IdList": [f"test_id_{i}" for i in range(0, 50)]},
        expected_ids=[f"test_id_{i}" for i in range(0, 50)],
    ),
    GetIdsTestCase(
        description="Valid response with 1,000,000 IDs",
        max_results=10,
        search_query="test_query",
        mock_response={"IdList": [f"test_id_{i}" for i in range(0, 1000000)]},
        expected_ids=[f"test_id_{i}" for i in range(0, 1000000)],
    ),
]


DOWNLOAD_DATA_VALID_CASES: List[DownloadDataTestCase] = [
    DownloadDataTestCase(
        description="Small dataset, folder size within limit",
        number_of_datapoints=10,
        search_query="test_query",
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
    DownloadDataTestCase(
        description="Folder size exceeds limit",
        number_of_datapoints=10,
        search_query="test_query",
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
    DownloadDataTestCase(
        description="Empty ID list",
        number_of_datapoints=10,
        search_query="test_query",
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
    DownloadDataTestCase(
        description="Batch size larger than ID list",
        number_of_datapoints=10,
        search_query="test_query",
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
    DownloadDataTestCase(
        description="Batch size of 1 (should download each ID separately)",
        number_of_datapoints=10,
        search_query="test_query",
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
    DownloadDataTestCase(
        description="Batch size not divisible by number of IDs",
        number_of_datapoints=10,
        search_query="test_query",
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


@pytest.mark.parametrize(
    "test_case",
    GET_IDS_VALID_CASES,
    ids=[case.description for case in GET_IDS_VALID_CASES],
)
def test_get_ids_valid(
    mock_entrez_esearch: MagicMock,
    mock_entrez_read: MagicMock,
    test_case: GetIdsTestCase,
):
    # Arrange
    mock_entrez_esearch.return_value = MagicMock()
    mock_entrez_read.return_value = test_case.mock_response

    # Act
    result: List[str] = get_ids(test_case.max_results, test_case.search_query)

    # Assert
    assert result == test_case.expected_ids


@pytest.mark.parametrize(
    "test_case",
    [
        # Invalid response format (missing IdList)
        (10, "test_query", {}, ValueError),
        # Invalid IdList format (non-string elements)
        (10, "test_query", {"IdList": [123, 456]}, ValueError),
        # Valid response but empty IdList
        (10, "test_query", {"IdList": []}, ValueError),
        # Valid response but missing IdList key
        (10, "test_query", {"Count": "3", "TranslationSet": []}, ValueError),
        # Invalid response format (not a dictionary)
        (10, "test_query", "test", ValueError),
    ],
)
def test_get_ids_with_invalid_responses(
    mock_entrez_esearch: MagicMock,
    mock_entrez_read: MagicMock,
    test_case: Tuple[int, str, dict, type[Exception]],
):
    # Arrange
    max_results: int
    search_query: str
    mock_response: dict
    expected_exception: type[Exception]

    max_results, search_query, mock_response, expected_exception = test_case

    mock_entrez_esearch.return_value = MagicMock()
    mock_entrez_read.return_value = mock_response

    # Act & Assert
    with pytest.raises(expected_exception):
        get_ids(max_results, search_query)


@pytest.mark.parametrize(
    "test_case",
    [
        # Simulate HTTPError
        (
            10,  # max_results
            "test_query",  # search_query
            HTTPError(
                "http://test.com", 500, "Internal Server Error", MagicMock, None
            ),  # side_effect
            HTTPError,  # expected_exception
        ),
        # Simulate URLError
        (
            10,
            "test_query",
            URLError("Network unreachable"),
            URLError,
        ),
        # Simulate Timeout
        (
            10,
            "test_query",
            timeout(),
            timeout,
        ),
    ],
)
def test_get_ids_with_exceptions(
    mock_entrez_esearch: MagicMock,
    mock_entrez_read: MagicMock,
    test_case: Tuple[int, str, Exception, type[Exception]],
):
    # Arrange
    max_results, search_query, side_effect, expected_exception = test_case
    mock_entrez_esearch.return_value = MagicMock()
    mock_entrez_read.side_effect = side_effect

    # Act & Assert
    with pytest.raises(expected_exception):
        get_ids(max_results, search_query)


@pytest.mark.parametrize(
    "test_case",
    [
        # Simulate HTTPError
        (
            10,
            "test_query",
            HTTPError("http://test.com", 500, "Internal Server Error", MagicMock, None),
            HTTPError,
        ),
        # Simulate URLError
        (10, "test_query", URLError("Network unreachable"), URLError),
        # Simulate Timeout
        (10, "test_query", timeout(), timeout),
    ],
)
def test_get_ids_external_exceptions(
    mock_entrez_esearch: MagicMock,
    test_case: Tuple[int, str, dict, type[Exception]],
):
    # Arrange
    max_results: int
    search_query: str
    mock_side_effect: dict
    expected_exception: type[Exception]

    max_results, search_query, mock_side_effect, expected_exception = test_case

    mock_entrez_esearch.side_effect = mock_side_effect

    # Act & Assert
    with pytest.raises(expected_exception):
        get_ids(max_results, search_query)


@pytest.mark.parametrize(
    "test_case",
    DOWNLOAD_DATA_VALID_CASES,
    ids=[case.description for case in DOWNLOAD_DATA_VALID_CASES],
)
@patch("src.data_loading.entrez_downloader.get_ids")
@patch("src.data_loading.entrez_downloader.download_batch")
@patch("os.makedirs")
def test_download_data(
    mock_makedirs: MagicMock,
    mock_download_batch: MagicMock,
    mock_get_ids: MagicMock,
    mock_get_folder_size: MagicMock,
    test_case: DownloadDataTestCase,
):
    # Arrange
    mock_get_ids.return_value = test_case.ids
    mock_get_folder_size.return_value = test_case.folder_size_gb

    # Act
    download_data(
        test_case.number_of_datapoints,
        test_case.search_query,
        test_case.folder_path,
        test_case.max_size_gb,
        test_case.batch_size,
        test_case.file_format,
        batch_retry_count=test_case.batch_retry_count,
        batch_delay_seconds=test_case.batch_delay_seconds,
    )

    # Assert
    mock_makedirs.assert_called_once_with(test_case.folder_path, exist_ok=True)
    mock_get_ids.assert_called_once_with(
        test_case.number_of_datapoints, test_case.search_query
    )
    assert mock_download_batch.call_count == test_case.expected_batch_count

    if test_case.expected_batch_count > 0:
        for i in range(test_case.expected_batch_count):
            batch_ids = test_case.ids[
                i * test_case.batch_size : (i + 1) * test_case.batch_size
            ]
            batch_number = i + 1
            file_path = os.path.join(
                test_case.folder_path, f"batch_{batch_number}.{test_case.file_format}"
            )
            mock_download_batch.assert_any_call(
                file_path,
                batch_ids,
                batch_number,
                test_case.batch_retry_count,
                test_case.batch_delay_seconds,
                test_case.file_format,
            )
    else:
        mock_download_batch.assert_not_called()


@pytest.mark.parametrize(
    "test_case",
    [
        # Case: File already exists, no calls to efetch or write
        (True, ["id1", "id2", "id3"], 1, 1, 0, 0),
        # Case: Successful download with no retries
        (False, ["id1", "id2", "id3"], 1, 1, 1, 1),
        # Case: Empty batch IDs, no calls to efetch or write
        (False, [], 1, 1, 0, 0),
    ],
)
def test_download_batch_no_retries(
    mock_file_exists,
    mock_open_file,
    mock_entrez_efetch,
    test_case,
):
    # Arrange
    (
        file_exists,
        batch_ids,
        retry_count,
        delay_seconds,
        expected_efetch_calls,
        expected_write_calls,
    ) = test_case
    mock_file_exists.return_value = file_exists
    mock_entrez_efetch.return_value = MagicMock()
    mock_entrez_efetch.return_value.read.return_value = "test_data"

    file_path = "test_file_path.fasta"
    batch_number = 1
    file_format = "fasta"

    # Act
    download_batch(
        file_path, batch_ids, batch_number, retry_count, delay_seconds, file_format
    )

    # Assert
    assert mock_entrez_efetch.call_count == expected_efetch_calls
    if expected_efetch_calls == 1:
        mock_entrez_efetch.assert_called_once_with(
            db="nucleotide",
            id=",".join(batch_ids),
            rettype=file_format,
            retmode="text",
        )
    if expected_write_calls == 1:
        mock_open_file.assert_called_once_with(file_path, "w")
        mock_open_file().write.assert_called_once_with("test_data")
    else:
        mock_open_file.assert_not_called()


@pytest.mark.parametrize(
    "test_case",
    [
        # Case: Retry on network error
        (
            ["id1", "id2", "id3"],  # batch_ids
            10,  # retry_count
            0,  # delay_seconds
            URLError("Network unreachable"),  # efetch_side_effect
            10,  # expected_efetch_calls
            9,  # expected_sleep_calls
            URLError,  # expected_exception
        ),
        # Case: Retry on HTTP error
        (
            ["id1", "id2", "id3"],
            3,
            0,
            HTTPError("http://test.com", 500, "Internal Server Error", MagicMock, None),
            3,
            2,
            HTTPError,
        ),
        # Case: Retry on timeout
        (
            ["id1", "id2", "id3"],
            3,
            0,
            timeout(),
            3,
            2,
            timeout,
        ),
        # Case: No retries
        (
            ["id1", "id2", "id3"],
            1,
            0,
            timeout(),
            1,
            0,
            timeout,
        ),
    ],
)
def test_download_batch_retries_on_exceptions(
    mock_file_exists,
    mock_sleep,
    mock_open_file,
    mock_entrez_efetch,
    test_case,
):
    # Arrange
    (
        batch_ids,
        retry_count,
        delay_seconds,
        efetch_side_effect,
        expected_efetch_calls,
        expected_sleep_calls,
        expected_exception,
    ) = test_case
    mock_file_exists.return_value = False
    mock_entrez_efetch.side_effect = efetch_side_effect

    file_path = "test_file_path"
    file_format = "fasta"
    batch_number = 1

    # Act & Assert
    with pytest.raises(expected_exception):
        download_batch(
            file_path, batch_ids, batch_number, retry_count, delay_seconds, file_format
        )

    # Assert
    assert mock_entrez_efetch.call_count == expected_efetch_calls
    assert mock_sleep.call_count == expected_sleep_calls
    if expected_efetch_calls > 0:
        expected_efetch_calls = [
            call(
                db="nucleotide",
                id=",".join(batch_ids),
                rettype=file_format,
                retmode="text",
            )
        ] * (expected_efetch_calls - 1)
        mock_entrez_efetch.assert_has_calls(expected_efetch_calls, any_order=True)
    if expected_sleep_calls > 0:
        expected_sleep_calls = [call(delay_seconds)] * (expected_sleep_calls)
        mock_sleep.assert_has_calls(expected_sleep_calls, any_order=True)

    mock_open_file.assert_not_called()


@pytest.mark.parametrize(
    "test_case",
    [
        # Case: Retry on network error and then succeed
        (
            ["id1", "id2", "id3"],  # batch_ids
            3,  # retry_count
            0,  # delay_seconds
            [
                URLError("Network unreachable"),
                URLError("Network unreachable"),
                MagicMock(read=MagicMock(return_value="test_data")),
            ],  # efetch_side_effects
            3,  # expected_efetch_calls
            2,  # expected_sleep_calls
        ),
        # Case: Retry on HTTP error and then succeed
        (
            ["id1", "id2", "id3"],
            10,
            0,
            [
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
            4,
            3,
        ),
    ],
)
def test_download_batch_retries_and_succeeds(
    mock_open_file,
    mock_file_exists,
    mock_sleep,
    mock_entrez_efetch,
    test_case,
):
    # Arrange
    (
        batch_ids,
        retry_count,
        delay_seconds,
        efetch_side_effects,
        expected_efetch_calls,
        expected_sleep_calls,
    ) = test_case
    mock_file_exists.return_value = False
    mock_entrez_efetch.side_effect = efetch_side_effects

    file_path = "test_file_path"
    file_format = "fasta"
    batch_number = 1

    # Act
    download_batch(
        file_path, batch_ids, batch_number, retry_count, delay_seconds, file_format
    )

    # Assert
    assert mock_entrez_efetch.call_count == expected_efetch_calls
    assert mock_sleep.call_count == expected_sleep_calls
    if expected_efetch_calls > 0:
        expected_efetch_calls = [
            call(
                db="nucleotide",
                id=",".join(batch_ids),
                rettype=file_format,
                retmode="text",
            )
        ] * (expected_efetch_calls - 1)
        mock_entrez_efetch.assert_has_calls(expected_efetch_calls, any_order=False)
    if expected_sleep_calls > 0:
        expected_sleep_calls = [call(delay_seconds)] * (expected_sleep_calls)
        mock_sleep.assert_has_calls(expected_sleep_calls, any_order=False)
    mock_open_file.assert_called_once_with(file_path, "w")
    mock_open_file().write.assert_called_once_with("test_data")
