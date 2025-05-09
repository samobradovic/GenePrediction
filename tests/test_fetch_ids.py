from pathlib import Path
from typing import List, NamedTuple, Tuple
from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.data_loading.fetch_ids import (
    download_ids,
    fetch_ids,
    get_tax_ids_for_ids,
    round_robin_by_tax_id,
)


@pytest.fixture
def mock_entrez_read():
    with patch("Bio.Entrez.read") as mock_read:
        yield mock_read


@pytest.fixture
def mock_entrez_esearch():
    with patch("Bio.Entrez.esearch") as mock_esearch:
        yield mock_esearch


@pytest.fixture
def mock_entrez_esummary():
    with patch("Bio.Entrez.esummary") as mock_esummary:
        yield mock_esummary


@pytest.fixture
def mock_open_file():
    with patch("builtins.open", new_callable=mock_open) as mock:
        yield mock


class TestDownloadIds:
    class CaseData(NamedTuple):
        description: str
        max_results: int
        search_query: str
        file_path: Path
        min_h_value: float
        fetch_ids_returns: List[List[str]]
        tax_ids_returns: List[List[int]]
        diversity_scores: List[float] | None = None
        final_subset: List[str] | None = None
        expected_fetch_calls: int | None = None
        round_robin_returns: List[List[tuple[str, int]]] | None = None

    TEST_CASES = [
        CaseData(
            description="Returns after first iteration when diversity > min_h",
            max_results=2,
            search_query="test query",
            file_path=Path("test.txt"),
            min_h_value=0.4,
            fetch_ids_returns=[["id1", "id2"]],
            tax_ids_returns=[[1, 2]],
            diversity_scores=[0.5],
            final_subset=["id1", "id2"],
            expected_fetch_calls=1,
            round_robin_returns=[[("id1", 1), ("id2", 2)]],
        ),
        CaseData(
            description="Takes 3 iterations to reach min diversity",
            max_results=2,
            search_query="test query",
            file_path=Path("test.txt"),
            min_h_value=0.7,
            fetch_ids_returns=[["id1", "id2"], ["id3", "id4"], ["id5", "id6"]],
            tax_ids_returns=[[1, 1], [1, 2], [1, 3]],
            diversity_scores=[0.2, 0.6, 0.8],
            final_subset=["id1", "id3", "id5"],
            expected_fetch_calls=3,
            round_robin_returns=[
                [("id1", 1), ("id2", 1)],
                [("id1", 1), ("id3", 2)],
                [("id1", 1), ("id3", 2), ("id5", 3)],
            ],
        ),
    ]

    ERROR_TEST_CASES = [
        CaseData(
            description="Tax IDs list shorter than IDs list",
            max_results=2,
            search_query="test query",
            file_path=Path("test.txt"),
            min_h_value=0.4,
            fetch_ids_returns=[["id1", "id2"]],
            tax_ids_returns=[[1]],
        ),
        CaseData(
            description="Tax IDs list longer than IDs list",
            max_results=2,
            search_query="test query",
            file_path=Path("test.txt"),
            min_h_value=0.4,
            fetch_ids_returns=[["id1"]],
            tax_ids_returns=[[1, 2]],
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        TEST_CASES,
        ids=[case.description for case in TEST_CASES],
    )
    @patch("src.data_loading.fetch_ids.normalized_shannon_diversity")
    @patch("src.data_loading.fetch_ids.round_robin_by_tax_id")
    @patch("src.data_loading.fetch_ids.get_tax_ids_for_ids")
    @patch("src.data_loading.fetch_ids.fetch_ids")
    def test_download_ids(
        self,
        mock_fetch_ids: MagicMock,
        mock_get_tax_ids: MagicMock,
        mock_round_robin: MagicMock,
        mock_diversity: MagicMock,
        mock_open_file: MagicMock,
        test_case: CaseData,
    ):
        # Arrange
        assert test_case.diversity_scores is not None, "diversity_scores cannot be None"
        assert test_case.final_subset is not None, "final_subset cannot be None"
        assert test_case.expected_fetch_calls is not None, (
            "expected_fetch_calls cannot be None"
        )
        assert test_case.round_robin_returns is not None, (
            "round_robin_returns cannot be None"
        )

        mock_fetch_ids.side_effect = test_case.fetch_ids_returns
        mock_get_tax_ids.side_effect = test_case.tax_ids_returns
        mock_diversity.side_effect = test_case.diversity_scores
        mock_round_robin.side_effect = test_case.round_robin_returns

        # Act
        download_ids(
            test_case.max_results,
            test_case.search_query,
            test_case.file_path,
            test_case.min_h_value,
        )

        # Assert
        assert mock_fetch_ids.call_count == test_case.expected_fetch_calls
        assert mock_get_tax_ids.call_count == test_case.expected_fetch_calls
        assert mock_round_robin.call_count == test_case.expected_fetch_calls
        assert mock_diversity.call_count == test_case.expected_fetch_calls

        mock_open_file.assert_called_once_with(test_case.file_path, "w")
        handle = mock_open_file()
        for id in test_case.final_subset:
            handle.write.assert_any_call(f"{id}\n")

    @pytest.mark.parametrize(
        "test_case",
        ERROR_TEST_CASES,
        ids=[case.description for case in ERROR_TEST_CASES],
    )
    @patch("src.data_loading.fetch_ids.get_tax_ids_for_ids")
    @patch("src.data_loading.fetch_ids.fetch_ids")
    def test_download_ids_raises(
        self,
        mock_fetch_ids: MagicMock,
        mock_get_tax_ids: MagicMock,
        test_case: CaseData,
    ):
        # Arrange
        mock_fetch_ids.side_effect = test_case.fetch_ids_returns
        mock_get_tax_ids.side_effect = test_case.tax_ids_returns

        # Act & Assert
        with pytest.raises(ValueError):
            download_ids(
                test_case.max_results,
                test_case.search_query,
                test_case.file_path,
                test_case.min_h_value,
            )


class TestFetchIds:
    class CaseData(NamedTuple):
        description: str
        max_results: int
        search_query: str
        offset: int
        mock_response: dict
        expected_ids: List[str] | None = None
        expected_exception: type[Exception] | None = None

    TEST_CASES: List[CaseData] = [
        CaseData(
            description="Valid response with IDs",
            max_results=10,
            search_query="test_query",
            offset=0,
            mock_response={"IdList": ["NM_001301717", "NM_001301718"]},
            expected_ids=["NM_001301717", "NM_001301718"],
            expected_exception=None,
        ),
        CaseData(
            description="Valid response with single ID",
            max_results=1,
            search_query="test_query",
            offset=0,
            mock_response={"IdList": ["NM_001301717"]},
            expected_ids=["NM_001301717"],
            expected_exception=None,
        ),
        CaseData(
            description="Valid response with max results",
            max_results=100,
            search_query="test_query",
            offset=0,
            mock_response={"IdList": [f"NM_{i:09}" for i in range(100)]},
            expected_ids=[f"NM_{i:09}" for i in range(100)],
            expected_exception=None,
        ),
        CaseData(
            description="Valid response with offset",
            max_results=10,
            search_query="test_query",
            offset=50,
            mock_response={"IdList": [f"NM_{i:09}" for i in range(50, 60)]},
            expected_ids=[f"NM_{i:09}" for i in range(50, 60)],
            expected_exception=None,
        ),
    ]

    ERROR_TEST_CASES: List[CaseData] = [
        CaseData(
            description="Empty response with no IDs",
            max_results=10,
            search_query="test_query",
            offset=0,
            mock_response={"IdList": []},
            expected_exception=ValueError,
        ),
        CaseData(
            description="Invalid response format (missing IdList key)",
            max_results=10,
            search_query="test_query",
            offset=0,
            mock_response={},
            expected_exception=ValueError,
        ),
        CaseData(
            description="Invalid IdList format (not a list)",
            max_results=10,
            search_query="test_query",
            offset=0,
            mock_response={"IdList": "not a list"},
            expected_exception=ValueError,
        ),
        CaseData(
            description="Invalid response format (wrong dict structure)",
            max_results=10,
            search_query="test_query",
            offset=0,
            mock_response={"wrong_key": ["id1", "id2"]},
            expected_exception=ValueError,
        ),
        CaseData(
            description="Invalid ID format in list",
            max_results=10,
            search_query="test_query",
            offset=0,
            mock_response={"IdList": [1, 2, 3]},
            expected_exception=ValueError,
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        TEST_CASES,
        ids=[case.description for case in TEST_CASES],
    )
    def test_fetch_ids(
        self,
        mock_entrez_esearch: MagicMock,
        mock_entrez_read: MagicMock,
        test_case: CaseData,
    ):
        # Arrange
        mock_entrez_esearch.return_value = MagicMock()
        mock_entrez_read.return_value = test_case.mock_response

        # Act
        result = fetch_ids(
            max_results=test_case.max_results,
            search_query=test_case.search_query,
            offset=test_case.offset,
        )

        # Assert
        assert result == test_case.expected_ids

    @pytest.mark.parametrize(
        "test_case",
        ERROR_TEST_CASES,
        ids=[case.description for case in ERROR_TEST_CASES],
    )
    def test_fetch_ids_raises(
        self,
        mock_entrez_esearch: MagicMock,
        mock_entrez_read: MagicMock,
        test_case: CaseData,
    ):
        # Arrange
        mock_entrez_esearch.return_value = MagicMock()
        mock_entrez_read.return_value = test_case.mock_response
        # Act & Assert
        assert test_case.expected_exception is not None
        with pytest.raises(test_case.expected_exception):
            fetch_ids(
                max_results=test_case.max_results,
                search_query=test_case.search_query,
                offset=test_case.offset,
            )


class TestGetTaxIdsForIds:
    class CaseData(NamedTuple):
        description: str
        ids: List[str]
        mock_response: object
        expected_tax_ids: List[int] | None = None
        expected_exception: type[Exception] | None = None

    TEST_CASES = [
        CaseData(
            description="Valid response with tax IDs",
            ids=["28864546", "28800981"],
            mock_response=[
                {"Item": [], "Id": "28864546", "Caption": "AY207443", "TaxId": 9606},
                {"Item": [], "Id": "28800981", "Caption": "AY205604", "TaxId": 9606},
            ],
            expected_tax_ids=[9606, 9606],
            expected_exception=None,
        ),
        CaseData(
            description="Valid response with different tax IDs",
            ids=["28864546", "28800981", "28800982"],
            mock_response=[
                {"Item": [], "Id": "28864546", "Caption": "AY207443", "TaxId": 9606},
                {"Item": [], "Id": "28800981", "Caption": "AY205604", "TaxId": 10090},
                {"Item": [], "Id": "28800982", "Caption": "AY205605", "TaxId": 7227},
            ],
            expected_tax_ids=[9606, 10090, 7227],
            expected_exception=None,
        ),
        CaseData(
            description="Valid response with single ID",
            ids=["28864546"],
            mock_response=[
                {"Item": [], "Id": "28864546", "Caption": "AY207443", "TaxId": 9606},
            ],
            expected_tax_ids=[9606],
            expected_exception=None,
        ),
    ]

    ERROR_TEST_CASES = [
        CaseData(
            description="Empty response",
            ids=["28864546"],
            mock_response=[],
            expected_tax_ids=None,
            expected_exception=ValueError,
        ),
        CaseData(
            description="Invalid response format",
            ids=["28864546"],
            mock_response=None,
            expected_tax_ids=None,
            expected_exception=ValueError,
        ),
        CaseData(
            description="Missing TaxId field in response",
            ids=["28864546"],
            mock_response=[
                {"Item": [], "Id": "28864546", "Caption": "AY207443"},
            ],
            expected_tax_ids=None,
            expected_exception=ValueError,
        ),
        CaseData(
            description="Invalid TaxId type in response",
            ids=["28864546"],
            mock_response=[
                {
                    "Item": [],
                    "Id": "28864546",
                    "Caption": "AY207443",
                    "TaxId": "invalid",
                },
            ],
            expected_tax_ids=None,
            expected_exception=ValueError,
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        TEST_CASES,
        ids=[case.description for case in TEST_CASES],
    )
    def test_get_tax_ids_for_ids(
        self,
        mock_entrez_esummary: MagicMock,
        mock_entrez_read: MagicMock,
        test_case: CaseData,
    ):
        # Arrange
        mock_entrez_esummary.return_value = MagicMock()
        mock_entrez_read.return_value = test_case.mock_response

        # Act
        result = get_tax_ids_for_ids(test_case.ids)

        # Assert
        assert result == test_case.expected_tax_ids

    @pytest.mark.parametrize(
        "test_case",
        ERROR_TEST_CASES,
        ids=[case.description for case in ERROR_TEST_CASES],
    )
    def test_get_tax_ids_for_ids_raises(
        self,
        mock_entrez_esummary: MagicMock,
        mock_entrez_read: MagicMock,
        test_case: CaseData,
    ):
        # Arrange
        mock_entrez_esummary.return_value = MagicMock()
        mock_entrez_read.return_value = test_case.mock_response

        # Act & Assert
        assert test_case.expected_exception is not None
        with pytest.raises(test_case.expected_exception):
            get_tax_ids_for_ids(test_case.ids)


class TestRoundRobinByTaxId:
    class CaseData(NamedTuple):
        description: str
        id_tax_pairs: List[Tuple[str, int]]
        subset_length: int
        expected_result: List[Tuple[str, int]]

    TEST_CASES = [
        CaseData(
            description="Single tax ID, subset length equal to input length",
            id_tax_pairs=[("id1", 1), ("id2", 1), ("id3", 1)],
            subset_length=3,
            expected_result=[("id1", 1), ("id2", 1), ("id3", 1)],
        ),
        CaseData(
            description="Multiple tax IDs, equal distribution",
            id_tax_pairs=[
                ("id1", 1),
                ("id2", 2),
                ("id3", 3),
                ("id4", 1),
                ("id5", 2),
                ("id6", 3),
            ],
            subset_length=6,
            expected_result=[
                ("id1", 1),
                ("id2", 2),
                ("id3", 3),
                ("id4", 1),
                ("id5", 2),
                ("id6", 3),
            ],
        ),
        CaseData(
            description="Multiple tax IDs, unequal distribution",
            id_tax_pairs=[("id1", 1), ("id2", 1), ("id3", 1), ("id4", 2), ("id5", 3)],
            subset_length=5,
            expected_result=[
                ("id1", 1),
                ("id4", 2),
                ("id5", 3),
                ("id2", 1),
                ("id3", 1),
            ],
        ),
        CaseData(
            description="Subset length smaller than input",
            id_tax_pairs=[
                ("id1", 1),
                ("id2", 2),
                ("id3", 3),
                ("id4", 1),
                ("id5", 2),
                ("id6", 3),
            ],
            subset_length=4,
            expected_result=[("id1", 1), ("id2", 2), ("id3", 3), ("id4", 1)],
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        TEST_CASES,
        ids=[case.description for case in TEST_CASES],
    )
    def test_round_robin_by_tax_id(self, test_case: CaseData):
        # Act
        result = round_robin_by_tax_id(test_case.id_tax_pairs, test_case.subset_length)

        # Assert
        assert result == test_case.expected_result
        assert len(result) <= test_case.subset_length
