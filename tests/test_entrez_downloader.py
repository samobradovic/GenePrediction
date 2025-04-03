# from pathlib import Path
# from unittest.mock import MagicMock, patch

# import pytest
# from Bio.Entrez.Parser import DictionaryElement

# from src.data_loading.entrez_downloader import (
#     download,
#     download_batch,
#     download_data,
#     get_ids,
# )

# # Mock parameters
# mock_params = {
#     "retry_entrez_request": {"count": 3, "delay_seconds": 1},
#     "file_format": "fasta",
#     "dataset_sizes": {
#         "small": {"genes": 10, "genomes": 5},
#         "big": {"genes": 100, "genomes": 50},
#     },
#     "paths": {
#         "genes_raw_data_folder": "data/raw/genes",
#         "genomes_raw_data_folder": "data/raw/genomes",
#     },
#     "max_sizes_gb": {"genes": 1, "genomes": 1},
#     "batch_sizes": {"genes": 5, "genomes": 2},
#     "entrez_email": "test@example.com",
# }


# @pytest.fixture
# def mock_entrez_read():
#     with patch("src.data_loading.entrez_downloader.Entrez.read") as mock_read:
#         yield mock_read


# @pytest.fixture
# def mock_entrez_esearch():
#     with patch("src.data_loading.entrez_downloader.Entrez.esearch") as mock_esearch:
#         yield mock_esearch


# @pytest.fixture
# def mock_entrez_efetch():
#     with patch("src.data_loading.entrez_downloader.Entrez.efetch") as mock_efetch:
#         yield mock_efetch


# @pytest.fixture
# def mock_get_folder_size():
#     with patch("src.data_loading.entrez_downloader.get_folder_size") as mock_size:
#         yield mock_size


# @pytest.mark.parametrize(
#     "max_results, search_query, mock_response, expected",
#     [
#         (
#             10,
#             "test_query",
#             DictionaryElement(
#                 {
#                     "Count": "125",
#                     "RetMax": "10",
#                     "RetStart": "0",
#                     "IdList": ["NM_001301717", "NM_001301718", "NM_001301719"],
#                     "TranslationSet": [],
#                     "TranslationStack": [
#                         {
#                             "Term": "BRCA1[All Fields]",
#                             "Field": "All Fields",
#                             "Count": "125",
#                         }
#                     ],
#                     "QueryTranslation": "BRCA1[All Fields]",
#                 },
#                 attrs=None,
#                 allowed_tags=None,
#             ),
#             ["NM_001301717", "NM_001301718", "NM_001301719"],
#         ),
#         (10, "test_query", {"IdList": ["1", "2", "3"]}, ["1", "2", "3"]),
#         (10, "test_query", {"IdList": []}, []),
#         (10, "test_query", {}, []),
#     ],
# )
# def test_get_ids(
#     mock_entrez_esearch,
#     mock_entrez_read,
#     max_results,
#     search_query,
#     mock_response: DictionaryElement,
#     expected,
# ):
#     mock_entrez_esearch.return_value = MagicMock()
#     mock_entrez_read.return_value = mock_response

#     result = get_ids(max_results, search_query)
#     assert result == expected


# # Test download_batch
# @pytest.mark.parametrize(
#     "file_exists, batch_ids, batch_number, should_write",
#     [
#         (True, ["1", "2"], 1, False),
#         (False, ["1", "2"], 1, True),
#     ],
# )
# def test_download_batch(
#     mock_entrez_efetch, tmp_path, file_exists, batch_ids, batch_number, should_write
# ):
#     file_path = tmp_path / "batch_1.fasta"
#     if file_exists:
#         file_path.touch()

#     mock_entrez_efetch.return_value.read.return_value = ">MockSequence\nATCG"

#     if should_write:
#         download_batch(str(file_path), batch_ids, batch_number)
#         assert file_path.exists()
#         with open(file_path, "r") as f:
#             assert f.read() == ">MockSequence\nATCG"
#     else:
#         download_batch(str(file_path), batch_ids, batch_number)
#         assert file_path.exists()  # File should not be overwritten


# # Test download_data
# def test_download_data(
#     mock_get_folder_size,
#     mock_entrez_esearch,
#     mock_entrez_read,
#     mock_entrez_efetch,
#     tmp_path,
# ):
#     mock_get_folder_size.return_value = 0
#     mock_entrez_esearch.return_value = MagicMock()
#     mock_entrez_read.return_value = {"IdList": ["1", "2", "3", "4", "5"]}
#     mock_entrez_efetch.return_value.read.return_value = ">MockSequence\nATCG"

#     folder_path = tmp_path / "genes"
#     download_data(
#         number_of_datapoints=5,
#         search_query="test_query",
#         folder_path=folder_path,
#         max_size_gb=1,
#         batch_size=2,
#     )

#     # Check that files are created
#     assert len(list(folder_path.iterdir())) == 3  # 3 batches for 5 IDs


# # Test download
# @pytest.mark.parametrize(
#     "dataset_size, expected_genes, expected_genomes",
#     [
#         ("small", 10, 5),
#         ("big", 100, 50),
#     ],
# )
# def test_download(
#     mock_get_folder_size,
#     mock_entrez_esearch,
#     mock_entrez_read,
#     mock_entrez_efetch,
#     tmp_path,
#     dataset_size,
#     expected_genes,
#     expected_genomes,
# ):
#     mock_get_folder_size.return_value = 0
#     mock_entrez_esearch.return_value = MagicMock()
#     mock_entrez_read.return_value = {"IdList": ["1", "2", "3", "4", "5"]}
#     mock_entrez_efetch.return_value.read.return_value = ">MockSequence\nATCG"

#     mock_params["paths"]["genes_raw_data_folder"] = str(tmp_path / "genes")
#     mock_params["paths"]["genomes_raw_data_folder"] = str(tmp_path / "genomes")

#     with patch("src.data_loading.entrez_downloader.params", mock_params):
#         download(dataset_size)

#     # Check that files are created for genes and genomes
#     genes_folder = Path(mock_params["paths"]["genes_raw_data_folder"])
#     genomes_folder = Path(mock_params["paths"]["genomes_raw_data_folder"])
#     assert len(list(genes_folder.iterdir())) > 0
#     assert len(list(genomes_folder.iterdir())) > 0
