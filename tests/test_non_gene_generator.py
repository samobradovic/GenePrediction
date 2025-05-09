from pathlib import Path
from typing import List, NamedTuple
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data_preprocessing.non_gene_generator import (
    generate_non_genes,
    generate_random_non_genes,
    get_random_gene_frame,
)


class TestGetRandomGeneFrame:
    class GetRandomGeneFrameTestCase(NamedTuple):
        description: str
        genome: str
        random_length: int
        expected_length: int

    TEST_CASES = [
        GetRandomGeneFrameTestCase(
            description="Genome longer than random length",
            genome="ATCGATCGATCG",
            random_length=4,
            expected_length=4,
        ),
        GetRandomGeneFrameTestCase(
            description="Genome equal to random length",
            genome="GCTAGCTA",
            random_length=8,
            expected_length=8,
        ),
        GetRandomGeneFrameTestCase(
            description="Genome shorter than random length",
            genome="A",
            random_length=1,
            expected_length=1,
        ),
        GetRandomGeneFrameTestCase(
            description="Genome with special characters",
            genome="ATCG@#%$^&*",
            random_length=4,
            expected_length=4,
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        TEST_CASES,
        ids=[case.description for case in TEST_CASES],
    )
    def test_get_random_gene_frame(self, test_case: GetRandomGeneFrameTestCase):
        result = get_random_gene_frame(test_case.genome, test_case.random_length)
        assert len(result) == test_case.expected_length
        assert result in test_case.genome


class TestGenerateRandomNonGenes:
    class CaseData(NamedTuple):
        description: str
        percentage_of_genes: float
        random_length: int
        number_of_genes: int
        genomes: List[str]
        random_gene_frames: List[str]
        expected_count: int
        expected_exception: type[Exception] | None = None
        expected_message: str | None = None

    TEST_CASES = [
        CaseData(
            description="Generate non-genes from multiple genomes",
            percentage_of_genes=0.5,
            random_length=4,
            number_of_genes=6,
            genomes=["ATCGATCG", "GCTAGCTA"],
            random_gene_frames=["ATCG", "GCTA", "TAGC"],
            expected_count=3,
            expected_exception=None,
        ),
        CaseData(
            description="Generate non-genes from a single genome",
            percentage_of_genes=0.5,
            random_length=3,
            number_of_genes=4,
            genomes=["ATCG"],
            random_gene_frames=["ATC", "TCG"],
            expected_count=2,
            expected_exception=None,
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        TEST_CASES,
        ids=[case.description for case in TEST_CASES],
    )
    @patch("src.data_preprocessing.non_gene_generator.get_random_gene_frame")
    def test_generate_random_non_genes(
        self,
        mock_get_random_gene_frame: MagicMock,
        test_case: CaseData,
    ):
        # Arrange
        mock_get_random_gene_frame.side_effect = test_case.random_gene_frames

        # Act
        result = generate_random_non_genes(
            test_case.percentage_of_genes,
            test_case.random_length,
            test_case.number_of_genes,
            test_case.genomes,
        )

        # Assert
        assert len(result) == test_case.expected_count
        assert result == test_case.random_gene_frames

        assert mock_get_random_gene_frame.call_count == test_case.expected_count
        _, kwargs = mock_get_random_gene_frame.call_args
        assert "random_length" in kwargs
        assert "genome" in kwargs
        assert kwargs["genome"] in test_case.genomes

    TEST_CASES_EXCEPTIONS = [
        CaseData(
            description="Generate non-genes with empty genome list",
            percentage_of_genes=0.5,
            random_length=4,
            number_of_genes=6,
            genomes=[],
            random_gene_frames=[],
            expected_count=0,
            expected_exception=ValueError,
            expected_message="No genomes provided",
        ),
        CaseData(
            description="Generate non-genes with zero percentage",
            percentage_of_genes=0.0,
            random_length=4,
            number_of_genes=6,
            genomes=["ATCGATCG", "GCTAGCTA"],
            random_gene_frames=[],
            expected_count=0,
            expected_exception=ValueError,
        ),
        CaseData(
            description="Generate non-genes with zero number of genes",
            percentage_of_genes=0.5,
            random_length=4,
            number_of_genes=0,
            genomes=["ATCGATCG", "GCTAGCTA"],
            random_gene_frames=[],
            expected_count=0,
            expected_exception=ValueError,
        ),
        CaseData(
            description="Generate non-genes with random length greater than genome len",
            percentage_of_genes=0.5,
            random_length=20,
            number_of_genes=6,
            genomes=["ATCGATCG", "GCTAGCTA"],
            random_gene_frames=[],
            expected_count=0,
            expected_exception=ValueError,
        ),
        CaseData(
            description="Generate non-genes with random length equal to genome length",
            percentage_of_genes=0.5,
            random_length=8,
            number_of_genes=6,
            genomes=["ATCGATCG", "GCTAGCTA"],
            random_gene_frames=[],
            expected_count=0,
            expected_exception=ValueError,
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        TEST_CASES_EXCEPTIONS,
        ids=[case.description for case in TEST_CASES_EXCEPTIONS],
    )
    @patch("src.data_preprocessing.non_gene_generator.get_random_gene_frame")
    def test_generate_random_non_genes_exception(
        self,
        mock_get_random_gene_frame: MagicMock,
        test_case: CaseData,
    ):
        # Arrange
        mock_get_random_gene_frame.side_effect = test_case.random_gene_frames

        # Act & Assert
        assert test_case.expected_exception
        with pytest.raises(
            test_case.expected_exception, match=test_case.expected_message
        ):
            generate_random_non_genes(
                test_case.percentage_of_genes,
                test_case.random_length,
                test_case.number_of_genes,
                test_case.genomes,
            )


class TestGenerateNonGenes:
    class GenerateNonGenesTestCase(NamedTuple):
        description: str
        genes_csv_file: Path
        genomes_raw_data_folder: Path
        combined_csv_file: Path
        gene_lengths: pd.Series
        random_non_genes: List[str]
        genome_sequences: List[str]
        percentage_of_genes: float

    TEST_CASES = [
        GenerateNonGenesTestCase(
            description="Generate non-genes and write to CSV",
            genes_csv_file=Path("test_genes.csv"),
            genomes_raw_data_folder=Path("test_genomes"),
            combined_csv_file=Path("test_combined.csv"),
            gene_lengths=pd.Series([4, 5, 6]),
            random_non_genes=["ATCG", "GCTA", "TAGC"],
            genome_sequences=["ATCGATCG", "GCTAGCTA"],
            percentage_of_genes=0.5,
        ),
        GenerateNonGenesTestCase(
            description="Generate non-genes with more lengths then genomes",
            genes_csv_file=Path("test_genes.csv"),
            genomes_raw_data_folder=Path("test_genomes"),
            combined_csv_file=Path("test_combined.csv"),
            gene_lengths=pd.Series([3, 4, 5, 6, 2, 5, 6, 3, 3, 2]),
            random_non_genes=["ATC", "GCT", "TAG"],
            genome_sequences=["ATCGATCG", "GCTAGCTA"],
            percentage_of_genes=0.5,
        ),
        GenerateNonGenesTestCase(
            description="Generate non-genes with empty genome sequences",
            genes_csv_file=Path("test_genes.csv"),
            genomes_raw_data_folder=Path("test_genomes"),
            combined_csv_file=Path("test_combined.csv"),
            gene_lengths=pd.Series([4, 5, 6]),
            random_non_genes=[],
            genome_sequences=[],
            percentage_of_genes=0.5,
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        TEST_CASES,
        ids=[case.description for case in TEST_CASES],
    )
    @patch(
        "src.data_preprocessing.non_gene_generator.get_iterable_series_sequence_lengths_from_csv"
    )
    @patch(
        "src.data_preprocessing.non_gene_generator.get_string_sequences_from_directory"
    )
    @patch("src.data_preprocessing.non_gene_generator.copy_csv_file")
    @patch("src.data_preprocessing.non_gene_generator.write_string_sequences_to_csv")
    @patch("src.data_preprocessing.non_gene_generator.generate_random_non_genes")
    def test_generate_non_genes(
        self,
        mock_generate_random_non_genes: MagicMock,
        mock_write_string_sequences_to_csv: MagicMock,
        mock_copy_csv_file: MagicMock,
        mock_get_string_sequences_from_directory: MagicMock,
        mock_get_iterable_series_sequence_lengths_from_csv: MagicMock,
        test_case: GenerateNonGenesTestCase,
    ):
        # Arrange
        mock_get_iterable_series_sequence_lengths_from_csv.return_value = (
            test_case.gene_lengths
        )
        mock_get_string_sequences_from_directory.return_value = (
            test_case.genome_sequences
        )

        mock_generate_random_non_genes.return_value = test_case.random_non_genes

        # Act
        generate_non_genes(
            test_case.percentage_of_genes,
            test_case.genes_csv_file,
            test_case.genomes_raw_data_folder,
            test_case.combined_csv_file,
        )

        # Assert
        mock_get_iterable_series_sequence_lengths_from_csv.assert_called_once_with(
            test_case.genes_csv_file,
        )
        mock_get_string_sequences_from_directory.assert_called_once_with(
            test_case.genomes_raw_data_folder,
        )
        mock_copy_csv_file.assert_called_once_with(
            source_csv_file=test_case.genes_csv_file,
            destination_csv_file=test_case.combined_csv_file,
        )
        mock_write_string_sequences_to_csv.assert_called_once_with(
            sequences=test_case.random_non_genes,
            csv_destination_path=test_case.combined_csv_file,
            write_header=False,
            is_real_gene=False,
        )

        mock_generate_random_non_genes.assert_called_once()
        _, kwargs = mock_generate_random_non_genes.call_args
        assert "random_length" in kwargs
        assert "number_of_genes" in kwargs
        assert "genomes" in kwargs
        assert "percentage_of_genes" in kwargs
        assert kwargs["percentage_of_genes"] == test_case.percentage_of_genes
        assert kwargs["number_of_genes"] == len(test_case.gene_lengths)
        assert kwargs["genomes"] == test_case.genome_sequences
        assert kwargs["random_length"] in test_case.gene_lengths.values
