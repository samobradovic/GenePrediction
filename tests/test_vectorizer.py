from pathlib import Path
from typing import List, NamedTuple
from unittest.mock import ANY, MagicMock, call, patch

import pytest
from pandas import DataFrame
from scipy.sparse import csr_matrix

from src.data_preprocessing.vectorizer import (
    get_kmers,
    validate_and_extract_sequences,
    vectorize,
    vectorize_sequences,
)


@pytest.fixture
def mock_read_csv():
    with patch("pandas.read_csv") as mock:
        yield mock


@pytest.fixture
def mock_save_pkl():
    with patch("src.data_preprocessing.vectorizer.save_pkl") as mock:
        yield mock


class TestVectorize:
    class VectorizeTestCase(NamedTuple):
        description: str
        training_csv_path: Path
        testing_csv_path: Path
        validation_csv_path: Path
        training_vectorized_path: Path
        testing_vectorized_path: Path
        validation_vectorized_path: Path
        vectorizer_path: Path
        kmer_size: int
        mock_data: dict

    VECTORIZE_CASES: List[VectorizeTestCase] = [
        VectorizeTestCase(
            description="Valid input CSVs",
            training_csv_path=Path("training.csv"),
            testing_csv_path=Path("testing.csv"),
            validation_csv_path=Path("validation.csv"),
            training_vectorized_path=Path("training_vectorized.pkl"),
            testing_vectorized_path=Path("testing_vectorized.pkl"),
            validation_vectorized_path=Path("validation_vectorized.pkl"),
            vectorizer_path=Path("vectorizer.pkl"),
            kmer_size=3,
            mock_data={
                "X_train_sequence": ["ATCG", "GCTA"],
                "X_test_sequence": ["CGTA"],
                "X_val_sequence": ["TAGC"],
                "train_vectorized": csr_matrix([[1, 0, 0], [0, 1, 0]]),
                "test_vectorized": csr_matrix([[0, 1, 0]]),
                "val_vectorized": csr_matrix([[0, 0, 1]]),
            },
        )
    ]

    @pytest.mark.parametrize(
        "test_case",
        VECTORIZE_CASES,
        ids=[case.description for case in VECTORIZE_CASES],
    )
    @patch("src.data_preprocessing.vectorizer.vectorize_sequences")
    @patch("src.data_preprocessing.vectorizer.validate_and_extract_sequences")
    def test_vectorize(
        self,
        mock_validate_and_extract_sequences: MagicMock,
        mock_vectorize_sequences: MagicMock,
        mock_save_pkl: MagicMock,
        test_case: VectorizeTestCase,
    ):
        # Arrange
        mock_validate_and_extract_sequences.side_effect = [
            test_case.mock_data["X_train_sequence"],
            test_case.mock_data["X_test_sequence"],
            test_case.mock_data["X_val_sequence"],
        ]

        mock_vectorize_sequences.return_value = (
            test_case.mock_data["train_vectorized"],
            test_case.mock_data["test_vectorized"],
            test_case.mock_data["val_vectorized"],
        )

        # Act
        vectorize(
            training_csv_path=test_case.training_csv_path,
            testing_csv_path=test_case.testing_csv_path,
            validation_csv_path=test_case.validation_csv_path,
            training_vectorized_path=test_case.training_vectorized_path,
            testing_vectorized_path=test_case.testing_vectorized_path,
            validation_vectorized_path=test_case.validation_vectorized_path,
            vectorizer_path=test_case.vectorizer_path,
            kmer_size=test_case.kmer_size,
        )

        # Assert
        mock_vectorize_sequences.assert_called_once_with(
            ANY,
            test_case.mock_data["X_train_sequence"],
            test_case.mock_data["X_test_sequence"],
            test_case.mock_data["X_val_sequence"],
            test_case.kmer_size,
        )
        calls = [
            call(
                test_case.mock_data["train_vectorized"],
                test_case.training_vectorized_path,
            ),
            call(
                test_case.mock_data["test_vectorized"],
                test_case.testing_vectorized_path,
            ),
            call(
                test_case.mock_data["val_vectorized"],
                test_case.validation_vectorized_path,
            ),
            call(
                ANY,
                test_case.vectorizer_path,
            ),
        ]
        mock_save_pkl.assert_has_calls(calls, any_order=False)


class TestValidateAndExtractSequences:
    class ValidateAndExtractTestCase(NamedTuple):
        description: str
        csv_content: DataFrame
        file_type: str
        expected_output: List[str]
        raises: type[Exception] | None

    VALIDATE_AND_EXTRACT_TEST_CASES = [
        ValidateAndExtractTestCase(
            description="Valid CSV with sequences",
            csv_content=DataFrame({"Sequence": ["ATCG", "GCTA"]}),
            file_type="Training",
            expected_output=["ATCG", "GCTA"],
            raises=None,
        ),
        ValidateAndExtractTestCase(
            description="Empty CSV",
            csv_content=DataFrame(),
            file_type="Testing",
            expected_output=[],
            raises=ValueError,
        ),
        ValidateAndExtractTestCase(
            description="CSV without 'Sequence' column",
            csv_content=DataFrame({"OtherColumn": ["ATCG", "GCTA"]}),
            file_type="Validation",
            expected_output=[],
            raises=ValueError,
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        VALIDATE_AND_EXTRACT_TEST_CASES,
        ids=[case.description for case in VALIDATE_AND_EXTRACT_TEST_CASES],
    )
    @patch("src.data_preprocessing.vectorizer.pd.read_csv")
    def test_validate_and_extract_sequences(
        self, mock_read_csv: MagicMock, test_case: ValidateAndExtractTestCase
    ):
        # Arrange
        mock_read_csv.return_value = test_case.csv_content

        # Act & Assert
        if test_case.raises:
            with pytest.raises(test_case.raises):
                validate_and_extract_sequences(Path("dummy.csv"), test_case.file_type)
        else:
            result = validate_and_extract_sequences(
                Path("dummy.csv"), test_case.file_type
            )
            assert result == test_case.expected_output


class TestVectorizeSequences:
    class VectorizeSequencesTestCase(NamedTuple):
        description: str
        X_train: List[str]
        train_kmers: List[str]
        X_test: List[str]
        test_kmers: List[str]
        X_val: List[str]
        val_kmers: List[str]
        kmer_size: int
        expected_output: tuple[csr_matrix, csr_matrix, csr_matrix]

    VECTORIZE_SEQUENCES_TEST_CASES = [
        VectorizeSequencesTestCase(
            description="Valid sequences with kmer_size=3",
            X_train=["ATCG", "GCTA"],
            train_kmers=["ATC TCG", "GCT CTA"],
            X_test=["CGTA"],
            test_kmers=["CGT GTA"],
            X_val=["TAGC"],
            val_kmers=["TAG AGC"],
            kmer_size=3,
            expected_output=(
                csr_matrix([[1, 0, 0], [0, 1, 0]]),
                csr_matrix([[0, 1, 0]]),
                csr_matrix([[0, 0, 1]]),
            ),
        ),
        VectorizeSequencesTestCase(
            description="Valid sequences with kmer_size=2",
            X_train=["ATCG", "GCTA"],
            train_kmers=["AT CG", "GC TA"],
            X_test=["CGTA"],
            test_kmers=["CG TA"],
            X_val=["TAGC"],
            val_kmers=["TA GC"],
            kmer_size=2,
            expected_output=(
                csr_matrix([[1, 0], [0, 1]]),
                csr_matrix([[0, 1]]),
                csr_matrix([[1, 0]]),
            ),
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        VECTORIZE_SEQUENCES_TEST_CASES,
        ids=[case.description for case in VECTORIZE_SEQUENCES_TEST_CASES],
    )
    @patch("src.data_preprocessing.vectorizer.get_kmers")
    @patch("src.data_preprocessing.vectorizer.CountVectorizer")
    def test_vectorize_sequences(
        self,
        mock_count_vectorizer: MagicMock,
        mock_get_kmers: MagicMock,
        test_case: VectorizeSequencesTestCase,
    ):
        # Arrange
        mock_vectorizer_instance = MagicMock()
        mock_count_vectorizer.return_value = mock_vectorizer_instance

        mock_get_kmers.side_effect = [
            [kmer]
            for kmer in test_case.train_kmers
            + test_case.test_kmers
            + test_case.val_kmers
        ]

        mock_vectorizer_instance.fit_transform.return_value = test_case.expected_output[
            0
        ]
        mock_vectorizer_instance.transform.side_effect = [
            test_case.expected_output[1],
            test_case.expected_output[2],
        ]

        # Act
        result = vectorize_sequences(
            vectorizer=mock_count_vectorizer(),
            X_train=test_case.X_train,
            X_test=test_case.X_test,
            X_val=test_case.X_val,
            kmer_size=test_case.kmer_size,
        )

        # Assert
        for i, output in enumerate(result):
            assert (output - test_case.expected_output[i]).nnz == 0
        get_kmers_calls = [
            call(sequence, test_case.kmer_size)
            for sequence in test_case.X_train + test_case.X_test + test_case.X_val
        ]
        mock_get_kmers.assert_has_calls(get_kmers_calls, any_order=False)
        mock_vectorizer_instance.fit_transform.assert_called_once_with(
            test_case.train_kmers
        )
        transform_calls = [
            call(test_case.test_kmers),
            call(test_case.val_kmers),
        ]
        mock_vectorizer_instance.transform.assert_has_calls(
            transform_calls, any_order=False
        )


class TestGetKmers:
    class GetKmersTestCase(NamedTuple):
        description: str
        sequence: str
        kmer_size: int
        expected_output: List[str]
        raises: type[Exception] | None = None
        exception_message: str | None = None

    GET_KMERS_TEST_CASES = [
        GetKmersTestCase(
            description="Valid sequence with k=3",
            sequence="ATCG",
            kmer_size=3,
            expected_output=["ATC", "TCG"],
        ),
        GetKmersTestCase(
            description="Valid sequence with k=2",
            sequence="GCTA",
            kmer_size=2,
            expected_output=["GC", "CT", "TA"],
        ),
        GetKmersTestCase(
            description="Valid sequence with k=4",
            sequence="ACGTACGTACGT",
            kmer_size=4,
            expected_output=[
                "ACGT",
                "CGTA",
                "GTAC",
                "TACG",
                "ACGT",
                "CGTA",
                "GTAC",
                "TACG",
                "ACGT",
            ],
        ),
        GetKmersTestCase(
            description="Valid sequence with special characters",
            sequence="A@C#G$T",
            kmer_size=3,
            expected_output=["A@C", "@C#", "C#G", "#G$", "G$T"],
        ),
        GetKmersTestCase(
            description="Empty sequence",
            sequence="",
            kmer_size=3,
            expected_output=[],
            raises=ValueError,
            exception_message="The input sequence is empty.",
        ),
        GetKmersTestCase(
            description="Invalid k (negative value)",
            sequence="ATCG",
            kmer_size=-1,
            expected_output=[],
            raises=ValueError,
            exception_message="k must be a positive integer less"
            " than or equal to the sequence length.",
        ),
        GetKmersTestCase(
            description="Invalid k (zero value)",
            sequence="ATCG",
            kmer_size=0,
            expected_output=[],
            raises=ValueError,
            exception_message="k must be a positive integer less"
            " than or equal to the sequence length.",
        ),
        GetKmersTestCase(
            description="Invalid k (greater than sequence length)",
            sequence="ATCG",
            kmer_size=5,
            expected_output=[],
            raises=ValueError,
            exception_message="k must be a positive integer less"
            " than or equal to the sequence length.",
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        GET_KMERS_TEST_CASES,
        ids=[case.description for case in GET_KMERS_TEST_CASES],
    )
    def test_get_kmers(self, test_case: GetKmersTestCase):
        # Act & Assert
        if test_case.raises:
            with pytest.raises(test_case.raises, match=test_case.exception_message):
                get_kmers(test_case.sequence, test_case.kmer_size)
        else:
            result = get_kmers(test_case.sequence, test_case.kmer_size)
            assert result == test_case.expected_output
