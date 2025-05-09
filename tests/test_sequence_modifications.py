from typing import NamedTuple

import pytest

from src.data_preprocessing.sequence_modifications import mutate_string_sequence


class TestMutateSequence:
    class CaseData(NamedTuple):
        description: str
        sequence: str
        mutation_rate: float
        number_of_expected_mutations: int
        exception: type[Exception] | None = None
        exception_message: str | None = None

    TEST_CASES = [
        CaseData(
            description="25 percent of the sequence is mutated",
            sequence="ATCGATCGATCGATCG",
            mutation_rate=0.25,
            number_of_expected_mutations=4,
        ),
        CaseData(
            description="50 percent of the sequence is mutated",
            sequence="ATCGATCGATCGATCG",
            mutation_rate=0.5,
            number_of_expected_mutations=8,
        ),
        CaseData(
            description="75 percent of the sequence is mutated",
            sequence="ATCGATCGATCGATCG",
            mutation_rate=0.75,
            number_of_expected_mutations=12,
        ),
        CaseData(
            description="0 percent of the sequence is mutated",
            sequence="ATCGATCGATCGATCG",
            mutation_rate=0.0,
            number_of_expected_mutations=0,
        ),
        CaseData(
            description="100 percent of the sequence is mutated",
            sequence="ATCGATCGATCGATCG",
            mutation_rate=1.0,
            number_of_expected_mutations=16,
        ),
        CaseData(
            description="Single base sequence, rounded down no change",
            sequence="A",
            mutation_rate=0.5,
            number_of_expected_mutations=0,
        ),
        CaseData(
            description="Single base sequence, rounded down no change",
            sequence="A",
            mutation_rate=0.6,
            number_of_expected_mutations=0,
        ),
        CaseData(
            description="Single base sequence, rounded up change",
            sequence="A",
            mutation_rate=1.0,
            number_of_expected_mutations=1,
        ),
        CaseData(
            description="Uneven sequence, should have 1 mutation",
            sequence="ATG",
            mutation_rate=0.5,
            number_of_expected_mutations=1,
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        TEST_CASES,
        ids=[case.description for case in TEST_CASES],
    )
    def test_mutate_sequence(self, test_case: CaseData):
        # Arrange
        sequence = test_case.sequence
        mutation_rate = test_case.mutation_rate

        # Act
        mutated_sequence = mutate_string_sequence(sequence, mutation_rate)

        # Assert
        assert len(mutated_sequence) == len(sequence)
        assert all(base in "ATCG" for base in mutated_sequence)
        assert (
            sum(
                base == original
                for base, original in zip(mutated_sequence, sequence, strict=True)
            )
            == len(sequence) - test_case.number_of_expected_mutations
        )

    EXCEPTION_TEST_CASES = [
        CaseData(
            description="Mutation rate is negative",
            sequence="ATCGATCGATCGATCG",
            mutation_rate=-0.1,
            number_of_expected_mutations=0,
            exception=ValueError,
            exception_message="Mutation rate must be between 0 and 1.",
        ),
        CaseData(
            description="Mutation rate is greater than 1",
            sequence="ATCGATCGATCGATCG",
            mutation_rate=1.1,
            number_of_expected_mutations=0,
            exception=ValueError,
            exception_message="Mutation rate must be between 0 and 1.",
        ),
        CaseData(
            description="Empty sequence",
            sequence="",
            mutation_rate=0.5,
            number_of_expected_mutations=0,
            exception=ValueError,
            exception_message="Sequence must not be empty.",
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        EXCEPTION_TEST_CASES,
        ids=[case.description for case in EXCEPTION_TEST_CASES],
    )
    def test_mutate_sequence_exceptions(self, test_case: CaseData):
        # Arrange
        sequence = test_case.sequence
        mutation_rate = test_case.mutation_rate

        # Act & Assert
        assert test_case.exception is not None
        with pytest.raises(test_case.exception, match=test_case.exception_message):
            mutate_string_sequence(sequence, mutation_rate)
