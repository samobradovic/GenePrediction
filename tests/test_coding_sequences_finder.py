from typing import List

import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from gene_prediction.coding_sequence_finder import (
    get_coding_sequences, get_coding_sequences_for_gene, read_orf)


@pytest.mark.parametrize(
    "genes, expected_sequences",
    [
        # Single gene, valid start and stop codons
        (
            [SeqRecord(Seq("AUGGCCUAA"))],
            ["AUGGCCUAA"],
        ),
        # Multiple genes with coding sequences
        (
            [
                SeqRecord(Seq("AUGAAAUAA")),
                SeqRecord(Seq("CCCAUGGGGUAG")),
            ],
            ["AUGAAAUAA", "AUGGGGUAG"],
        ),
        # No start codon
        (
            [SeqRecord(Seq("CCCGGGTTT"))],
            [],
        ),
        # Start but no stop codon
        (
            [SeqRecord(Seq("AUGGCCGGG"))],
            ["AUGGCCGGG!"],
        ),
    ],
)
def test_get_coding_sequences(genes: List[SeqRecord], expected_sequences: List[str]):
    assert get_coding_sequences(iter(genes)) == expected_sequences


@pytest.mark.parametrize(
    "gene, expected_sequences",
    [
        (SeqRecord(Seq("AUGGCCUAA")), ["AUGGCCUAA"]),
        (SeqRecord(Seq("AUGAAAUAG")), ["AUGAAAUAG"]),
        (SeqRecord(Seq("CCCAUGGGGUAG")), ["AUGGGGUAG"]),
        (SeqRecord(Seq("CCCGGGTTT")), []),
        (SeqRecord(Seq("AUGGCCGGG")), ["AUGGCCGGG!"]),
    ],
)
def test_get_coding_sequences_for_gene(gene: SeqRecord, expected_sequences: List[str]):
    assert get_coding_sequences_for_gene(gene) == expected_sequences


@pytest.mark.parametrize(
    "sliced_dna_sequence, expected_sequence, expected_index",
    [
        ("AUGGCCUAA", "AUGGCCUAA", 9),  # Start to valid stop
        ("AUGAAAUAG", "AUGAAAUAG", 9),  # Start to another valid stop
        ("AUGGCCGGG", "AUGGCCGGG!", -1),  # Start but no stop codon
        ("AUGGCCUAAAAAAUGGGG", "AUGGCCUAA", 9),  # Multiple start codons
    ],
)
def test_read_orf(
    sliced_dna_sequence: str, expected_sequence: str, expected_index: int
):
    assert read_orf(sliced_dna_sequence) == (expected_sequence, expected_index)


@pytest.mark.parametrize(
    "sliced_dna_sequence",
    [
        "CCCGGGTTT",  # No start codon
        "ABCDEF",  # Invalid codons
        "CA",  # Invalid codon
        "AU",  # Invalid codon
        "A",  # Invalid codon
    ],
)
def test_read_orf_valueerror(sliced_dna_sequence: str):
    with pytest.raises(ValueError, match="No start codon found"):
        read_orf(sliced_dna_sequence)
