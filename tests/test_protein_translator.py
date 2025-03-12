import pytest

from translation.protein_translator import (save_protein_sequences_to_fasta,
                                            translate_dna_to_protein,
                                            translate_multiple_dna_sequences)


@pytest.mark.parametrize(
    "dna_sequence, expected_protein",
    [
        # Normal sequences
        ("ATGAAATAG", "MK"),  # Methionine, Lysine, Stop codon
        ("ATGGTGTAG", "MV"),  # Methionine, Valine, Stop codon
        ("ATGTGA", "M"),  # Methionine followed by stop codon (TGA)
        ("ATGAAAAG", "MK!"),  # Methionine, Lysine (no stop codon)
        # Sequences with ATG not at the start
        ("GATGAAATAG", "MK"),  # Methionine after G (ignores before ATG)
        ("GGTGTAGATGGTGTAG", "MV"),  # Methionine, Valine after non-ATG codons
        # Testing all stop codons
        ("ATGAAATAA", "MK"),  # Methionine, Stop codon (TAA)
        ("ATGAAATAG", "MK"),  # Methionine, Stop codon (TAG)
        ("ATGAAATGA", "MK"),  # Methionine, Stop codon (TGA)
        # Sequences with multiple stop codons
        ("ATGAAATAAGTAG", "MK"),  # Methionine, Stop codons (TAA, TAG)
        ("ATGAAATGATAAG", "MK"),  # Methionine, Stop codons (TGA, TAG)
        # Invalid codons
        ("ATGBXGTAG", "MX"),  # Invalid codon (X)
        ("ATGYYGTAG", "MX"),  # Invalid codon (Y)
        ("ATGAAACG", "MK!"),  # Invalid codon (CG not in dictionary)
        # Empty DNA sequence
        ("", ""),  # Empty DNA sequence
    ],
)
def test_translate_dna_to_protein(dna_sequence, expected_protein):
    assert translate_dna_to_protein(dna_sequence) == expected_protein


@pytest.mark.parametrize(
    "sequences, expected_proteins",
    [
        (  # Valid sequences
            ["ATGAAATAG", "ATGGTGTAG", "ATGAAATGA"],
            ["MK", "MV", "MK"],
        ),
        # (["ATGAAATAG", "ATGBXGTAG"], ["MK", "MX"]),  # Mixed valid and invalid sequences
        # ([], []),  # Empty list of sequences
    ],
)
def test_translate_multiple_dna_sequences(sequences, expected_proteins):
    assert translate_multiple_dna_sequences(sequences) == expected_proteins


@pytest.mark.parametrize(
    "protein_sequences, file_name, expected_lines",
    [
        (  # Normal case with 2 protein sequences
            ["MK", "MV"],
            "test_output.fasta",
            [">Protein_1", "MK", ">Protein_2", "MV"],
        ),
        ([], "empty_output.fasta", []),  # Empty list of sequences
        (  # Single sequence
            ["MK"],
            "single_protein_output.fasta",
            [">Protein_1", "MK"],
        ),
        (  # Large sequences
            ["M" * 1000, "V" * 1000],
            "large_protein_output.fasta",
            [">Protein_1", "M" * 1000, ">Protein_2", "V" * 1000],
        ),
    ],
)
def test_save_protein_sequences_to_fasta(
    tmp_path, protein_sequences, file_name, expected_lines
):
    file_path = tmp_path / file_name
    save_protein_sequences_to_fasta(protein_sequences, file_path)

    # Verify the file content
    with open(file_path, "r") as fasta_file:
        lines = fasta_file.readlines()
        for i, line in enumerate(expected_lines):
            assert lines[i].strip() == line
