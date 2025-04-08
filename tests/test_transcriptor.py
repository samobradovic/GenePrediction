# import pytest
# from Bio.Seq import Seq
# from Bio.SeqRecord import SeqRecord
# from transcription.transcriptor import transcribe_dna_sequence_into_rna


# @pytest.mark.parametrize(
#     "dna_record, expected_rna",
#     [
#         (SeqRecord(Seq("ATGCGT"), id="seq1"), "AUGCGU"),  # Normal case
#         (SeqRecord(Seq("TACGTA"), id="seq2"), "UACGUA"),  # Another normal case
#         (SeqRecord(Seq(""), id="seq3"), ""),  # Empty sequence
#         (SeqRecord(Seq("T"), id="seq4"), "U"),  # Single nucleotide
#         (
#             SeqRecord(Seq("TTTTTTTTTTTTTTTTTTTTTT"), id="seq4"),
#             "UUUUUUUUUUUUUUUUUUUUUU",
#         ),  # Long sequence
#         (SeqRecord(Seq("ATGAAATAG"), id="seq5"), "AUGAAAUAG"),  # Normal case
#     ],
# )
# def test_transcribe_dna_into_rna(dna_record, expected_rna):
#     rna_record = transcribe_dna_sequence_into_rna(dna_record)
#     assert str(rna_record.seq) == expected_rna
#     assert rna_record.id == dna_record.id
#     assert rna_record.description == "RNA sequence"


# def test_transcribe_dna_into_rna_none_sequence():
#     with pytest.raises(ValueError, match="The sequence attribute is None."):
#         transcribe_dna_sequence_into_rna(SeqRecord(None, id="seq1"))
