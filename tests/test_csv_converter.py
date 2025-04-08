# import pytest
# from Bio.Seq import Seq
# from Bio.SeqRecord import SeqRecord

# from src.data_preprocessing.csv_converter import (
#     get_random_genome,
#     save_non_genes_to_csv,
# )


# @pytest.fixture
# def setup_csv_file(tmp_path):
#     csv_file = tmp_path / "processed.csv"
#     with open(csv_file, "w") as f:
#         f.write("IsGene,Sequence\n")
#     return csv_file


# @pytest.mark.parametrize(
#     "is_gene, sequences, expected_lines",
#     [
#         (True, ["ATGC", "CGTA"], ["1,ATGC\n", "1,CGTA\n"]),
#         (False, ["ATGC", "CGTA"], ["0,ATGC\n", "0,CGTA\n"]),
#         (True, [], []),  # Empty sequence list
#         (False, ["A"], ["0,A\n"]),  # Single nucleotide
#     ],
# )
# def test_write_sequences_to_csv(setup_csv_file, is_gene, sequences, expected_lines):
#     write_sequences_to_csv(is_gene, sequences)
#     with open(setup_csv_file, "r") as f:
#         lines = f.readlines()[1:]  # Skip header
#     assert lines == expected_lines


# @pytest.mark.parametrize(
#     "gene_sequences, genomes, expected_length",
#     [
#         (["ATGC", "CGTA"], [SeqRecord(Seq("ATGCGTACGT"), id="genome1")], 2),
#         ([], [SeqRecord(Seq("ATGCGTACGT"), id="genome1")], 0),  # No gene sequences
#         (["ATGC"], [SeqRecord(Seq("ATGCGTACGT"), id="genome1")], 1),
#     ],
# )
# def test_generate_non_genes(gene_sequences, genomes, expected_length):
#     non_genes = generate_non_genes(gene_sequences, genomes)
#     assert len(non_genes) == expected_length
#     for non_gene in non_genes:
#         assert non_gene not in gene_sequences


# @pytest.mark.parametrize(
#     "gene_sequences, expected_lengths",
#     [
#         (["ATGC", "CGTA", "ATGCGT"], [4, 4, 6]),
#         ([], []),  # No gene sequences
#         (["A"], [1]),  # Single nucleotide
#     ],
# )
# def test_get_gene_length_distribution(gene_sequences, expected_lengths):
#     gene_lengths = get_gene_length_distribution(gene_sequences)
#     assert gene_lengths == expected_lengths


# @pytest.mark.parametrize(
#     "gene_sequences, possible_lengths",
#     [
#         (["ATGC", "CGTA", "ATGCGT"], [4, 6]),
#         (["A"], [1]),  # Single nucleotide
#         ([], []),  # No gene sequences
#     ],
# )
# def test_get_random_length_from_gene_distribution(gene_sequences, possible_lengths):
#     if gene_sequences:
#         random_length = get_random_length_from_gene_distribution(gene_sequences)
#         assert random_length in possible_lengths
#     else:
#         with pytest.raises(IndexError):
#             get_random_length_from_gene_distribution(gene_sequences)


# @pytest.mark.parametrize(
#     "gene_sequences, genome, possible_lengths",
#     [
#         (["ATGC", "CGTA"], "ATGCGTACGT", [4, 6]),
#         (["A"], "A", [1]),  # Single nucleotide
#         (["ATGC"], "ATGC", [4]),  # Exact match
#     ],
# )
# def test_get_random_gene_frame(gene_sequences, genome, possible_lengths):
#     random_gene_frame = get_random_gene_frame(gene_sequences, genome)
#     assert len(random_gene_frame) in possible_lengths


# def test_save_genes_to_csv(setup_csv_file):
#     genes = [SeqRecord(Seq("ATGC"), id="gene1"), SeqRecord(Seq("CGTA"), id="gene2")]
#     gene_sequences = save_genes_to_csv(genes)
#     with open(setup_csv_file, "r") as f:
#         lines = f.readlines()[1:]  # Skip header
#     assert lines == ["1,ATGC\n", "1,CGTA\n"]
#     assert gene_sequences == ["ATGC", "CGTA"]


# def test_save_non_genes_to_csv(setup_csv_file):
#     gene_sequences = ["ATGC", "CGTA"]
#     genomes = [SeqRecord(Seq("ATGCGTACGT"), id="genome1")]
#     non_genes = save_non_genes_to_csv(gene_sequences, genomes)
#     with open(setup_csv_file, "r") as f:
#         lines = f.readlines()[1:]  # Skip header
#     assert len(non_genes) == len(gene_sequences)
#     for non_gene in non_genes:
#         assert non_gene not in gene_sequences


# @pytest.mark.parametrize(
#     "genomes",
#     [
#         (
#             [
#                 SeqRecord(Seq("ATGCGTACGT"), id="genome1"),
#             ]
#         ),
#         (
#             [
#                 SeqRecord(Seq("ATGCGTACGT"), id="genome1"),
#                 SeqRecord(Seq("CGTACGTAGC"), id="genome2"),
#             ]
#         ),
#         (
#             [
#                 SeqRecord(Seq("ATGCGTACGT"), id="genome1"),
#                 SeqRecord(Seq("CGTACGTAGC"), id="genome2"),
#                 SeqRecord(Seq("CGTACGAAGC"), id="genome3"),
#                 SeqRecord(Seq("CGTTCGTAGC"), id="genome4"),
#             ]
#         ),
#     ],
# )
# def test_get_random_genome():
#     genomes = [
#         SeqRecord(Seq("ATGCGTACGT"), id="genome1"),
#         SeqRecord(Seq("CGTACGTAGC"), id="genome2"),
#     ]
#     random_genome = get_random_genome(genomes)
#     assert random_genome in genomes


# def test_get_random_genome_empty_list():
#     genomes = []
#     with pytest.raises(ValueError, match="The genomes list is empty."):
#         get_random_genome(genomes)


# def test_save_non_genes_to_csv_empty_genes():
#     gene_sequences = []
#     genomes = [SeqRecord(Seq("ATGCGTACGT"), id="genome1")]

#     with pytest.raises(ValueError, match="The genomes list is empty."):
#         save_non_genes_to_csv(gene_sequences, genomes)


# def test_save_non_genes_to_csv_empty_genomes():
#     gene_sequences = ["ATGC", "CGTA"]
#     genomes = []

#     with pytest.raises(ValueError, match="The genes list is empty."):
#         save_non_genes_to_csv(gene_sequences, genomes)
