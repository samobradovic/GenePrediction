from typing import Iterator, List

from Bio.SeqRecord import SeqRecord

from src.data_loading.genome_fetcher import load_genomes
from src.data_loading.gene_fetcher import load_genes


def run():

    # genomes = load_genomes(load_from_disk=False)
    genes = load_genes(load_from_disk=False)

    # proteins = load_proteins_for_genomes(genomes, load_from_disk=False)

    # rna_sequences: List[SeqRecord] = transcribe_genes_into_rna(annotated_genes)

    # protein_sequences: List[str] = translate_multiple_rna_sequences(rna_sequences)

    # save_protein_sequences_to_fasta(
    #     protein_sequences, "./data/generated_data/proteins.fasta"
    # )

    # benchmark_protein_match(protein_sequences, protein_fasta)
