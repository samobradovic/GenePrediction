import random

# This is not used currently.


def mutate_string_sequence(sequence: str, mutation_rate: float) -> str:
    """
    Mutate a sequence by randomly changing characters based on the mutation rate.
    Double mutations are not allowed, meaning that if a character is mutated, it cannot
    be mutated again in the same pass.

    Args:
        sequence (str): The original sequence.
        mutation_rate (float): The probability of mutating each character in the seq.

    Returns:
        str: The mutated sequence.
    """

    if mutation_rate < 0 or mutation_rate > 1:
        raise ValueError("Mutation rate must be between 0 and 1.")
    if len(sequence) == 0:
        raise ValueError("Sequence must not be empty.")

    possible_nucleotides = "ACGT"

    number_of_mutations = int(len(sequence) * mutation_rate)

    mutated_indices = {-1}
    index_to_mutate: int = -1
    for _ in range(number_of_mutations):
        while index_to_mutate in mutated_indices:
            index_to_mutate = random.randint(0, len(sequence) - 1)
        mutated_indices.add(index_to_mutate)
        current_char = sequence[index_to_mutate]

        new_char = current_char
        while new_char == current_char:
            new_char = random.choice(possible_nucleotides)

        sequence = (
            sequence[:index_to_mutate] + new_char + sequence[index_to_mutate + 1 :]
        )

    return sequence
