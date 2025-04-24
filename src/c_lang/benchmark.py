import timeit

from c_lang.build import kmers

sequence = "1234567890" * 100000000
k = 2

start_time = timeit.default_timer()
c_result = kmers.get_kmers(sequence, k)
print(f"CCode time: {timeit.default_timer() - start_time}")

start_time_python = timeit.default_timer()
result = [sequence[i : i + k] for i in range(len(sequence) - k + 1)]
print(f"Python time: {timeit.default_timer() - start_time_python}")

print(f"Result: {c_result[:10]}...")
