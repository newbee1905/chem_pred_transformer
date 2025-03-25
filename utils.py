import itertools

def chunks(iterable, chunk_size):
	it = iter(iterable)
	while True:
		chunk = list(itertools.islice(it, chunk_size))
		if not chunk:
			break
		yield chunk
