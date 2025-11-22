# Evaluation — Trigram Language Model (one-page summary)

Design goals
- Build a clear, correct trigram (N=3) model from scratch that can (1) train on plain text, (2) handle rare/unknown words, and (3) generate text by probabilistic sampling.

Data structures
- Trigram counts are stored in a nested dictionary keyed by the two-word prefix: `counts[(w1, w2)][w3] = count`.
	This is implemented with `defaultdict(lambda: defaultdict(int))` which makes counting concise and fast.
- Unigram frequencies are kept in `self.unigram` during training to support UNK mapping. The final vocabulary after UNK-mapping is kept in `self.vocab` and cached as a sorted list for deterministic sampling ordering.

Cleaning, tokenization, and padding
- Text is split into sentences using a simple regex on sentence enders (`. ? !`) so each sentence can be padded independently.
- Each sentence is lowercased and cleaned via `re.sub(r"[^\w\s]", "", ...)` to remove punctuation while preserving words and whitespace. This keeps tokenization simple and predictable for the assignment scope.
- For trigram contexts we pad each sentence with two `'<START>'` tokens at the front and a single `'<END>'` token at the end. Padding per-sentence prevents trigrams from crossing sentence boundaries and preserves sentence-level generation behavior.

Unknown words (UNK)
- After building unigram frequencies we map rare tokens to the special `'<UNK>'` token. The mapping threshold is controlled by `unk_threshold` passed to `fit()`.
	For example, `unk_threshold=1` maps hapax legomena (tokens with frequency 1) to `'<UNK>'` while `unk_threshold=0` keeps all tokens. This design provides flexibility: tests and demonstrations can disable UNK mapping while real data runs can enable it.

Probability estimation and smoothing
- Raw trigram counts are converted to sampling weights when generating. To avoid zero-probability outcomes for unseen next words, the generator applies add-
	alpha (Laplace) smoothing by adding a `smoothing` constant (default 1.0) to every candidate count before sampling. This ensures unseen words still have non-zero mass and supports more varied generation.

Generation
- Generation starts from the context `['<START>', '<START>']` and repeatedly samples the next word conditioned on the last two tokens.
- For a given prefix `(w1,w2)` the code retrieves `counts[(w1,w2)]` (may be missing). It then constructs a weight vector over the whole vocabulary where each candidate weight = count + smoothing. Sampling uses `random.choices` with these weights.
- Generation stops when `'<END>'` is selected or when `max_length` is reached.

Trade-offs and notes
- Backoff: This implementation does not implement an explicit backoff/interpolation to bigram/unigram models. Instead the Laplace smoothing over the full vocabulary ensures unseen trigrams can still be sampled (though a principled backoff would often produce better fluency).
- Tokenization: The tokenizer is intentionally simple (regex-based). For production or more fluent results, using a more sophisticated tokenizer (preserving contractions, handling apostrophes, or using sentence segmentation libraries) is recommended.
- Determinism: The vocabulary list is sorted when cached to provide a deterministic ordering for sampling weights (helpful for tests). Randomness still comes from `random.choices`.

How to reproduce
- Train: `model.fit(text, unk_threshold=1)`
- Generate: `model.generate(max_length=50, smoothing=1.0)`

This document summarizes the main design choices made to meet the assignment requirements: trigram counts, sentence-level padding, unknown-word mapping, and probabilistic generation with Laplace smoothing.
