import pytest
from src.ngram_model import TrigramModel

def test_fit_and_generate():
    model = TrigramModel()
    text = "I am a test sentence. This is another test sentence."
    model.fit(text)
    generated_text = model.generate()
    assert isinstance(generated_text, str)
    assert len(generated_text.split()) > 0

def test_empty_text():
    model = TrigramModel()
    text = ""
    model.fit(text)
    generated_text = model.generate()
    assert generated_text == ""

def test_short_text():
    model = TrigramModel()
    text = "I am."
    model.fit(text)
    generated_text = model.generate()
    assert isinstance(generated_text, str)

def test_punctuation_removal():
    model = TrigramModel()
    text = "Hello, world! This is a test."
    model.fit(text)
    # Check internal counts to verify punctuation is gone
    # "Hello," should become "hello"
    # "world!" should become "world"
    # Trigram: (<START>, <START>) -> hello
    assert model.counts[('<START>', '<START>')]['hello'] > 0
    assert model.counts[('hello', 'world')]['this'] > 0

def test_case_insensitivity():
    model = TrigramModel()
    text = "The the THE"
    model.fit(text)
    # Should treat all as "the"
    # Trigrams: 
    # (<START>, <START>) -> the
    # (<START>, the) -> the
    # (the, the) -> the
    assert model.counts[('<START>', '<START>')]['the'] == 1
    assert model.counts[('<START>', 'the')]['the'] == 1
    assert model.counts[('the', 'the')]['the'] == 1

def test_repeated_words():
    model = TrigramModel()
    text = "buffalo buffalo buffalo buffalo"
    model.fit(text)
    # (buffalo, buffalo) -> buffalo
    assert model.counts[('buffalo', 'buffalo')]['buffalo'] > 0

def test_single_word():
    model = TrigramModel()
    text = "Stop."
    model.fit(text)
    # (<START>, <START>) -> stop
    # (<START>, stop) -> <END>
    assert model.counts[('<START>', '<START>')]['stop'] == 1
    assert model.counts[('<START>', 'stop')]['<END>'] == 1



