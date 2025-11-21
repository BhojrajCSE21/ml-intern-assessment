import random
from collections import defaultdict
import re

class TrigramModel:
    def __init__(self):
        """
        Initializes the TrigramModel.
        """
        # Dictionary to store trigram counts: counts[(w1, w2)][w3] = count
        self.counts = defaultdict(lambda: defaultdict(int))
        
    def fit(self, text):
        """
        Trains the trigram model on the given text.

        Args:
            text (str): The text to train the model on.
        """
        # 1. Clean the text
        # Convert to lowercase and remove punctuation (keeping spaces)
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        
        # 2. Tokenize the text into words
        words = text.split()
        
        if not words:
            return

        # 3. Pad the text with start and end tokens
        # For trigram, we need 2 start tokens
        padded_words = ['<START>', '<START>'] + words + ['<END>']
        
        # 4. Count the trigrams
        for i in range(len(padded_words) - 2):
            w1 = padded_words[i]
            w2 = padded_words[i+1]
            w3 = padded_words[i+2]
            self.counts[(w1, w2)][w3] += 1

    def generate(self, max_length=50):
        """
        Generates new text using the trained trigram model.

        Args:
            max_length (int): The maximum length of the generated text.

        Returns:
            str: The generated text.
        """
        if not self.counts:
            return ""

        current_sequence = ['<START>', '<START>']
        generated_words = []
        
        for _ in range(max_length):
            w1 = current_sequence[-2]
            w2 = current_sequence[-1]
            
            possible_next_words = self.counts[(w1, w2)]
            
            if not possible_next_words:
                break
                
            # Convert counts to probabilities (or just sample based on weights)
            words = list(possible_next_words.keys())
            counts = list(possible_next_words.values())
            
            next_word = random.choices(words, weights=counts, k=1)[0]
            
            if next_word == '<END>':
                break
                
            generated_words.append(next_word)
            current_sequence.append(next_word)
            
        return " ".join(generated_words)
