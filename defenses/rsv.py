from attackdefend import TextDefend
import numpy as np
import random
import nltk
from nltk.corpus import stopwords
import string



class RSV():
    def __init__(self, textdefend, defense_input, number_of_votes,):
        self.seed_iterations = [x for x in range(number_of_votes)]
        self.defense_input = defense_input
        self.textdefend = textdefend
        self.commonwords = set()

    def _get_random_synonym(self, word, seed=42):
        random.seed(seed)
        substitutes =self.defense_input.grouped_words[word.lower()]
        synonym  = random.choice(sorted(set(substitutes)))
        return synonym
    
    def _apply_random_substitution(self, text, number_of_words, seed=42):
        random.seed(seed)
        words = text.split()
        eligible_words = set(
        word for word in words
        if word.isalpha()                        # Exclude punctuation/numbers
        and word.lower() in self.defense_input.grouped_words.keys()
        and len(self.defense_input.grouped_words[word.lower()]) > 0               # Exclude words with no synonyms
        and word.lower() not in self.commonwords # Exclude words in self.commonwords

        )

        # Ensure we don’t select more words than available
        num_words_to_replace = min(number_of_words, len(eligible_words))

        # Randomly select words for substitution
        words_to_replace = random.sample(sorted(eligible_words), num_words_to_replace)

        # Replace the words while keeping the structure intact
        modified_words = [
            self._get_random_synonym(word) if word in words_to_replace else word
            for word in words
        ]
        
        # Join and return the modified sentence
        return ' '.join(modified_words)

    def get_stop_words(self, threshold=0.02):
        # get the top threshold % of words based on frequency in the text and add to stopwords
        nltk.download('stopwords')
        nltk.download('punkt_tab')

        # Define stopwords and punctuation
        stop_words = set(stopwords.words('english'))
        punctuations =set(string.punctuation)

        sorted_dict = {k:v for k, v in sorted(
            self.defense_input.word_freq.items(), 
            key=lambda item: item[1], 
            reverse=True
            )}
        top_threshold = round(len(sorted_dict.keys())*0.02)
        [self.commonwords.add(x) for x in list(sorted_dict.keys())[:top_threshold]]
        [self.commonwords.add(x) for x in stop_words]
        [self.commonwords.add(x) for x in punctuations]
            
    