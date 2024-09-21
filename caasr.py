import copy
import nltk
from nltk.corpus import stopwords
import string
import re
import torch
from transformers import BertTokenizer, BertForMaskedLM
from scipy.spatial.distance import cosine
import numpy as np


class CAASR:
    """Context-Aware Adversarial Substitution Ranking (CAASR)"""
    def __init__(self, textdefend, defense_input):
        self.textdefend = textdefend
        self.defense_input = defense_input
        self._get_stop_words()
        self.max_freq = max(self.defense_input.word_freq.values())

        # Initialize BERT tokenizer and model for masked language modeling
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        # self.bert_model.eval()
    
    def _sentence_to_list(self, text):
        tokens = re.findall(r'\w+|[^\w\s]|\s+', text)
        return tokens

    def _get_word_saliency_cnn(self, text_list, probability, word_position):
        """
        Calculate the saliency of a word at a specific position in the sentence list for a Word-CNN model.
        
        Parameters:
        - sentence_list (list): List of tokens (words and punctuation) of the sentence.
        - probability (float): The probability output of the original sentence from the model.
        - word_position (int): The index of the word in sentence_list for which to calculate saliency.
        
        Returns:
        - float: The saliency score for the word.
        """
        # for future replace word with [MASK] for BERT and delete for LSTM
        # Create a copy of the sentence list so the original sentence remains unmodified
        modified_sentence_list = copy.deepcopy(text_list)

        # Replace the word at the specified position with a neutral word (e.g., "the")
        modified_sentence_list[word_position] = 'the'  # Or you can use a [PAD] token equivalent for your model

        # Join the modified sentence list back into a string for model prediction
        modified_sentence = ''.join(modified_sentence_list)
        # Get the model's prediction probability for the modified sentence
        modified_probability, _ = self.textdefend.get_prediction_and_score([modified_sentence])

        # saliency as how much probability changes if modified. we're looking for words with the highest magnitude of change
        saliency = probability - modified_probability

        return saliency  # Return the absolute difference

    def _get_stop_words(self):
        nltk.download('stopwords')
        nltk.download('punkt_tab')

        # Define stopwords and punctuation
        stop_words = set(stopwords.words('english'))
        punctuations =set(string.punctuation)
        self.stop = stop_words.union(punctuations)

    def get_bert_probability(self, text_list, word_position):
        """
        Get the BERT probability of a word at a specific position in a tokenized sentence list.

        Parameters:
        - text_list (list): A list of tokens (words, spaces, and punctuation) of the sentence.
        - word_position (int): The position of the word in the tokenized list (text_list).

        Returns:
        - float: The probability of the word at the given position in the sentence based on BERT.
        """
        # Join the tokenized list back into a string, skipping spaces
        original_sentence = ''.join([token for token in text_list if token != ' '])
        
        # Tokenize the sentence using BERT tokenizer
        inputs = self.tokenizer(original_sentence, return_tensors='pt')
        bert_tokens = self.tokenizer.tokenize(original_sentence)

        # Initialize a list to track token-to-BERT token mapping, ignoring spaces in text_list
        word_to_token_map = {}
        bert_token_index = 0
        
        for i, word in enumerate(text_list):
            if word == ' ':
                continue  # Skip spaces
            # Tokenize the word using BERT's tokenizer
            word_tokens = self.tokenizer.tokenize(word)
            
            # Map word in text_list to the corresponding BERT tokens
            word_to_token_map[i] = ((bert_token_index, bert_token_index + len(word_tokens) - 1))
            bert_token_index += len(word_tokens)
        
        # Get the start and end token positions for the word at the specified position
        start_token_position, end_token_position = word_to_token_map[word_position]

        # Mask the first token of the word in the tokenized input
        masked_input_ids = inputs.input_ids.clone()
        masked_input_ids[0, start_token_position] = self.tokenizer.mask_token_id

        # Run the masked input through BERT
        with torch.no_grad():
            outputs = self.bert_model(masked_input_ids)
            predictions = outputs.logits

        # Get the token corresponding to the word at `word_position`
        original_word_tokens = self.tokenizer.tokenize(text_list[word_position])
        original_word_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in original_word_tokens]
        print(original_word_tokens)
        # Calculate the probability for the first token (for simplicity)
        word_probabilities = torch.softmax(predictions[0, start_token_position], dim=-1)
        
        # Get the probability for the first token of the original word
        word_probability = word_probabilities[original_word_ids[0]].item()

        return word_probability

    def distance_from_most_common(self, word):
        common = self._find_most_common(word)
        if common is None or word is None:
            return 0.5
        return cosine(self.defense_input.embeddings[word], self.defense_input.embeddings[common])
        
        
    def _find_most_common(self, word):
        try:
            synonyms = self.defense_input.grouped_words[word]
        except KeyError:
            return None
        frequency = {}
        for x in synonyms:
            try:
                frequency[x] = self.defense_input.word_freq[x]
            except KeyError:
                pass
        if len(frequency) == 0:
            return None
        greatest = max(frequency, key=frequency.get)

        try:
            word_frequency = self.defense_input.word_freq[word]
        except KeyError:
            word_frequency = 1
        if self.defense_input.word_freq[greatest] > word_frequency:
            return greatest
        else:
            return word
        
    def get_softmax(self, sorted_word_info, alpha, beta, gamma):
        row = sorted_word_info.copy()
        for x in row.keys():
            row[x]['frequency_normalized'] = 1 - np.log(row[x]['frequency']+1)/np.log(self.max_freq)
            row[x]['exponent'] = np.e**(row[x]['saliency'] * alpha + row[x]['distance_from_most_common'] * beta + row[x]['frequency_normalized'] * gamma)

        sum_exp = sum([row[x]['exponent'] for x in row.keys()])

        for x in row.keys():
            row[x]['softmax'] = row[x]['exponent'] / sum_exp
        return row

        
    def get_scores(self, text):
        """
        Get the saliency scores and other attributes for all non-stopwords, non-punctuation words in the text.
        
        Parameters:
        - text (str): The input sentence.
        
        Returns:
        - dict: A dictionary where keys are word positions and values are sub-dictionaries containing:
                'word_name', 'saliency', and other attributes.
        """
        # Convert the text into a list of tokens
        text_list = self._sentence_to_list(text)

        # Get the model's original prediction and probability
        original_probability, _ = self.textdefend.get_prediction_and_score([text])

        # Initialize the dictionary to store the result
        word_info = {}

        # Filter out stopwords and punctuation, and calculate saliency and other attributes using a list comprehension
        for i, word in enumerate(text_list):
            if word.lower() not in self.stop and not word.isspace():
                # Calculate saliency for the word at position i
                saliency = self._get_word_saliency_cnn(text_list, original_probability, i)
        #        bert_prob = self.get_bert_probability(text_list, i)

                # Add other attributes here as needed, e.g., word frequency
                word_info[i] = {
                    'word_name': word,
                    'saliency': saliency,
        #            'bert_probability': bert_prob,
                    'frequency': self.defense_input.word_freq.get(word.lower(), 0),
                    'distance_from_most_common':self.distance_from_most_common(word)
                    # You can add more attributes here as needed
                }

        # Sort the dictionary by saliency values in descending order
        sorted_word_info = dict(sorted(word_info.items(), key=lambda item: item[1]['saliency'], reverse=True))

        return sorted_word_info
    