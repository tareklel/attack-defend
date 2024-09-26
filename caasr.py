import copy
import nltk
from nltk.corpus import stopwords
import string
import re
import torch
from transformers import BertTokenizer, BertForMaskedLM
from scipy.spatial.distance import cosine
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from helpers import assess_defense


class CAASR:
    """Context-Aware Adversarial Substitution Ranking (CAASR)"""

    def __init__(self, textdefend, defense_input):
        self.textdefend = textdefend
        self.defense_input = defense_input
        self.commonwords = set()
        self.max_freq = max(self.defense_input.word_freq.values())
        self.top_frequent_words_set = set()
        self.top_frequent_threshold = None
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
        # Or you can use a [PAD] token equivalent for your model
        modified_sentence_list[word_position] = 'the'

        # Join the modified sentence list back into a string for model prediction
        modified_sentence = ''.join(modified_sentence_list)
        # Get the model's prediction probability for the modified sentence
        modified_probability, _ = self.textdefend.get_prediction_and_score([
                                                                           modified_sentence])

        # saliency as how much probability changes if modified. we're looking for words with the highest magnitude of change
        saliency = probability - modified_probability

        return saliency  # Return the absolute difference

    def get_stop_words(self, threshold=0):
        nltk.download('stopwords')
        nltk.download('punkt_tab')

        # Define stopwords and punctuation
        stop_words = set(stopwords.words('english'))
        punctuations = set(string.punctuation)

        # sorted_dict = {k:v for k, v in sorted(
        #    self.defense_input.word_freq.items(),
        #    key=lambda item: item[1],
        #    reverse=True
        #    )}

        # top_threshold = round(len(sorted_dict.keys())*threshold)
        # [self.commonwords.add(x) for x in list(sorted_dict.keys())[:top_threshold]]
        [self.commonwords.add(x) for x in punctuations]
        [self.commonwords.add(x) for x in stop_words]

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
        original_sentence = ''.join(
            [token for token in text_list if token != ' '])

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
            word_to_token_map[i] = (
                (bert_token_index, bert_token_index + len(word_tokens) - 1))
            bert_token_index += len(word_tokens)

        # Get the start and end token positions for the word at the specified position
        start_token_position, end_token_position = word_to_token_map[word_position]

        # Mask the first token of the word in the tokenized input
        masked_input_ids = inputs.input_ids.clone()
        masked_input_ids[0,
                         start_token_position] = self.tokenizer.mask_token_id

        # Run the masked input through BERT
        with torch.no_grad():
            outputs = self.bert_model(masked_input_ids)
            predictions = outputs.logits

        # Get the token corresponding to the word at `word_position`
        original_word_tokens = self.tokenizer.tokenize(
            text_list[word_position])
        original_word_ids = [self.tokenizer.convert_tokens_to_ids(
            token) for token in original_word_tokens]
        print(original_word_tokens)
        # Calculate the probability for the first token (for simplicity)
        word_probabilities = torch.softmax(
            predictions[0, start_token_position], dim=-1)

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

    def get_softmax(self, sorted_word_info, alpha=1, beta=1, gamma=1, threshold=0):
        row = sorted_word_info.copy()
        if threshold == 0:
            pass
        else:
            top_frequent_words_set = self.get_top_frequent_words_set(threshold)
            row = {x: row[x] for x in row if row[x]['word_name'].lower(
            ) not in top_frequent_words_set.union(self.commonwords)}
        for x in row.keys():
            row[x]['frequency_normalized'] = 1 - \
                np.log(row[x]['frequency']+1)/np.log(self.max_freq)
            row[x]['rescaled_and_skewed_saliency'] = (
                (1 + row[x]['saliency'])/2)**2
            row[x]['exponent'] = np.e**(row[x]['rescaled_and_skewed_saliency'] * alpha + (
                1 - row[x]['distance_from_most_common']) * beta + row[x]['frequency_normalized'] * gamma)

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
        original_probability, _ = self.textdefend.get_prediction_and_score([
                                                                           text])

        # Initialize the dictionary to store the result
        word_info = {}

        # Filter out stopwords and punctuation, and calculate saliency and other attributes using a list comprehension
        for i, word in enumerate(text_list):
            if word.lower() not in self.commonwords and not word.isspace() and word.lower() in self.defense_input.grouped_words.keys():
                # Calculate saliency for the word at position i
                saliency = self._get_word_saliency_cnn(
                    text_list, original_probability, i)
        #        bert_prob = self.get_bert_probability(text_list, i)

                # Add other attributes here as needed, e.g., word frequency
                word_info[i] = {
                    'word_name': word,
                    'saliency': saliency,
                    #            'bert_probability': bert_prob,
                    'frequency': self.defense_input.word_freq.get(word.lower(), 0),
                    'distance_from_most_common': self.distance_from_most_common(word)
                    # You can add more attributes here as needed
                }

        # Sort the dictionary by saliency values in descending order
        sorted_word_info = dict(
            sorted(word_info.items(), key=lambda item: item[1]['saliency'], reverse=True))

        return sorted_word_info

    def substitute_words(self, tokenized_sentence, softmax_output, proportion_of_words, seed=42):
        """
        Substitute a proportion of words in a tokenized sentence based on their softmax probability and position,
        with an option to seed the random number generator for reproducibility.

        Parameters:
        - tokenized_sentence (list): A list of tokens from the sentence.
        - softmax_output (dict): Output of the get_softmax function containing word info with softmax probabilities.
        - alpha, beta, gamma (float): Hyperparameters for saliency, distance, and frequency weighting in softmax.
        - proportion_of_words (float): Proportion of words to substitute (0.0 to 1.0).
        - seed (int, optional): Seed for random number generation to ensure reproducibility.

        Returns:
        - list: The tokenized sentence with substituted words.
        """
        random.seed(seed)
        token_sentence = copy.deepcopy(tokenized_sentence)

        if len(softmax_output) == 0:
            return ''.join(token_sentence)

        # Get a list of eligible word positions (non-stopwords, non-punctuation) to consider for substitution
        eligible_word_positions = list(softmax_output.keys())

        # Determine how many words to substitute based on the proportion
        num_words_to_substitute = int(
            len(eligible_word_positions) * proportion_of_words)

        # Randomly select words for substitution, weighted by their softmax probabilities
        try:
            selected_positions = random.choices(
                eligible_word_positions,
                weights=[softmax_output[pos]['softmax']
                        for pos in eligible_word_positions],
                k=num_words_to_substitute
            )
        except IndexError:
            print(tokenized_sentence)
            print(softmax_output)
            raise IndexError


        # Substitute the selected words with a synonym from self.defense_input.grouped_words
        for pos in selected_positions:
            word = token_sentence[pos]
            synonyms = self.defense_input.grouped_words.get(word.lower(), None)

            if synonyms:
                # Randomly select a synonym
                synonym = random.choice(synonyms)
                # Replace the word in the token list with the synonym
                token_sentence[pos] = synonym

        # Return the modified sentence
        return ''.join(token_sentence)

    def get_list_of_substitutions(self, tokenized_sentence, softmax_output, proportion_of_words, list_length=1):
        subs = []
        for x in range(list_length):
            subbed_text = self.substitute_words(
                tokenized_sentence, softmax_output, proportion_of_words, seed=x)
            subs.append(subbed_text)
        return list(subs)

    def vote_for_substitution(self, sentences=list):
        predictions = self.textdefend.get_prediction_and_score(sentences)
        if len(sentences) == 1:
            predictions = [predictions]
        prediction_score = np.mean([x[0] for x in predictions])
        predition_label = int(np.round(prediction_score, 0))
        return predition_label, prediction_score

    def apply_caasr(self, sentence, scores, proportion_of_words, list_length, alpha, beta, gamma, threshold):
        softmax_output = self.get_softmax(
            scores, alpha, beta, gamma, threshold)
        tokenized_sentence = self._sentence_to_list(sentence)
        subs = self.get_list_of_substitutions(
            tokenized_sentence, softmax_output, proportion_of_words, list_length)
        return self.vote_for_substitution(subs)

    def apply_defense_and_reattack(self, df, proportion_of_words, list_length, alpha, beta, gamma, threshold, delta=0.5):
        # Initialize new columns to store the defense results
        df['defense_output_label'] = None
        df['defense_output_score'] = None
        df['defense_vs_attack_label_diff'] = None
        df['attack_vs_defense_score_diff'] = None
        df['predict_as_attack'] = None

        inputs = {
            'proportion_of_words': proportion_of_words,
            'list_length': list_length,
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'threshold': threshold

        }

        df[['defense_output_label', 'defense_output_score']] = \
            df.apply(lambda x: self.apply_caasr(
                x['perturbed_text'],
                x['scores_perturbed'],
                **inputs
            ), axis=1
        ).apply(pd.Series)

        df['defense_vs_attack_label_diff'] = df['defense_output_label'] != df['predicted_perturbed_label']
        df['attack_vs_defense_score_diff'] = df['perturbed_output_score'] - \
            df['defense_output_score']
        df['predict_as_attack'] = df['attack_vs_defense_score_diff'] > delta

        # check accuracy on original
        df[['original_replaced_output_label', 'original_replaced_output_score']] = \
            df.apply(lambda x: self.apply_caasr(
                x['original_text'],
                x['scores_original'],
                **inputs
            ), axis=1
        ).apply(pd.Series)

        # Append the results to the DataFrame
        df['original_vs_original_replaced_label_diff'] = df['original_replaced_output_label'] != df['original_predicted_label']
        df['original_replaced_label_correct'] = df['original_replaced_output_label'] == df['ground_truth_label']
        df['original_replaced_score_diff'] = df['original_prediction_score'] - \
            df['original_replaced_output_score']
        df['predict_original_as_attack'] = df['original_replaced_score_diff'] > delta

        return df

    def get_top_frequent_words_set(self, threshold):
        """
        Get or compute the set of top frequent words based on the threshold.

        Parameters:
        - threshold (int): The number of top frequent words to consider.

        Returns:
        - set: A set of the top 'threshold' frequent words.
        """
        # Check if the top_frequent_words_set already exists and if the threshold hasn't changed
        if self.top_frequent_words_set is not None and self.top_frequent_threshold == threshold:
            return self.top_frequent_words_set

        # Calculate the number of words based on the fraction
        total_words = len(self.defense_input.word_freq)
        top_n = int(total_words * threshold)

        # Get the top 'top_n' most frequent words
        top_frequent_words = sorted(self.defense_input.word_freq.items(
        ), key=lambda item: item[1], reverse=True)[:top_n]
        self.top_frequent_words_set = set(
            [word for word, freq in top_frequent_words])
        self.top_frequent_fraction = threshold  # Save the fraction

        return self.top_frequent_words_set

    def greedy_search(self, df, list_proportion_of_words, list_number_of_votes, list_alpha, list_beta, list_gamma, threshold_list):
        best_score = float('-inf')
        best_params = {}

        for number_of_votes in tqdm(list_number_of_votes):
            for proportion_of_words in list_proportion_of_words:
                for alpha in list_alpha:
                    for beta in list_beta:
                        for gamma in list_gamma:
                            for threshold in threshold_list:
                                # Apply defense and reattack with current parameter combination
                                df_copy = df.copy()  # Copy the dataframe to avoid overwriting original data
                                result_df = self.apply_defense_and_reattack(
                                    df_copy, proportion_of_words, number_of_votes, alpha, beta, gamma, threshold)

                                # Assess the defense using restored_accuracy
                                metrics = assess_defense(result_df)
                                restored_accuracy = metrics['restored_accuracy']

                                # Calculate delta
                                cond = (df_copy['ground_truth_label'] ==
                                        df_copy['original_predicted_label'])
                                df_copy.loc[cond, 'delta'] = df_copy['original_prediction_score'] - \
                                    df_copy['original_replaced_output_score']

                                # Get the 90th percentile of delta
                                sorted_deltas = df_copy[df_copy['delta'].notna(
                                )]['delta'].sort_values()
                                delta = np.percentile(sorted_deltas, 90)

                                # Update the best score and parameters if the current score is better
                                if restored_accuracy > best_score:
                                    best_score = restored_accuracy
                                    best_params = {
                                        'number_of_votes': number_of_votes,
                                        'proportion_of_words': proportion_of_words,
                                        'alpha':alpha,
                                        'beta':beta,
                                        'gamma':gamma,
                                        'word_threshold':threshold,
                                        'delta': delta
                                    }
                                    print(best_params)
                                    print(best_score)

        # Return the best parameters, restored accuracy, and recommended delta
        return best_params, best_score
