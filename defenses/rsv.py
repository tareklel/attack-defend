from attackdefend import TextDefend
import numpy as np
import random
import nltk
from nltk.corpus import stopwords
import string
import re
import pandas as pd
from collections import Counter
from helpers import assess_defense


class RSV():
    def __init__(self, textdefend, defense_input, number_of_votes,):
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

        # Tokenize everything: words, punctuation, and spaces.
        tokens = re.findall(r'\w+|[^\w\s]|\s+', text)

        # Find eligible words for substitution (exclude punctuation, spaces, and common words)
        eligible_words = set(
            token for token in tokens
            if token.isalpha()  # We only want alphabetic words, excluding punctuation and spaces
            and token.lower() in self.defense_input.grouped_words.keys()  # Ensure the word has a synonym
            and len(self.defense_input.grouped_words[token.lower()]) > 0  # Exclude words with no synonyms
            and token.lower() not in self.commonwords  # Exclude common words
        )

        # Ensure we don’t select more words than are available
        num_words_to_replace = min(number_of_words, len(eligible_words))

        # Randomly select words for substitution
        words_to_replace = random.sample(sorted(eligible_words), num_words_to_replace)

        # Replace words while preserving structure
        modified_tokens = [
            self._get_random_synonym(token, seed) if token in words_to_replace else token
            for token in tokens
        ]

        # Join the tokens back into a single string
        return ''.join(modified_tokens)
    
    def _get_list_of_substitutions(self, sentence, proportion_of_words=0.25, list_length=1):
        subs =[]
        for x in range(list_length):
            number_of_words = int(round(proportion_of_words*len(sentence.split()), 0))
            subbed_text = self._apply_random_substitution(sentence, number_of_words, seed=x)
            subs.append(subbed_text)
        return list(subs)
    
    def vote_for_substitution(self, sentences=list):
        predictions = self.textdefend.get_prediction_and_score(sentences)
        if len(sentences) == 1:
            predictions = [predictions]
        prediction_score = np.mean([x[0] for x in predictions])
        predition_label = int(np.round(prediction_score,0))
        return predition_label, prediction_score
    
    def apply_rsv(self, sentence, proportion_of_words, list_length=1):
        subs = self._get_list_of_substitutions(sentence, proportion_of_words, list_length)
        return self.vote_for_substitution(subs)
    
    def apply_defense_and_reattack(self, df, proportion_of_words, list_length=1, gamma=0.5):
        # Initialize new columns to store the defense results
        df['defense_output_label'] = None
        df['defense_output_score'] = None
        df['defense_vs_attack_label_diff'] = None
        df['attack_vs_defense_score_diff'] = None
        df['predict_as_attack'] = None

        df[['defense_output_label', 'defense_output_score']] = df['perturbed_text']\
            .apply(lambda x: self.apply_rsv(x, proportion_of_words, list_length))\
            .apply(pd.Series)

        df['defense_vs_attack_label_diff'] = df['defense_output_label'] != df['predicted_perturbed_label']
        df['attack_vs_defense_score_diff'] = df['perturbed_output_score'] - df['defense_output_score']
        df['predict_as_attack'] = df['attack_vs_defense_score_diff'] > gamma

        return df

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
        top_threshold = round(len(sorted_dict.keys())*threshold)
        [self.commonwords.add(x) for x in list(sorted_dict.keys())[:top_threshold]]
        [self.commonwords.add(x) for x in stop_words]
        [self.commonwords.add(x) for x in punctuations]
    
    def greedy_search(self, df, list_number_of_votes, list_proportion_of_words):
        best_score = float('-inf')
        best_params = {}
        recommended_gamma = None

        for number_of_votes in list_number_of_votes:
            for proportion_of_words in list_proportion_of_words:
                # Apply defense and reattack with current parameter combination
                df_copy = df.copy()  # Copy the dataframe to avoid overwriting original data
                result_df = self.apply_defense_and_reattack(df_copy, proportion_of_words, number_of_votes)

                # Assess the defense using restored_accuracy
                metrics = assess_defense(result_df)
                restored_accuracy = metrics['restored_accuracy']

                # Apply RSV to the original text for the gamma calculation
                df_copy[['defense_original_output_score', 'defense_original_output_label']] = df_copy['original_text']\
                    .apply(lambda x: self.apply_rsv(x, proportion_of_words, number_of_votes))\
                    .apply(pd.Series)

                # Calculate gamma
                cond = (df_copy['ground_truth_label'] == df_copy['original_predicted_label'])
                df_copy.loc[cond, 'gamma'] = df_copy['original_prediction_score'] - df_copy['defense_original_output_score']

                # Get the 90th percentile of gamma
                sorted_gammas = df_copy[df_copy['gamma'].notna()]['gamma'].sort_values()
                gamma = np.percentile(sorted_gammas, 90)


                # Update the best score and parameters if the current score is better
                if restored_accuracy > best_score:
                    best_score = restored_accuracy
                    best_params = {
                        'number_of_votes': number_of_votes,
                        'proportion_of_words': proportion_of_words,
                        'gamma':gamma
                    }

        # Return the best parameters, restored accuracy, and recommended gamma
        return best_params, best_score
        