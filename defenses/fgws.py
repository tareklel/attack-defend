from tqdm import tqdm
import numpy as np
from nltk.corpus import stopwords
import nltk
from attackdefend import TextDefend
import copy
import helpers
from datasets import Dataset
from textattack.datasets import HuggingFaceDataset
import pandas as pd


class FGWS():
    def __init__(self):
        pass
    
    def replace_low_frequency_words(self, x, delta, word_freq, synonyms):
        """
        Replace low-frequency words in a sentence with their most frequent synonym.
        https://aclanthology.org/2021.eacl-main.13.pdf
        gamma to be determined by applying defense to non-pertrubed, gamme determined such that only 10% of non-perturbed score - defense on non-perturbed exceed threshold
        set δ equal to the frequency representing the qth percentile of all frequencies observed by the words eligible for replacement in the training set, 
        and experiment with q ∈ {0, 10, . . . , 100}

        Parameters:
        - x (str): The input sentence.
        - delta (int): The frequency threshold. Words with a frequency below this will be replaced.
        - word_freq (dict): A dictionary where keys are words and values are their frequencies.
        - synonyms (dict): A dictionary where keys are words and values are lists of synonyms.

        Returns:
        - str: The modified sentence with low-frequency words replaced by synonyms.
        """
        
        # Load English stopwords
        stop_words = set(stopwords.words('english'))
        
        # Tokenize the input string into words
        words = nltk.word_tokenize(x)
        
        # List to store the modified words
        modified_sentence = []
        
        # Iterate over each word in the sentence
        for word in words:
            # If the word is a stopword or not in the frequency dictionary, keep it unchanged
            if word.lower() in stop_words or word.lower() not in word_freq:
                modified_sentence.append(word)
                continue
            
            # Check the frequency of the word against the threshold
            frequency = word_freq.get(word.lower(), 0)
            if frequency < delta:
                # Replace with a synonym that has the highest frequency, if available
                if word.lower() in synonyms:
                    # Filter out synonyms that are stopwords or have no frequency data
                    candidate_synonyms = [
                        (syn, word_freq.get(syn, 0)) 
                        for syn in synonyms[word.lower()] 
                        if syn not in stop_words and syn in word_freq
                    ]
                    
                    if candidate_synonyms:
                        # Choose the synonym with the highest frequency
                        best_synonym = max(candidate_synonyms, key=lambda item: item[1])[0]
                        modified_sentence.append(best_synonym)
                    else:
                        # If no valid synonym is found, keep the original word
                        modified_sentence.append(word)
                else:
                    # If the word has no synonyms, keep the original word
                    modified_sentence.append(word)
            else:
                # If the word's frequency is above the threshold, keep it unchanged
                modified_sentence.append(word)
        
        # Join the list into a single string and return the modified sentence
        return ' '.join(modified_sentence)

    
    def apply_defense_and_reattack(self, df, textdefend: TextDefend, defense_input, delta=10, gamma=0.5):
        word_freq = defense_input.word_freq
        union_group = defense_input.union_group
        # Initialize new columns to store the defense results
        df['replaced_sentence'] = None
        df['defense_vs_attack_label_diff'] = None
        df['attack_vs_defense_score_diff'] = None
        df['predict_as_attack'] = None

        df['replaced_sentence'] = df['perturbed_text'].apply(lambda x: self.replace_low_frequency_words(x, delta, word_freq, union_group))
        df[['defense_output_score', 'defense_output_label']] = df['replaced_sentence']\
            .apply(lambda x: textdefend.get_prediction_and_score(x))\
            .apply(pd.Series)

        # Append the results to the DataFrame
        df['defense_vs_attack_label_diff'] = df['defense_output_label'] != df['predicted_perturbed_label']
        df['attack_vs_defense_score_diff'] = df['perturbed_output_score'] - df['defense_output_score']
        df['predict_as_attack'] = df['attack_vs_defense_score_diff'] > gamma

        return df
        


    def find_delta_by_percentile(self, percentile, defense_input):
        # Calculate the delta that corresponds to the given percentile of word frequencies
        frequencies = sorted([freq for word, freq in defense_input.word_freq.items() if word in defense_input.union_group])
        return np.percentile(frequencies, percentile)

    def find_best_delta_gamma(self, percentiles, defense_input, validation_set, textdefend):
        """
        Find the best delta that maximizes restored accuracy while keeping false positives below the threshold.
        """
        best_delta = None
        best_gamma = None
        best_restored_accuracy = 0
        textdefend.set_up_attacker(validation_set, len(validation_set))
        textdefend.get_attack_results()
        df = helpers.process_analytical_dataset(textdefend.result)
        word_freq = defense_input.word_freq
        union_group = defense_input.union_group

        for percentile in percentiles:
            delta = self.find_delta_by_percentile(percentile, defense_input)

            df['replaced_perturbed_sentence'] = df['perturbed_text'].apply(lambda x: self.replace_low_frequency_words(x, delta, word_freq, union_group))
            df['replaced_original_sentence'] = df['original_text'].apply(lambda x: self.replace_low_frequency_words(x, delta, word_freq, union_group))

            df[['defense_perturbed_output_score', 'defense_perturbed_output_label']] = df['replaced_perturbed_sentence']\
            .apply(lambda x: textdefend.get_prediction_and_score(x))\
            .apply(pd.Series)

            df[['defense_original_output_score', 'defense_original_output_label']] = df['replaced_original_sentence']\
            .apply(lambda x: textdefend.get_prediction_and_score(x))\
            .apply(pd.Series)

            # get gamma
            cond = (df['ground_truth_label'] == df['original_predicted_label'])
            df.loc[cond, 'gamma'] = df['original_prediction_score'] - df['defense_original_output_score']

            sorted_gammas = df[df.gamma.notna()]['gamma'].sort_values()
            gamma = np.percentile(sorted_gammas, 90)

            # evaluate defense
            df['is_adversarial'] = df['perturbed_output_score'] - df['defense_perturbed_output_score'] > gamma

            df['inaccurate_denominator'] = (df['ground_truth_label'] != df['predicted_perturbed_label'])&(df['ground_truth_label'] == df['original_predicted_label'])
            df['correct_prediction'] = (df['ground_truth_label'] == df['defense_perturbed_output_label'])
            df['restored_accuracy_count'] = (df['correct_prediction']) & (df['is_adversarial']) & (df['inaccurate_denominator'])
            
            restored_accuracy = df['restored_accuracy_count'].sum() / df['inaccurate_denominator'].sum()
            
            print(f"percentile:{percentile}, \
                restored_accuracy:{restored_accuracy}, \
                gamma:{gamma}")
            if restored_accuracy > best_restored_accuracy:
                best_delta = delta
                best_restored_accuracy = restored_accuracy
                best_gamma = gamma

        return best_delta, best_gamma
