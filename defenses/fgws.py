import numpy as np
from nltk.corpus import stopwords
from attackdefend import TextDefend
from tqdm import tqdm
import pandas as pd
import re


class FGWS():
    def __init__(self):
        pass
    
    def replace_low_frequency_words(self, x, delta, word_freq, synonyms):
        # Load English stopwords
        stop_words = set(stopwords.words('english'))
        
        # Tokenize the input string into words and punctuation
        tokens = re.findall(r'\w+|[^\w\s]|\s+', x)
        
        # List to store the modified tokens
        modified_sentence = []
        
        # Iterate over each token
        for token in tokens:
            # If the token is not an alphabetic word (i.e., it's punctuation or space), keep it unchanged
            if not token.isalpha():
                modified_sentence.append(token)
                continue
            
            # If the token is a stopword or not in the frequency dictionary, keep it unchanged
            if token.lower() in stop_words or token.lower() not in word_freq:
                modified_sentence.append(token)
                continue
            
            # Check the frequency of the token against the threshold
            frequency = word_freq.get(token.lower(), 0)
            if frequency < delta:
                # Replace with a synonym that has the highest frequency, if available
                if token.lower() in synonyms:
                    # Filter out synonyms that are stopwords or have no frequency data
                    candidate_synonyms = [
                        (syn, word_freq.get(syn, 0)) 
                        for syn in synonyms[token.lower()] 
                        if syn not in stop_words and syn in word_freq
                    ]
                    
                    if candidate_synonyms:
                        # Choose the synonym with the highest frequency
                        best_synonym = max(candidate_synonyms, key=lambda item: item[1])[0]
                        modified_sentence.append(best_synonym)
                    else:
                        # If no valid synonym is found, keep the original token
                        modified_sentence.append(token)
                else:
                    # If the token has no synonyms, keep the original token
                    modified_sentence.append(token)
            else:
                # If the token's frequency is above the threshold, keep it unchanged
                modified_sentence.append(token)
        
        # Join the list into a single string and return the modified sentence
        return ''.join(modified_sentence)

    
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

        # check accuracy on original
        df['original_replaced'] = df['original_text'].apply(lambda x: self.replace_low_frequency_words(x, delta, word_freq, union_group))
        df[['original_replaced_output_score', 'original_replaced_output_label']] = df['original_replaced']\
            .apply(lambda x: textdefend.get_prediction_and_score(x))\
            .apply(pd.Series)

        # Append the results to the DataFrame
        df['original_vs_original_replaced_label_diff'] = df['original_replaced_output_label'] != df['original_predicted_label']
        df['original_replaced_label_correct'] = df['original_replaced_output_label'] == df['ground_truth_label']
        df['original_replaced_score_diff'] = df['original_prediction_score'] - df['original_replaced_output_score']
        df['predict_original_as_attack'] = df['original_replaced_score_diff'] > gamma


        return df
        


    def find_delta_by_percentile(self, percentile, defense_input):
        # Calculate the delta that corresponds to the given percentile of word frequencies
        frequencies = sorted([freq for word, freq in defense_input.word_freq.items() if word in defense_input.union_group])
        return np.percentile(frequencies, percentile)

    def find_best_delta_gamma(self, percentiles, defense_input, dataframe, textdefend):
        """
        Find the best delta that maximizes restored accuracy while keeping false positives below the threshold.
        """
        best_delta = None
        best_gamma = None
        best_restored_accuracy = 0
        df = dataframe.copy()
        word_freq = defense_input.word_freq
        union_group = defense_input.union_group

        for percentile in tqdm(percentiles):
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
