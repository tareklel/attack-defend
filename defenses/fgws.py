import numpy as np
from nltk.corpus import stopwords
from attackdefend import TextDefend
from tqdm import tqdm
import pandas as pd
import re
from helpers import assess_defense



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

        df['replaced_sentence'] = df.apply(lambda x: self.replace_low_frequency_words(x['perturbed_text'], delta, word_freq, union_group)   if x['scores_perturbed'] is not np.nan else [np.nan, np.nan], axis=1)
        df[['defense_output_score', 'defense_output_label']] = df\
            .apply(lambda x: textdefend.get_prediction_and_score(x['replaced_sentence'])   if x['scores_perturbed'] is not np.nan else [np.nan, np.nan], axis=1)\
            .apply(pd.Series)

        # Append the results to the DataFrame
        df['defense_vs_attack_label_diff'] = (df['defense_output_label'] != df['predicted_perturbed_label'])  & (df['predicted_perturbed_label'].notna())
        df['attack_vs_defense_score_diff'] = df['perturbed_output_score'] - df['defense_output_score']
        df['predict_as_attack'] = (df['attack_vs_defense_score_diff'] > gamma)  & (df['predicted_perturbed_label'].notna())

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

    def greedy_search(self, percentiles, defense_input, dataframe, textdefend):
        
        df = dataframe.copy()
        word_freq = defense_input.word_freq
        union_group = defense_input.union_group
        best_score = float('-inf')
        best_params = {}

        for percentile in tqdm(percentiles):
            delta = self.find_delta_by_percentile(percentile, defense_input)
            df_copy = df.copy()  # Copy the dataframe to avoid overwriting original data

            result_df = self.apply_defense_and_reattack(
                        df_copy, textdefend, defense_input, delta)

            # Assess the defense using f1_score
            metrics = assess_defense(result_df)
            restored_accuracy = metrics['restored_accuracy']
            negative_precision = metrics['negative_precision']
            negative_recall = metrics['negative_recall']
            f1_negative = metrics['f1_negative']

            # Calculate delta
            cond = (df_copy['ground_truth_label'] ==
                    df_copy['original_predicted_label']) & df['ground_truth_label'] == 1
            df_copy.loc[cond, 'delta'] = df_copy['original_prediction_score'] - \
                df_copy['original_replaced_output_score']

            # Get the 90th percentile of delta
            # sorted_deltas = df_copy[df_copy['delta'].notna(
            #)]['delta'].sort_values()
            # delta = np.percentile(sorted_deltas, 90)

            # Update the best score and parameters if the current score is better
            if f1_negative > best_score:
                best_score = f1_negative
                best_params = {
                    'delta': delta
                }

                best_performance = {
                    'f1_negative':f1_negative,
                    'negative_precision':negative_precision,
                    'negative_recall':negative_recall,
                    'restored_accuracy': restored_accuracy
                }
                print(best_params)
                print(best_performance)

        # Return the best parameters, restored accuracy, and recommended delta
        return best_params, best_performance
