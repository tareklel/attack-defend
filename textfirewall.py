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
from tqdm import tqdm

class TextFirewall():
    def __init__(self):
        self.commonwords = set()

    def firewall(self, scores, number_of_scores):
        results = [scores[x]['saliency'] for x in scores if x not in self.commonwords]
        results = sorted(results, key=abs, reverse=True)[:number_of_scores]
        sum_results = np.sum(results)
        return 1 if sum_results > 0 else 0

    def apply_defense(self, df, number_of_scores, stop_words=None):
        # Initialize new columns to store the defense results
        if stop_words is None:
            pass
        else:
            self.commonwords = set()
            self.get_stop_words(stop_words)

        df['defense_output_label'] = None

        df['defense_output_label'] = df['scores_perturbed'].apply(lambda x: self.firewall(x, number_of_scores) if pd.notna(x) else np.nan)
        df['original_replaced_output_label'] = df['scores_original'].apply(lambda x: self.firewall(x, number_of_scores) if pd.notna(x) else np.nan)


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
        #[self.commonwords.add(x) for x in stop_words]
        #[self.commonwords.add(x) for x in punctuations]
    
    def greedy_search(self, df, list_number_of_scores, list_stop_words):
        best_score = float('-inf')
        best_params = {}
        for stop_words in list_stop_words:
            for number_of_scores in tqdm(list_number_of_scores):
                # Apply defense and reattack with current parameter combination
                df_copy = df.copy()  # Copy the dataframe to avoid overwriting original data
                result_df = self.apply_defense(df_copy, number_of_scores, stop_words)

                # Assess the defense using restored_accuracy
                metrics = assess_defense(result_df)
                restored_accuracy = metrics['restored_accuracy']
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
                        'number_of_scores':number_of_scores,
                        'stop_words':stop_words,
                    }

                    best_performance = {
                        'f1_negative':f1_negative,
                        'negative_precision':negative_precision,
                        'negative_recall':negative_recall,
                        'restored_accuracy': restored_accuracy
                    }
                    print(best_params)
                    print(best_performance)

        # Return the best parameters, restored accuracy, and recommended gamma
        return best_params, best_score
        