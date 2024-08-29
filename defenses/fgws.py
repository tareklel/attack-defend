from tqdm import tqdm
import numpy as np
from nltk.corpus import stopwords
import nltk
from attackdefend import TextDefend
import copy
import helpers


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
        """
        Applies the defense mechanism to sentences where the attack was successful,
        and then evaluates the defended sentence using the model within the attack class.
        
        Parameters:
        - df (pd.DataFrame): The DataFrame containing the attack results.
        - attackdefend: TextDefend class containing attack
        - defense_input contains:
            word_freq (dict): A dictionary where keys are words and values are their frequencies.
            union_group (dict): A dictionary containing the union of synonyms and nearest neighbors.
        - delta (int): The frequency threshold for `replace_low_frequency_words`. Words with a frequency below this will be replaced.
        - gamma (float): The threshold for determining whether the defense is effective.
        
        Returns:
        - pd.DataFrame: The updated DataFrame with defense and re-evaluation results.
        """
        
        word_freq = defense_input.word_freq
        union_group = defense_input.union_group
        # Initialize new columns to store the defense results
        df['replaced_sentence'] = None
        df['defense_output_label'] = None
        df['defense_output_score'] = None
        df['defense_vs_attack_label_diff'] = None
        df['attack_vs_defense_score_diff'] = None
        df['predict_as_attack'] = None
        
        # Iterate over each row where the attack was successful
        for index, row in tqdm(df.iterrows()):
            # Apply the defense by replacing low-frequency words
            replaced_sentence = self.replace_low_frequency_words(row['perturbed_text'], delta, word_freq, union_group)
            df.at[index, 'replaced_sentence'] = replaced_sentence
            
            # Use the attack class's model to get the prediction for the defended sentence
            defense_results = textdefend.attack.attack(replaced_sentence, int(df.loc[index, 'ground_truth_label'])).original_result
            
            # Extract the defense output label and score
            defense_output_label = defense_results.output  
            defense_output_score = defense_results.score
            
            # Compare the defense output label with the original label
            defense_vs_attack_label_diff = (defense_output_label != row['predicted_perturbed_label'])
            
            # Calculate the difference in scores between the original model's score and the defense score
            perturbed_output_score = row['perturbed_output_score']
            attack_vs_defense_score_diff = perturbed_output_score - defense_output_score
            
            # Determine if the model still predicts the perturbed label after applying defense
            predict_as_attack = (attack_vs_defense_score_diff > gamma)
            
            # Append the results to the DataFrame
            df.at[index, 'defense_output_label'] = defense_output_label
            df.at[index, 'defense_output_score'] = defense_output_score
            df.at[index, 'defense_vs_attack_label_diff'] = defense_vs_attack_label_diff
            df.at[index, 'attack_vs_defense_score_diff'] = attack_vs_defense_score_diff
            df.at[index, 'predict_as_attack'] = predict_as_attack
        
        return df
    
    def find_delta_by_percentile(self, percentile, defense_input):
        # Calculate the delta that corresponds to the given percentile of word frequencies
        frequencies = sorted([freq for word, freq in defense_input.word_freq.items() if word in defense_input.union_group])
        return np.percentile(frequencies, percentile)

    def find_best_delta(self, percentiles, defense_input, validation_set, textdefend, threshold=0.1):
            """
            Find the best delta that maximizes restored accuracy while keeping false positives below the threshold.
            """
            best_delta = None
            best_restored_accuracy = 0
            textdefend.set_up_attacker(validation_set, len(validation_set))
            textdefend.get_attack_results()
            df = helpers.process_analytical_dataset(textdefend.result)
            word_freq = defense_input.word_freq
            union_group = defense_input.union_group

            for percentile in percentiles:
                delta = self.find_delta_by_percentile(percentile, defense_input)
                false_positive_count = 0
                total_count = 0
                restored_accuracy_count = 0
                inaccurate_denominator = 0
                correct_model_prediction = 0


                for _, row in df.iterrows():
                    # Apply perturbation using TextDefend 
                    #attack_output = textdefend.attack.attack(row.original_text, row.ground_truth_label)
                    perturbed_text = row.perturbed_text
                    perturbed_result = row.predicted_perturbed_label
                    
                    # Apply FGWS transformation with the current delta
                    transformed_text = self.replace_low_frequency_words(perturbed_text, delta, word_freq, union_group)
                    
                    # Predict the label using the model within TextDefend
                    predicted_label_transformed = textdefend.attack.attack(transformed_text, row.ground_truth_label).original_result.output
                    
                    if row.ground_truth_label != perturbed_result:
                        inaccurate_denominator += 1
                    # Compare the predicted label to the true label
                    if row.ground_truth_label == predicted_label_transformed and row.ground_truth_label != perturbed_result:
                        restored_accuracy_count += 1
                    
                    # False positive check on non-perturbed text
                    original_replace = self.replace_low_frequency_words(row.original_text, delta, word_freq, union_group)
                    original_replace_label = textdefend.attack.attack(original_replace, row.ground_truth_label).original_result.output

                    if row.original_predicted_label == row.ground_truth_label:
                        correct_model_prediction += 1

                    if original_replace_label != row.original_predicted_label and row.original_predicted_label == row.ground_truth_label:
                        false_positive_count += 1
                    
                    total_count += 1
                # Calculate the restored accuracy
                restored_accuracy = restored_accuracy_count / inaccurate_denominator
                
                # Check if this delta produces fewer false positives and better restored accuracy
                if false_positive_count / correct_model_prediction <= threshold and restored_accuracy > best_restored_accuracy:
                    best_delta = delta
                    best_restored_accuracy = restored_accuracy
                print(f"percentile:{percentile}, \
                    restored_accuracy_count:{restored_accuracy_count}, \
                    total_count:{total_count}, \
                    restored_accuracy:{restored_accuracy}, \
                    false_positive_count:{false_positive_count}, \
                    fdr:{false_positive_count / correct_model_prediction}\
                    ")

            return best_delta
