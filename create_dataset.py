import argparse
import helpers
from helpers import DatasetManager
import pandas as pd
from attackdefend import TextDefend
from textattack.attack_recipes import PWWSRen2019, BAEGarg2019, TextFoolerJin2019
from sgrv import SGRV
from helpers import DefenseInput
import numpy as np


# Define a function to map attack strings to attack objects
def get_attack_class(attack_name):
    if attack_name == 'PWWSRen2019':
        return PWWSRen2019
    elif attack_name == 'BAEGarg2019':
        return BAEGarg2019
    elif attack_name == 'TextFoolerJin2019':
        return TextFoolerJin2019
    else:
        raise ValueError(f"Unknown attack type: {attack_name}")


def main(algorithm, dataset_name, attack_name, num_examples):
    # Initialize the model and dataset
    print(f"Initializing model: {algorithm}")
    nlp_model = TextDefend(algorithm)
    
    print(f"Loading dataset: {dataset_name}")
    datamanager = DatasetManager(dataset_name)
    datamanager.create_validation_set([0, 1], num_examples)
    datamanager.create_test_set([0, 1], num_examples)
    print(f"Dataset loaded")
    
    # Select the attack type
    print(f"Setting up attack: {attack_name}")
    attack_class = get_attack_class(attack_name)

    # Run the attack on the test set
    print("Running attack on test set...")
    nlp_model.init_attack(attack_class)
    nlp_model.set_up_attacker(datamanager.test_set_0, len(datamanager.test_set_0))
    nlp_model.get_attack_results()
    print("Attack on test set complete.")

    # Process and save the test set results
    print("Processing test set results...")
    df_test_0 = helpers.process_analytical_dataset(nlp_model.result)
    df_test_1 = [[datamanager.test_set_1[x][0]['text'], 
    datamanager.test_set_1[x][1]] for x in range(len(datamanager.test_set_1))]
    df_test_1 = pd.DataFrame(df_test_1, columns=['original_text', 'ground_truth_label'])
    df_test_1[['original_prediction_score', 'original_predicted_label']] = df_test_1['original_text'].apply(lambda x: nlp_model.get_prediction_and_score(x)).apply(pd.Series)
    # Specify columns to impute from df1's first row
    impute_values = df_test_0.iloc[0]  # First row of df1

    # Reindex df2 to match df1's columns (this will add NaN for missing columns)
    df_test_1 = df_test_1.reindex(columns=df_test_0.columns)

    # Impute the missing columns explicitly using the first row of df1
    for col in ['dataset', 'model', 'attack_name']:
        df_test_1[col].fillna(impute_values[col], inplace=True)

    # Concatenate df1 and df2_reindexed (this is the joined DataFrame)
    result_df_test = pd.concat([df_test_0, df_test_1], ignore_index=True)



    file_name = f'{algorithm}_{dataset_name}_test_{attack_name}_df'.replace('/','_')
    test_file_name = f'saved_resources/{file_name}.csv'
    result_df_test.to_csv(test_file_name, index=False)
    print(f"Test set results saved to {test_file_name}")

    # Run the attack on the validation set
    print("Running attack on validation set...")
    nlp_model.init_attack(attack_class)
    nlp_model.set_up_attacker(datamanager.validation_set_0, len(datamanager.validation_set_0))
    nlp_model.get_attack_results()
    print("Attack on validation set complete.")

    # Process and save the validation set results
    print("Processing validation set results...")
    df_validation_0 = helpers.process_analytical_dataset(nlp_model.result)
    df_validation_1 = [[datamanager.validation_set_1[x][0]['text'], 
    datamanager.validation_set_1[x][1]] for x in range(len(datamanager.validation_set_1))]
    df_validation_1 = pd.DataFrame(df_validation_1, columns=['original_text', 'ground_truth_label'])
    df_validation_1[['original_prediction_score', 'original_predicted_label']] = df_validation_1['original_text'].apply(lambda x: nlp_model.get_prediction_and_score(x)).apply(pd.Series)
    # Specify columns to impute from df1's first row
    impute_values = df_test_0.iloc[0]  # First row of df1

    # Reindex df2 to match df1's columns (this will add NaN for missing columns)
    df_validation_1 = df_validation_1.reindex(columns=df_validation_0.columns)

    # Impute the missing columns explicitly using the first row of df1
    for col in ['dataset', 'model', 'attack_name']:
        df_validation_1[col].fillna(impute_values[col], inplace=True)

    # Concatenate df1 and df2_reindexed (this is the joined DataFrame)
    result_df_validation = pd.concat([df_validation_0, df_validation_1], ignore_index=True)

    
    file_name = f'{algorithm}_{dataset_name}_validation_{attack_name}_df'.replace('/','_')
    validation_file_name = f'saved_resources/{file_name}.csv'
    result_df_validation.to_csv(validation_file_name, index=False)
    print(f"Validation set results saved to {validation_file_name}")

    defense_input = DefenseInput()
    if dataset_name == 'yelp_polarity':
        train_path = 'saved_resources/train_yelp_text.pkl'
        train_frequency = 'saved_resources/train_yelp_word_frequency.pkl'
    elif dataset_name == 'stanfordnlp/imdb':
        train_path = 'saved_resources/train_imdb_text.pkl'
        train_frequency = 'saved_resources/train_imdb_word_frequency.pkl'
    
    defense_input.extract_text_from_dataset(train_path, datamanager.train_set)
    defense_input.count_word_frequencies(train_frequency)

    defense_input.load_glove_embeddings('saved_resources/glove.6B.50d.txt')
    defense_input.load_grouped_words('saved_resources/nearest_neighbors_glove_embeddings.pkl')
    defense_input.union_wordnet_neighbors()
    
    sgrv = SGRV(nlp_model, defense_input)
    sgrv.get_stop_words()
    result_df_test['scores_original'] = result_df_test['original_text'].apply(lambda x: sgrv.get_scores(x))
    result_df_test['scores_perturbed'] = result_df_test['perturbed_text'].apply(lambda x: sgrv.get_scores(x) if x is not np.nan else np.nan)
    result_df_test.to_csv(test_file_name, index=False)
    print('saliency scores added for test')

    result_df_validation['scores_original'] = result_df_validation['original_text'].apply(lambda x: sgrv.get_scores(x))
    result_df_validation['scores_perturbed'] = result_df_validation['perturbed_text'].apply(lambda x: sgrv.get_scores(x) if x is not np.nan else np.nan)
    result_df_validation.to_csv(validation_file_name, index=False)
    print('saliency scores added for validation')


    print("Script finished successfully!")


if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Run text attack on a specified dataset and model')
    
    # Add arguments
    parser.add_argument('--algorithm', type=str, required=True, help='The model/algorithm to use (e.g., nlp_model-imdb)')
    parser.add_argument('--dataset', type=str, required=True, help='The dataset to use (e.g., stanfordnlp/imdb)')
    parser.add_argument('--attack', type=str, required=True, choices=['PWWSRen2019', 'BAEGarg2019', 'TextFoolerJin2019'], help='The attack type (PWWSRen2019, BAEGarg2019, TextFoolerJin2019)')
    parser.add_argument('--num_examples', type=int, required=True, help='Number of examples in validation and test')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the main function with parsed arguments
    main(args.algorithm, args.dataset, args.attack, args.num_examples)
