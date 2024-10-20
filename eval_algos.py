import argparse
import os
import logging
import helpers
from helpers import DefenseInput
import pandas as pd
from attackdefend import TextDefend
from textattack.attack_recipes import PWWSRen2019, BAEGarg2019, TextFoolerJin2019
from sgrv import SGRV
from defenses.rsv import RSV
from defenses.fgws import FGWS
import numpy as np
import ast

# Set up the logger
def setup_logger(log_file_path):
    # Ensure the directory for the log file exists
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    logger = logging.getLogger('DefenseLogger')

    # Clear any existing handlers to avoid duplication issues
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    # Create a file handler for writing log to file
    file_handler = logging.FileHandler(log_file_path, mode='w')  # 'w' mode to overwrite log file each run
    file_handler.setLevel(logging.INFO)

    # Create a console handler for printing to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Flush the file handler to ensure immediate logging to the file
    file_handler.flush()

    return logger

def create_directory_for_test_set(test_set):
    # Create a directory for saving resources based on the test_set name
    dir_name = os.path.join('saved_resources', os.path.splitext(os.path.basename(test_set))[0])
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def save_defense_to_csv(dir_name, defense_data, defense_name):
    file_path = os.path.join(dir_name, f'{defense_name}.csv')
    defense_data.to_csv(file_path, index=False)
    logging.getLogger('DefenseLogger').info(f"Defense data saved to CSV: {file_path}")

def log_variables_to_file(dir_name, variables):
    log_file_path = os.path.join(dir_name, 'output.txt')
    try:
        with open(log_file_path, 'w') as file:
            for variable_name, variable_value in variables:
                if isinstance(variable_value, pd.DataFrame):  # Skip DataFrames, they are saved as CSVs
                    continue
                file.write(f"### {variable_name} ###\n")
                file.write(str(variable_value))
                file.write("\n\n")
                logging.getLogger('DefenseLogger').info(f"Saved to file: {variable_name}")
        logging.getLogger('DefenseLogger').info(f"Results have been successfully logged to: {log_file_path}")
    except Exception as e:
        logging.getLogger('DefenseLogger').error(f"Error occurred while logging to file: {e}")

def main(algorithm, validation_set, test_set, word_frequency):
    # Create directory for saving outputs based on the test set name
    save_dir = create_directory_for_test_set(test_set)
    log_file_path = os.path.join(save_dir, 'process.log')
    
    # Set up the logger
    logger = setup_logger(log_file_path)
    logger.info(f"Starting the process with algorithm: {algorithm}")

    logger.info("Step 1: Initializing the model")
    nlp_model = TextDefend(algorithm)

    logger.info("Step 2: Loading validation and test datasets")
    validate = pd.read_csv(validation_set)
    validate['scores_original'] = validate['scores_original'].apply(ast.literal_eval)
    validate['scores_perturbed'] = validate['scores_perturbed'].apply(lambda x: ast.literal_eval(x) if x is not np.nan else np.nan)

    test = pd.read_csv(test_set)
    test['scores_original'] = test['scores_original'].apply(ast.literal_eval)
    test['scores_perturbed'] = test['scores_perturbed'].apply(lambda x: ast.literal_eval(x) if x is not np.nan else np.nan)

    logger.info("Step 3: Initializing DefenseInput and loading resources")
    defense_input = DefenseInput()
    defense_input.count_word_frequencies(word_frequency)
    defense_input.load_glove_embeddings('saved_resources/glove.6B.50d.txt')
    defense_input.load_grouped_words('saved_resources/nearest_neighbors_glove_embeddings.pkl')
    defense_input.union_wordnet_neighbors()

    logger.info("Step 4: Running SGRV defense")
    sgrv = SGRV(nlp_model, defense_input)
    sgrv.get_stop_words(0)
    # Original commented-out line
    inputs = {'list_number_of_votes': [5, 7, 9], 'list_alpha': [0.5, 1, 3, 5, 10, 20, 30], 'threshold_list': [0.25, 0.33, 0.5, 1]}
    # inputs = {'list_number_of_votes': [5], 'list_alpha': [0.5], 'threshold_list': [0.25]}
    best_params_sgrv_validate, best_performance_sgrv_validate = sgrv.greedy_search(validate, **inputs)
    sgrv_defense = sgrv.apply_defense_and_reattack(test, **best_params_sgrv_validate)
    sgrv_assessment = helpers.assess_defense(sgrv_defense)
    save_defense_to_csv(save_dir, sgrv_defense, 'sgrv_defense')

    logger.info("Step 5: Running RSV defense")
    rsv = RSV(nlp_model, defense_input)
    # Original commented-out line
    best_params_rsv_validate, best_performance_rsv_validate = rsv.greedy_search(validate, [0.5, 0.6, 0.7, 0.8, 0.9, 1], [1, 2, 3, 5, 10, 25], [0, 0.01, 0.02])
    #bbest_params_rsv_validate, best_performance_rsv_validate = rsv.greedy_search(validate, [0.5], [1], [0])
    rsv_defense = rsv.apply_defense_and_reattack(test, **best_params_rsv_validate)
    rsv_assessment = helpers.assess_defense(rsv_defense)
    save_defense_to_csv(save_dir, rsv_defense, 'rsv_defense')

    logger.info("Step 6: Running FGWS defense")
    fgws = FGWS()
    # Original commented-out line
    best_params_fgws_validate, best_performance_fgws_validate = fgws.greedy_search([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], defense_input, validate, nlp_model)
    # best_params_fgws_validate, best_performance_fgws_validate = fgws.greedy_search([100], defense_input, validate, nlp_model)
    fgws_defense = fgws.apply_defense_and_reattack(test, nlp_model, defense_input, **best_params_fgws_validate)
    fgws_assessment = helpers.assess_defense(fgws_defense)
    save_defense_to_csv(save_dir, fgws_defense, 'fgws_defense')

    logger.info("Step 7: Saving specific variables to the output file")
    # Save non-DataFrame variables to a file if they match the specified suffixes
    variables_to_log = [
        ('fgws_assessment', fgws_assessment),
        ('fgws_validate', best_params_fgws_validate),
        ('rsv_assessment', rsv_assessment),
        ('rsv_validate', best_params_rsv_validate),
        ('sgrv_assessment', sgrv_assessment),
        ('sgrv_validate', best_params_sgrv_validate)
    ]
    log_variables_to_file(save_dir, variables_to_log)

    logger.info("All steps completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run defense algorithms on a test set.")
    parser.add_argument('--algorithm', type=str, required=True, help='Algorithm to use for the model')
    parser.add_argument('--validation_set', type=str, required=True, help='Path to the validation set CSV file')
    parser.add_argument('--test_set', type=str, required=True, help='Path to the test set CSV file')
    parser.add_argument('--word_frequency', type=str, required=True, help='Path to the word frequency file')

    args = parser.parse_args()
    main(args.algorithm, args.validation_set, args.test_set, args.word_frequency)

    # python3 eval_algos.py --algorithm lstm-imdb --validation_set saved_resources/mini_validate.csv --test_set saved_resources/mini_test.csv --word_frequency saved_resources/train_imdb_word_frequency.pkl
