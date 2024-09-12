import argparse
import helpers
from helpers import DatasetManager
import pandas as pd
from attackdefend import TextDefend
from textattack.attack_recipes import PWWSRen2019, BAEGarg2019, TextFoolerJin2019


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


def main(algorithm, dataset_name, attack_name):
    # Initialize the model and dataset
    print(f"Initializing model: {algorithm}")
    lstm = TextDefend(algorithm)
    
    print(f"Loading dataset: {dataset_name}")
    datamanager = DatasetManager(dataset_name)
    datamanager.create_validation_set(0, 4000)
    datamanager.create_test_set(0, 4000)
    print(f"Dataset loaded: {len(datamanager.test_set)} samples in the test set, {len(datamanager.validation_set)} samples in the validation set.")
    
    # Select the attack type
    print(f"Setting up attack: {attack_name}")
    attack_class = get_attack_class(attack_name)

    # Run the attack on the test set
    print("Running attack on test set...")
    lstm.init_attack(attack_class)
    lstm.set_up_attacker(datamanager.test_set, len(datamanager.test_set))
    lstm.get_attack_results()
    print("Attack on test set complete.")

    # Process and save the test set results
    print("Processing test set results...")
    df_test = helpers.process_analytical_dataset(lstm.result)
    file_name = f'{algorithm}_{dataset_name}_test_{attack_name}_df'.replace('/','_')
    test_file_name = f'saved_resources/{file_name}.csv'
    df_test.to_csv(test_file_name, index=False)
    print(f"Test set results saved to {test_file_name}")

    # Run the attack on the validation set
    print("Running attack on validation set...")
    lstm.init_attack(attack_class)
    lstm.set_up_attacker(datamanager.validation_set, len(datamanager.validation_set))
    lstm.get_attack_results()
    print("Attack on validation set complete.")

    # Process and save the validation set results
    print("Processing validation set results...")
    df_validation = helpers.process_analytical_dataset(lstm.result)
    file_name = f'{algorithm}_{dataset_name}_validation_{attack_name}_df'.replace('/','_')
    validation_file_name = f'saved_resources/{file_name}.csv'
    df_validation.to_csv(validation_file_name, index=False)
    print(f"Validation set results saved to {validation_file_name}")

    print("Script finished successfully!")


if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Run text attack on a specified dataset and model')
    
    # Add arguments
    parser.add_argument('--algorithm', type=str, required=True, help='The model/algorithm to use (e.g., lstm-imdb)')
    parser.add_argument('--dataset', type=str, required=True, help='The dataset to use (e.g., stanfordnlp/imdb)')
    parser.add_argument('--attack', type=str, required=True, choices=['PWWSRen2019', 'BAEGarg2019', 'TextFoolerJin2019'], help='The attack type (PWWSRen2019, BAEGarg2019, TextFoolerJin2019)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the main function with parsed arguments
    main(args.algorithm, args.dataset, args.attack)
