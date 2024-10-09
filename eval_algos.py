import argparse
import helpers
from helpers import DefenseInput
import pandas as pd
from attackdefend import TextDefend
from textattack.attack_recipes import PWWSRen2019, BAEGarg2019, TextFoolerJin2019
from sgrv import SGRV
from defenses.rsv import RSV
from defenses.fgws import FGWS
from helpers import DefenseInput
import numpy as np
import ast


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


def main(algorithm, validation_set, test_set, word_frequency):
    # Initialize the model and dataset
    print(f"Initializing model: {algorithm}")
    nlp_model = TextDefend(algorithm)

    validate = pd.read_csv(validation_set)
    validate['scores_original'] = validate['scores_original'].apply(
        ast.literal_eval)
    validate['scores_perturbed'] = validate['scores_perturbed'].apply(
        lambda x: ast.literal_eval(x) if x is not np.nan else np.nan)

    test = pd.read_csv(test_set)

    test['scores_original'] = test['scores_original'].apply(ast.literal_eval)
    test['scores_perturbed'] = test['scores_perturbed'].apply(
        lambda x: ast.literal_eval(x) if x is not np.nan else np.nan)

    defense_input = DefenseInput()

    defense_input.count_word_frequencies(word_frequency)
    defense_input.load_glove_embeddings('saved_resources/glove.6B.50d.txt')
    defense_input.load_grouped_words(
        'saved_resources/nearest_neighbors_glove_embeddings.pkl')
    defense_input.union_wordnet_neighbors()

    sgrv = SGRV(nlp_model, defense_input)
    sgrv.get_stop_words(0)

    inputs = {
        'list_number_of_votes': [5, 7, 9],
        'list_alpha': [0.5, 1, 3, 5, 10, 20, 30],
        'threshold_list': [0.25, 0.33, 0.5, 1]
    }

    best_params_sgrv_validate, best_performance_sgrv_validate = sgrv.greedy_search(
        validate, **inputs)
    sgrv_defense = sgrv.apply_defense_and_reattack(
        test, **best_params_sgrv_validate)
    sgrv_assessment = helpers.assess_defense(sgrv_defense)

    rsv = RSV(nlp_model, defense_input)
    rsv.get_stop_words(0.02)
    best_params_rsv_validate, best_performance_rsv_validate = rsv.greedy_search(validate,
                                                                                [1, 2, 3, 5, 10, 25],
                                                                                [0.5, 0.6, 0.7,
                                                                                    0.8, 0.9, 1],
                                                                          )
    
    rsv_defense = rsv.apply_defense_and_reattack(
        test, **best_params_rsv_validate)
    rsv_assessment = helpers.assess_defense(rsv_defense)

    fgws = FGWS()

    best_params_fgws_validate, best_performance_fgws_validate =  fgws.greedy_search(
    [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 
    defense_input, 
    validate,
    nlp_model
    )

    fgws_defense = fgws.apply_defense_and_reattack(
        test, **best_params_fgws_validate
    )


