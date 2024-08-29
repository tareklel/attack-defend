import textattack
from textattack import Attacker, AttackArgs
from textattack.attack_recipes import PWWSRen2019, BAEGarg2019, TextFoolerJin2019
import pandas as pd
import numpy as np
from tqdm import tqdm
from textattack.models.wrappers import PyTorchModelWrapper


class TextDefend():
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_wrapper = self.init_model()

    def init_model(self):
        # get type of model
        model_type = self.model_name.split('-')[0]
        if model_type == 'lstm':
            lstm_model = textattack.models.helpers.LSTMForClassification.from_pretrained(
                self.model_name)
            model_wrapper = PyTorchModelWrapper(
                lstm_model, lstm_model.tokenizer)
            return model_wrapper
        else: 
            # raise error modelnotfound
            raise ModuleNotFoundError

    def init_attack(self, attack):
        self.attack = attack.build(self.model_wrapper)
    
    def set_up_attacker(self, dataset, num_attacks):
        self.attack_args = AttackArgs(
            num_examples=num_attacks, 
            disable_stdout=True,  # Disable output to console for cleaner display
        )
        self.attacker = Attacker(self.attack, dataset, self.attack_args)
    
    def get_attack_results(self):
        self.result = self.attacker.attack_dataset()