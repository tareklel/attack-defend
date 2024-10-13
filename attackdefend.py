import textattack
from textattack import Attacker, AttackArgs
from textattack.attack_recipes import PWWSRen2019, BAEGarg2019, TextFoolerJin2019
import pandas as pd
import numpy as np
from tqdm import tqdm
from textattack.models.wrappers import PyTorchModelWrapper
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from textattack.models.wrappers import HuggingFaceModelWrapper


class TextDefend():
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available
        self.model_wrapper = self.init_model()

    def check_model_type(self):
        model_name = self.model_name.lower()  # Make the string lowercase to handle case-insensitive matching
        if 'bert' in self.model_name:
            return 'bert'
        elif 'cnn' in self.model_name:
            return 'cnn'
        elif 'lstm' in self.model_name:
            return 'lstm'
        else:
            return 'unknown'

    def init_model(self):
        model_type = self.check_model_type()
        
        if model_type == 'lstm':
            lstm_model = textattack.models.helpers.LSTMForClassification.from_pretrained(
                self.model_name)
            lstm_model = lstm_model.to(self.device)  # Move model to GPU if available
            model_wrapper = PyTorchModelWrapper(
                lstm_model, lstm_model.tokenizer)
            return model_wrapper
        
        elif model_type == 'cnn':
            cnn_model = textattack.models.helpers.WordCNNForClassification.from_pretrained(self.model_name)
            cnn_model = cnn_model.to(self.device)  # Move model to GPU if available
            model_wrapper = PyTorchModelWrapper(
                cnn_model, cnn_model.tokenizer)
            return model_wrapper
        
        elif model_type == 'bert':
            # Load the pretrained BERT model and tokenizer
            bert_model = BertForSequenceClassification.from_pretrained(self.model_name)
            bert_model = bert_model.to(self.device)  # Move model to GPU if available
            tokenizer = BertTokenizer.from_pretrained(self.model_name)

            # Wrap the model with TextAttack's HuggingFaceModelWrapper
            model_wrapper = HuggingFaceModelWrapper(bert_model, tokenizer)
            return model_wrapper
        else:
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
    
    def get_prediction_and_score(self, text: list):
        # Ensure the input is a list of texts
        if isinstance(text, str):
            text = [text]

        # Move inputs to the device (GPU/CPU)
        evals = self.model_wrapper(text)
        output = []

        # Iterate over the logits for each input text
        for logits in evals:
            # Apply exponentiation to logits and calculate softmax for binary classification
            exps = np.exp(logits)
            prediction_score = np.round(exps[1] / exps.sum(), 6)
            
            # Determine the predicted label (0 or 1) based on the prediction score
            predicted_label = int(np.round(prediction_score))
            
            # Append the result as a tuple (score, label) to the output list
            output.append((prediction_score, predicted_label))

        # Return the result, either as a single tuple or a list of tuples
        return output[0] if len(output) == 1 else output
