import random
from textattack.datasets import HuggingFaceDataset
import pandas as pd
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import string
import pickle
import os
from annoy import AnnoyIndex
from tqdm import tqdm
import numpy as np
from datasets import DatasetDict, Dataset



class DatasetManager:
    def __init__(self, dataset_name, subset=None, seed=42):
        # Load the dataset using TextAttack's HuggingFaceDataset class
        self.train_dataset = HuggingFaceDataset(dataset_name, subset=subset, split='train')
        self.test_dataset = HuggingFaceDataset(dataset_name, subset=subset, split='test')
        self.seed = seed  # Store the seed for consistency
        self.train_set = None
        self.validation_set = None
        self.test_set = None

    def get_class_distribution(self):
        # Get the distribution of classes in the dataset
        def count_labels(dataset):
            label_count = {}
            for i in range(len(dataset._dataset)):
                _, label = dataset[i]
                if label in label_count:
                    label_count[label] += 1
                else:
                    label_count[label] = 1
            return label_count

        train_distribution = count_labels(self.train_dataset)
        test_distribution = count_labels(self.test_dataset)
        
        return train_distribution, test_distribution

    def create_train_set(self, output_class=None, num_examples=None):
        # Create a training set with the specified number of examples and output class
        train_subset = [self.train_dataset[i] for i in range(len(self.train_dataset._dataset))
                        if output_class is None or self.train_dataset[i][1] == output_class]
        random.seed(self.seed)  # Set the random seed for reproducibility
        random.shuffle(train_subset)
        train_subset = train_subset[:num_examples] if num_examples else train_subset

        # Return a new HuggingFaceDataset instance with the subset
        train_data = Dataset.from_dict({
            'text': [x[0]['text'] for x in train_subset],
            'label': [x[1] for x in train_subset]
        })
        self.train_set = HuggingFaceDataset(train_data)
        return self.train_set

    def create_validation_set(self, output_classes=None, num_examples=None):
        # Create validation sets for each class and assign them to separate instance variables
        if output_classes is None:
            output_classes = list(set([self.test_dataset[i][1] for i in range(len(self.test_dataset._dataset))]))

        self.validation_texts = set()  # To store all validation texts across classes

        for output_class in output_classes:
            # Gather examples for the current class
            class_examples = [self.test_dataset[i] for i in range(len(self.test_dataset._dataset))
                            if self.test_dataset[i][1] == output_class]
            random.seed(self.seed)  # Set the random seed for reproducibility
            random.shuffle(class_examples)

            # Limit the number of examples for each class to num_examples
            if num_examples:
                class_examples = class_examples[:num_examples]

            # Create the validation set for the current class
            validation_data = Dataset.from_dict({
                'text': [x[0]['text'] for x in class_examples],
                'label': [x[1] for x in class_examples]
            })

            # Add validation texts to the set to avoid overlap with the test set
            self.validation_texts.update([x[0]['text'] for x in class_examples])

            # Dynamically set the class-specific validation dataset as an instance variable
            setattr(self, f'validation_set_{output_class}', HuggingFaceDataset(validation_data))

        return {f'validation_set_{output_class}': getattr(self, f'validation_set_{output_class}') for output_class in output_classes}

    def create_test_set(self, output_classes=None, num_examples=None):
        # Create test sets for each class and assign them to separate instance variables, excluding validation examples
        if output_classes is None:
            output_classes = list(set([self.test_dataset[i][1] for i in range(len(self.test_dataset._dataset))]))

        for output_class in output_classes:
            # Gather examples for the current class, excluding those in the validation set
            class_examples = [self.test_dataset[i] for i in range(len(self.test_dataset._dataset))
                            if self.test_dataset[i][1] == output_class and self.test_dataset[i][0]['text'] not in self.validation_texts]
            random.seed(self.seed)  # Set the random seed for reproducibility
            random.shuffle(class_examples)

            # Limit the number of examples for each class to num_examples
            if num_examples:
                class_examples = class_examples[:num_examples]

            # Create the test set for the current class
            test_data = Dataset.from_dict({
                'text': [x[0]['text'] for x in class_examples],
                'label': [x[1] for x in class_examples]
            })

            # Dynamically set the class-specific test dataset as an instance variable
            setattr(self, f'test_set_{output_class}', HuggingFaceDataset(test_data))

        return {f'test_set_{output_class}': getattr(self, f'test_set_{output_class}') for output_class in output_classes}



def process_analytical_dataset(results):
    # results is a list of AttackResult objects obtained after running the attack
    results_list = []
    # Process each result in the `results`
    for result in results:
        original_text = result.original_text()
        perturbed_text = result.orginal_text() if result.perturbed_text(
        ) is None else result.perturbed_text()
        original_label = result.original_result.ground_truth_output
        predicted_label = result.original_result.output
        predicted_perturbed_label = result.original_result.output if result.perturbed_result.output is None else result.perturbed_result.output
        original_output_score = result.original_result.score
        perturbed_output_score = result.original_result.score if result.perturbed_result.score is None else result.perturbed_result.score
        skip = (original_label != predicted_label)
        attack_success = (predicted_label != predicted_perturbed_label) and (
            original_label == predicted_label)

        # Initialize the dictionary to store swapped words
        swapped_words_dict = {}

        # Extract modified indices from the attack attributes
        modified_indices = result.perturbed_result.attacked_text.attack_attrs.get(
            'modified_indices', [])

        for index in modified_indices:
            original_word = result.original_result.attacked_text.words[index]
            perturbed_word = result.perturbed_result.attacked_text.words[index]
            swapped_words_dict[index] = [original_word, perturbed_word]

        number_of_swapped_words = len(swapped_words_dict)

        # Append the result to the list as a dictionary
        results_list.append({
            'dataset': 'imdb',  # Assuming the dataset is IMDb, adjust as necessary
            'model': 'lstm-imdb',  # Assuming the model is LSTM trained on IMDb, adjust as necessary
            'attack_name': 'PWWSRen2019',  # Assuming the attack used is PWWSRen2019
            'original_text': original_text,
            'perturbed_text': perturbed_text,
            'ground_truth_label': original_label,
            'original_predicted_label': predicted_label,
            'predicted_perturbed_label': predicted_perturbed_label,
            'original_model_fail_prediction': skip,
            'attack_success': attack_success,
            'predicted_label': predicted_label,
            'swapped_words_dict': swapped_words_dict,
            'number_of_swapped_words': number_of_swapped_words,
            'original_prediction_score': original_output_score,
            'perturbed_output_score': perturbed_output_score
        })

    # Convert the results list into a DataFrame
    df = pd.DataFrame(results_list)
    return df


def assess_defense(df):
    original_accurate = df['ground_truth_label'] == df['original_predicted_label']
    original_accuracy = original_accurate.sum() / len(df['ground_truth_label'])
    original_positive_accuracy = ((df['ground_truth_label'] == 1) & (df['original_predicted_label'] == 1)).sum()/(df['ground_truth_label'] == 1).sum()
    adversarial_accurate = df['ground_truth_label'] == df['predicted_perturbed_label']
    adversarial_accuracy = adversarial_accurate.sum() / len(df[df['predicted_perturbed_label'].notna()]['ground_truth_label'])
    defense_accurate = df['defense_output_label'] == df['ground_truth_label']
    restored_accuracy = defense_accurate.sum() / len(df[df['predicted_perturbed_label'].notna()]['ground_truth_label'])
    restored_accuracy_delta = restored_accuracy - adversarial_accuracy
    # positive here means is adversarial
    negative_tp = (df['ground_truth_label'] == 0) & (df['defense_output_label'] == 0)
    negative_fp = (df['ground_truth_label'] == 1) & (df['original_replaced_output_label'] == 0)
    negative_precision = negative_tp.sum() / (negative_tp.sum() + negative_fp.sum())
    negative_fn = (df['ground_truth_label'] == 0)  & (df['defense_output_label'] == 1)
    negative_recall = negative_tp.sum() / (negative_tp.sum() + negative_fn.sum()) 
    f1_negative = 2*(negative_precision * negative_recall)/(negative_precision + negative_recall)

    # adversarial
    # adversarial_fn = ~(df['predict_as_attack']) & (df['attack_success'])
    # adversarial_fnr = adversarial_fn.sum() / df['attack_success'].sum()
    # recall tp / (tp+fn)
    # adversarial_tp = (df['predict_as_attack']) & (df['attack_success'])
    # adversarial_recall = adversarial_tp.sum() / df['attack_success'].sum()
    # precision tp / (tp+fp)
    # adversarial_precision = adversarial_tp.sum() / df['predict_as_attack'].sum()
    # adversarial_f1 = 2*(adversarial_precision*adversarial_recall)/(adversarial_precision + adversarial_recall)

    # defense
    # df['attack_success_original_success'] = (df['attack_success'].apply(lambda x: True if x==1 else False)) & ~df['original_model_fail_prediction'].apply(lambda x: False if x==0 else True)
    # defense_fn = ~(df['predict_as_attack']) & df['attack_success_original_success']
    # defense_fnr = defense_fn.sum() / df['attack_success_original_success'].sum()
    # recall tp / (tp+fn)
    # defense_tp = (df['predict_as_attack']) & (df['attack_success_original_success'])
    # defense_recall = defense_tp.sum() / df['attack_success_original_success'].sum()
    # precision tp / (tp+fp)
    # df['predict_attack_original_success'] = (df['predict_as_attack'].apply(lambda x: True if x==1 else False)) & ~df['original_model_fail_prediction'].apply(lambda x: False if x==0 else True)
    # defense_precision = defense_tp.sum() / df['predict_attack_original_success'].sum()
    # defense_f1 = 2*(defense_precision*defense_recall)/(defense_precision + defense_recall)

    benign_accuracy = df['original_replaced_label_correct'].sum() / len(df['ground_truth_label'])

    return {
        'original_accuracy':original_accuracy,
        'original_positive_accuracy':original_positive_accuracy,
        'benign_accuracy':benign_accuracy,
        'adversarial_accuracy':adversarial_accuracy,
        'restored_accuracy':restored_accuracy,
        'negative_precision':negative_precision,
        'negative_recall':negative_recall,
        'f1_negative':f1_negative,
        # delta between restored_accuracy and adversarial_accuracy
        'restored_accuracy_delta':restored_accuracy_delta,
        # check if adversarial incorrectly labeled as non-adversarial
        # 'defense_false_negative_rate':defense_fnr,
        # 'defense_recall':defense_recall,
        # 'defense_precision':defense_precision,
        # 'defense_f1-score':defense_f1
    }


def extract_text_from_dataset(dataset, text_index=0):
    """
    Extracts the text data from a HuggingFaceDataset and returns it as a list of strings.

    Args:
        dataset (HuggingFaceDataset): The dataset from which to extract text.
        text_column (str): The name of the column containing the text data.

    Returns:
        list: A list of text strings.
    """
    # Extract the text column
    text_data = [example[text_index]['text'] for example in dataset]
    return text_data


def count_word_frequencies(corpus):
    # Ensure stopwords are downloaded
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    # Initialize a counter
    word_freq = Counter()

    # Define stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    punctuations = set(string.punctuation)

    for document in corpus:
        # Tokenize the document
        words = nltk.word_tokenize(document.lower())

        # Filter out stopwords and punctuation
        filtered_words = [
            word for word in words if word not in stop_words and word not in punctuations]

        # Update the word frequency counter
        word_freq.update(filtered_words)

    return dict(word_freq)


def save_to_pickle(data, file_path):
    # Function to save data to a pickle file
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def load_from_pickle(file_path):
    # Function to load data from a pickle file
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


class DefenseInput():
    def __init__(self):
        self.text_data = []
        self.word_freq = {}

    def extract_text_from_dataset(self, pickle_file_path, corpus=None):
        # Check if the pickle file exists
        if os.path.exists(pickle_file_path):
            # Load data from the pickle file
            self.text_data = load_from_pickle(pickle_file_path)
            print("Loaded data from pickle.")
        else:
            # If the pickle file doesn't exist, generate the data
            self.text_data = extract_text_from_dataset(corpus)

            # Save the generated data to a pickle file
            save_to_pickle(self.text_data, pickle_file_path)

    def count_word_frequencies(self, pickle_file_path):
        # Check if the pickle file exists
        if os.path.exists(pickle_file_path):
            # Load data from the pickle file
            self.word_freq = load_from_pickle(pickle_file_path)
            print("Loaded data from pickle.")
        else:
            # If the pickle file doesn't exist, generate the data
            self.word_freq = count_word_frequencies(self.text_data)

            # Save the generated data to a pickle file
            save_to_pickle(self.word_freq, pickle_file_path)

    def load_glove_embeddings(self, glove_file_path):
        embeddings = {}
        with open(glove_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[word] = vector
        self.embeddings = embeddings

    # Function to build the Annoy index with more accuracy
    def build_annoy_index(self, num_trees=50):
        dim = len(next(iter(self.embeddings.values())))
        annoy_index = AnnoyIndex(dim, 'angular')

        # Add all word vectors to the Annoy index
        for i, (word, vector) in enumerate(self.embeddings.items()):
            annoy_index.add_item(i, vector)

        # Build the index with a larger number of trees for more accuracy
        annoy_index.build(num_trees)
        return annoy_index, list(self.embeddings.keys())

    def get_annoy_neighbors(self, annoy_index, word_list, word, n_neighbors=8, search_k=-1):
        if word not in self.embeddings:
            return []

        word_index = word_list.index(word)

        # Adjust search_k to make the search more exhaustive
        if search_k == -1:
            # A reasonable default if not specified
            search_k = len(word_list) * n_neighbors

        nearest_indices = annoy_index.get_nns_by_item(
            word_index, n_neighbors + 1, search_k=search_k)

        # Exclude the word itself from its neighbors
        nearest_indices = [idx for idx in nearest_indices if idx != word_index]

        nearest_neighbors = [word_list[idx]
                             for idx in nearest_indices[:n_neighbors]]
        return nearest_neighbors

    # Function to group words by neighbors with Annoy and tqdm
    def group_words_by_annoy_neighbors(self, n_neighbors=8, num_trees=50, search_k=-1):
        grouped_words = {}

        # Build the Annoy index with more trees for better accuracy
        annoy_index, word_list = self.build_annoy_index(num_trees=num_trees)

        # Find nearest neighbors using the Annoy index with tqdm progress bar
        for word in tqdm(self.embeddings.keys(), desc="Finding Neighbors"):
            neighbors = self.get_annoy_neighbors(
                annoy_index, word_list, word, n_neighbors, search_k=search_k)
            grouped_words[word] = neighbors

        self.grouped_words = grouped_words

    def load_grouped_words(self, pickle_file_path, n_neighbors=8, num_trees=50, search_k=-1):
        # Check if the pickle file exists
        if os.path.exists(pickle_file_path):
            # Load data from the pickle file
            self.grouped_words = load_from_pickle(pickle_file_path)
            print("Loaded data from pickle.")
        else:
            # If the pickle file doesn't exist, generate the data
            self.grouped_words = self.group_words_by_annoy_neighbors(
                n_neighbors, num_trees, search_k)

            # Save the generated data to a pickle file
            save_to_pickle(self.text_data, pickle_file_path)

    def get_wordnet_synonyms(self, word):
        synonyms = set()
        for synset in wn.synsets(word):
            for lemma in synset.lemmas():
                # Add lemma names (words) as synonyms
                synonyms.add(lemma.name().lower())
        return synonyms

    def union_wordnet_neighbors(self):
        self.union_group={}
        for word in tqdm(self.grouped_words):
            synonyms = set(self.get_wordnet_synonyms(word))
            self.union_group[word] = list(
                set(self.grouped_words[word]).union(synonyms))
