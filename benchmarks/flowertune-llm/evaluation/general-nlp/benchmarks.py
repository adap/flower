"""
This script implement a factory pattern to implement any benchmark for evaluation.
"""

import os
import json
import random
import pandas as pd

from datasets import load_dataset, Dataset, load_from_disk


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def benchmark_factory(name):
    """
    Creates a benchmark object.

    :param name: str, with the benchmark name.
    return:
    """
    # Note: benchmark is instantiated *after* selection.
    factories = {
        "medmcqa": MedMCQA,
        "pubmedqa": ClosedPubMedQA,
        "open_pubmedqa": PubMedQA,
        "medqa": MedQA,
        "medqa4": MedQA4,
        "blurb": Blurb,
        "medicationqa": MedicationQA,
        "mmlu_medical": MMLU,
        "mmlu_general": MMLU,
        "truthfulqa": TruthfulQA,
        "gsm8k": GSM8K
    }
    if name not in factories:
        raise ValueError("Benchmark {} not found. \
                         Select one of the following: {}".format(name, list(factories.keys())))
    return factories[name](name)


def load_instruction(prompt_name):
    """
    Loads the instruction for the given benchmark.

    :param benchmark: str, the name of the benchmark
    :param prompt_name: str, the name of the prompt to be used
    """
    path = os.path.join(ROOT_DIR, 'instructions.json')
    if not os.path.exists(path):
        raise FileNotFoundError('Please save the different prompts to instructions.json')

    with open(path) as f:
        prompts = json.load(f)
    return prompts[prompt_name]


class Benchmark:
    def __init__(self, name):
        """
        Class to implement a benchmark for evaluation.

        :param name: str, with the benchmark name.
        :param path: str (optional), the path to the benchmark data.
        :param splits: list of str, the splits of the data: train / test
        :param hub_name: str, the name of the HuggingFace hub dataset.
        :param dir_name: str, the name of the directory where the data is stored.
        :param train_data: HuggingFace Dataset, the train data.
        :param test_data: HuggingFace Dataset, the test data.
        :param generations: HuggingFace Dataset, the generations.
        :param subsets: list of str (optional), the subsets of the data to download from the HuggingFace hub.
        :param has_instruction: bool, whether the dataset already contains instructions.
        :param local_path: str (optional), the path to a directory holding train and test json local data files.
        """
        self.name = name
        self.path = None
        self.splits = None
        self.hub_name = None
        self.dir_name = None
        self.train_data = None
        self.test_data = None
        self.generations = None
        self.subsets = None
        self.has_instructions = False
        self.local_path = None

    def load_from_hub(self):
        """
        Downloads the benchmark data from the HuggingFace hub (for 1st time loading)
        This is specific to each benchmark and must be implemented in the extended class.
        """
        print(f'Downloading benchmark from HuggingFace hub ({self.hub_name}).')
        try:
            if self.subsets is None:
                load_dataset(self.hub_name,
                             cache_dir=os.path.join(ROOT_DIR, 'benchmarks', 'datasets'),
                             download_mode='force_redownload')
            else:
                for subset in self.subsets:
                    load_dataset(self.hub_name,
                                 subset,
                                 cache_dir=os.path.join(ROOT_DIR, 'benchmarks', 'datasets'),
                                 download_mode='force_redownload')
        except:
            raise ValueError("Default Huggingface loader failed for benchmark {}. \
                             Try implementing a custom load_from_hub function.".format(self.name))

    def load_data(self, partition='train'):
        """
        Loads benchmark data from a local directory, or from the HuggingFace hub if not yet downloaded.
        Based on the input partition type, instantiates the respective class attribute.

        :param path: str (optional), the path to the benchmark data.
        :param partition: str, the split of the data: train / test
        """
        print('='*50 + f'\nLoading data for benchmark {self.name}.\n')
        if partition not in self.splits:
            raise ValueError("Please provide a valid partition split: {}".format(self.splits))
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            self.load_from_hub()
        try:
            if self.subsets is None:
                if partition == 'train':
                    self.train_data = load_dataset(self.path, split=partition)
                elif partition in ['test', 'validation']:
                    self.test_data = load_dataset(self.path, split=partition)
            else:
                if partition == 'train':
                    self.train_data = aggregate_datasets(self.path, self.subsets, partition=partition)
                elif partition in ['test', 'validation']:
                    self.test_data = aggregate_datasets(self.path, self.subsets, partition=partition)

        except ValueError as e:
            print(e)
            raise ValueError("Couldn't load benchmark {} from local path.".format(self.name))

    def save_data(self, partition='train'):
        """
        Saves any preprocessing data partition.

        :param data: pd.DataFrame
        :param file_name: str
        """
        path = os.path.join('benchmarks', 'preprocessing', f"{self.name}_{partition}")
        print("Saving {} data to the following path: {}".format(self.name, path))
        if partition == 'train':
            pd.to_pickle(self.train_data, path)
        elif partition == 'test':
            pd.to_pickle(self.test_data, path)

    def preprocessing(self, partition='train'):
        """
        Applies a custom pre-processing over the partition.
        If instruction is provided, preprends it to the question
        Updates the train or test self attributes.

        :param _preprocess: function: dict -> dict, the preprocessing function to apply.
        :param partition: str, the split of the data: train / test
        """
        try:
            if partition == 'train':
                self.train_data = self.train_data.map(self.custom_preprocessing)
            elif partition in ['test', 'validation']:
                self.test_data = self.test_data.map(self.custom_preprocessing)
            else:
                raise ValueError("Please provide a valid partition split: train or test")
        except Exception as e:
            print(e)
            raise ValueError("Error when pre-processing {} {} data.".format(self.name, partition))

    def custom_preprocessing(self):
            """
            Wraps a pre-processing function (dict -> dict) specific to the benchmark.
            Needs to be overriden in the extended class.

            The return dictionary must contains keys 'prompt' & 'answer' for inference to work.
            """
            raise NotImplementedError('Implement custom_preprocessing() in a child class.')

    def add_instruction(self, instruction=None, partition='train'):
        """
        Adds instructions to the data based on the input partition.

        :param instruction: dict, with the `system` and `user` instructions. If None, then it creates prompt with few shot
        :param partition: str, the split of the data: train / test
        """
        def _add_instruction(row):
            row['prompt'] = '{}\n{}\n{}\n'.format(
                instruction['system'],
                row['prompt'],
                instruction['user'])
            return row

        if partition == 'train':
            self.train_data = self.train_data.map(_add_instruction)
        elif partition == 'test' or partition == 'validation':
            self.test_data = self.test_data.map( _add_instruction)
        else:
            raise ValueError("Please provide a valid partition split: {}".format(self.splits))


    def add_generations(self, data):
        """
        Adds the generations to the respective class attribute as a HuggingFace Dataset.

        :param data: pd.DataFrame or HuggingFace Dataset
        """
        if isinstance(data, pd.DataFrame):
            self.generations = Dataset.from_pandas(data)
        elif isinstance(data, Dataset):
            self.generations = data

    
    def save_generations(self, dataset_name, run_name):
        """
        Saves the generations in the respective directory.
        """
        path = os.path.join(ROOT_DIR, 'benchmarks', 'generations')
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        gen_path = os.path.join(path, f"{dataset_name}-{run_name}.jsonl")

        self.generations.to_json(gen_path, orient="records")
        print("Stored {} generations to the following path: {}".format(self.name, gen_path))


    def load_generations(self, benchmark_name):
        """
        Loads the generations from the respective directory.
        """
        path = os.path.join(ROOT_DIR, 'benchmarks', 'generations', f"{self.name}_{benchmark_name}.json")
        if not os.path.exists(path):
            raise ValueError("No generations found for {} at path: {}. \
                             Please run inference first.".format(self.name, path))
        print("Loading {} generations from the following path: {}".format(self.name, path))
        self.generations = pd.read_json(path)


class MedMCQA(Benchmark):
    '''
    MedMCQA is a large-scale, Multiple-Choice Question Answering (MCQA) dataset
    designed to address real-world medical entrance exam questions.

    Huggingface card: https://huggingface.co/datasets/medmcqa
    '''
    def __init__(self, name='medmcqa') -> None:
        super().__init__(name)
        self.hub_name = 'medmcqa'
        self.dir_name = 'medmcqa'
        self.path = os.path.join(ROOT_DIR, 'benchmarks', 'datasets', self.dir_name)
        self.splits = ['train', 'validation', 'test']
        self.num_options = 4

    @staticmethod
    def custom_preprocessing(row):
        options = [row['opa'], row['opb'], row['opc'], row['opd']]
        answer = int(row['cop'])
        row['prompt'] = format_mcq(row['question'], options)
        row['gold'] = chr(ord('A')+answer) if answer in [0, 1, 2, 3] else None
        return row


class PubMedQA(Benchmark):
    '''
    PubMedQA is a novel biomedical question answering (QA) dataset.
    Its task is to answer research biomedical questions with yes/no/maybe using PubMed abstracts.

    Huggingface card: https://huggingface.co/datasets/bigbio/pubmed_qa
    '''
    def __init__(self, name='pubmedqa') -> None:
        super().__init__(name)
        self.hub_name = "bigbio/pubmed_qa"
        self.dir_name = 'bigbio___pubmed_qa'
        self.path = os.path.join(ROOT_DIR, 'benchmarks', 'datasets', self.dir_name)
        self.splits = ['train', 'validation', 'test']
        self.subsets = ['pubmed_qa_labeled_fold0_source']
        self.num_options = 3

    @staticmethod
    def custom_preprocessing(row):

        row["prompt"] = row['QUESTION'] #  f"{row['CONTEXTS'][0]}\n{row['QUESTION']}"
        row["gold"] = row['final_decision']
        row["long_answer"] = row["LONG_ANSWER"]
        return row

class ClosedPubMedQA(Benchmark):
    '''
    PubMedQA is a novel biomedical question answering (QA) dataset.
    Its task is to answer research biomedical questions with yes/no/maybe using PubMed abstracts.

    Huggingface card: https://huggingface.co/datasets/bigbio/pubmed_qa
    '''
    def __init__(self, name='pubmedqa') -> None:
        super().__init__(name)
        self.hub_name = "bigbio/pubmed_qa"
        self.dir_name = 'bigbio___pubmed_qa'
        self.path = os.path.join(ROOT_DIR, 'benchmarks', 'datasets', self.dir_name)
        self.splits = ['train', 'validation', 'test']
        self.subsets = ['pubmed_qa_labeled_fold0_source']
        self.num_options = 3

    @staticmethod
    def custom_preprocessing(row):
        context = '\n'.join(row['CONTEXTS'])
        row["prompt"] = f"{context}\n{row['QUESTION']}"
        row["gold"] = row['final_decision']
        row["long_answer"] = row["LONG_ANSWER"]
        return row

class PubMedQAValidation(Benchmark):
    '''
    PubMedQA is a novel biomedical question answering (QA) dataset.
    Its task is to answer research biomedical questions with yes/no/maybe using PubMed abstracts.

    Huggingface card: https://huggingface.co/datasets/bigbio/pubmed_qa
    '''
    def __init__(self, name='pubmedqa') -> None:
        super().__init__(name)
        self.hub_name = "bigbio/pubmed_qa"
        self.dir_name = 'bigbio___pubmed_qa'
        self.path = os.path.join(ROOT_DIR, 'benchmarks', 'datasets', self.dir_name)
        self.splits = ['validation']
        # self.subsets = ['pubmed_qa_labeled_fold1_bigbio_qa'] + ['pubmed_qa_artificial_source']
        self.num_options = 3
        self.local_path = os.path.join(ROOT_DIR, 'benchmarks', 'datasets', 'pubmedqa_pubmedqa_validation.jsonl')

    @staticmethod
    def custom_preprocessing(row):
        row["prompt"] = row['QUESTION']
        row["gold"] = row['final_decision']
        return row


class MedQA(Benchmark):
    '''
    MedQA is a dataset for solving medical problems collected from the professional medical board exams.

    Huggingface card: https://huggingface.co/datasets/bigbio/med_qa
    '''
    def __init__(self, name='medqa') -> None:
        super().__init__(name)
        self.hub_name = 'bigbio/med_qa'
        self.dir_name = 'bigbio___med_qa'
        self.path = os.path.join(ROOT_DIR, 'benchmarks', 'datasets', self.dir_name)
        self.splits = ['train', 'validation', 'test']
        self.num_options = 5
        self.subsets = ['med_qa_en_4options_source']

    @staticmethod
    def custom_preprocessing(row):
        choices = [opt['value'] for opt in row['options']]
        row["prompt"] = format_mcq(row['question'], choices)
        for opt in row['options']:
            if opt['value'] == row['answer']:
                row['gold'] = opt['key']
                break
        return row

class MedQA4(Benchmark):
    '''
    MedQA is a dataset for solving medical problems collected from the professional medical board exams.

    Huggingface card: https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options
    '''
    def __init__(self, name='medqa4') -> None:
        super().__init__(name)
        self.hub_name = 'GBaker/MedQA-USMLE-4-options'
        self.dir_name = 'GBaker___med_qa-usmle-4-options'
        self.path = os.path.join(ROOT_DIR, 'benchmarks', 'datasets', self.dir_name)
        self.splits = ['train', 'test']
        self.num_options = 4

    @staticmethod
    def custom_preprocessing(row):
        choices = [row['options'][opt] for opt in row['options']]
        row["prompt"] = format_mcq(row['question'], choices)
        for opt in row['options']:
            if row['options'][opt] == row['answer']:
                row['gold'] = opt
                break
        return row


class MedicationQA(Benchmark):
    '''
    MedicationQA is a benchmark to measure whether a language model is truthful in generating answers to questions
    Huggingface card: https://huggingface.co/datasets/truthful_qa
    '''
    def __init__(self, name='medicationqa') -> None:
        super().__init__(name)
        self.hub_name = 'truehealth/medicationqa'
        self.dir_name = 'truehealth___parquet'
        self.path = os.path.join(ROOT_DIR, 'benchmarks', 'datasets', self.dir_name)
        self.splits = ['train']

    @staticmethod
    def custom_preprocessing(row):
        """
        Wraps a pre-processing function (dict -> dict) specific to the benchmark.
        Probably will need to be overriden in the extended class.
        """
        row["prompt"] = row['Question'],
        row["gold"] = row['Answer']
        return row


class TruthfulQA(Benchmark):
    '''
    TruthfulQA is a dataset of consumer health questions about medications.
    Huggingface card: https://huggingface.co/datasets/truehealth/medicationqa
    '''
    def __init__(self, name='truthfulqa') -> None:
        super().__init__(name)
        self.hub_name = 'truthful_qa'
        self.dir_name = 'truthful_qa'
        self.path = os.path.join(ROOT_DIR, 'benchmarks', 'datasets', self.dir_name)
        self.splits = ['validation']
        self.subsets = ['multiple_choice']
        self.num_options = 4

    @staticmethod
    def custom_preprocessing(row):
        options = row['mc1_targets']['choices']
        labels = row['mc1_targets']['labels']
        gold_option = options[labels.index(1)]
        options.remove(gold_option)
        wrong_options = random.choices(options, k=3)
        choices = [gold_option] + wrong_options
        random.shuffle(choices)
        gold_id = choices.index(gold_option)
        row["prompt"] = format_mcq(row['question'], choices)
        row["gold"] = ['A', 'B', 'C', 'D'][gold_id]
        return row

class MMLU(Benchmark):
    '''
    Measuring Massive Multitask Language Understanding
    Huggingface card: https://huggingface.co/datasets/lukaemon/mmlu
    '''
    def __init__(self, name) -> None:
        super().__init__(name)
        self.hub_name = 'lukaemon/mmlu'
        self.dir_name = 'lukaemon___mmlu'
        self.path = os.path.join(ROOT_DIR, 'benchmarks', 'datasets', self.dir_name)
        self.splits = ['train', 'validation', 'test']
        self.subsets_general = ['college_computer_science', 'college_mathematics', 'elementary_mathematics',
                        'high_school_computer_science', 'high_school_mathematics',
                        'high_school_statistics','machine_learning', 'global_facts']
        self.subsets_medical = [
            'anatomy',
            'college_biology',
            'college_medicine',
            'professional_medicine',
            'medical_genetics',
            'virology',
            'clinical_knowledge',
            'high_school_biology',
            'high_school_chemistry',
            'nutrition',
            'college_chemistry'
        ]
        self.subsets = self.subsets_medical
        if name == 'mmlu_general':
            self.subsets = self.subsets_general
        self.num_options = 4

    @staticmethod
    def custom_preprocessing(row):
        options = [row['A'], row['B'], row['C'], row['D']]
        row["prompt"] = format_mcq(row['input'], options)
        row["gold"] = row['target']
        row["subset"] = row["subset"]
        return row


class GSM8K(Benchmark):
    '''
    GSM8K (Grade School Math 8K) is a dataset of 8.5K grade school math word problems.
    Huggingface card: https://huggingface.co/datasets/gsm8k
    '''
    def __init__(self, name='gsm8k') -> None:
        super().__init__(name)
        self.hub_name = 'gsm8k'
        self.dir_name = 'gsm8k'
        self.path = os.path.join(ROOT_DIR, 'benchmarks', 'datasets', self.dir_name)
        self.splits = ['train', 'test']
        self.subsets = ['main']

    @staticmethod
    def custom_preprocessing(row):
        row["prompt"] = row['question']
        row["gold"] = row['answer'].split('####')[1].strip()
        row["steps"] = row['answer'].split('####')[0].strip()
        return row

class Blurb(Benchmark):
    '''
    BLURB is a collection of resources for biomedical natural language processing.
    Huggingface card: https://huggingface.co/datasets/bigbio/blurb
    '''
    def __init__(self, name="blurb") -> None:
        super().__init__(name)
        self.hub_name = "bigbio/blurb"
        self.dir_name = "bigbio___blurb"
        self.path = os.path.join(ROOT_DIR, 'benchmarks', 'datasets', self.dir_name)
        self.splits = ["train", "validation", "test"]
        self.subsets = ["bc2gm", "bc5chem", "bc5disease", "jnlpba", "ncbi_disease"]

    @staticmethod
    def custom_preprocessing(row):
        tokens = row["tokens"]
        tags = row["ner_tags"]
        entity_type = row["type"]

        instruction = f"Given the following sentence, tell me which part of this sentence is a {entity_type} expression. There may be multiple expressions in this sentence."
        prompt = f"{instruction}\n\n Sentence: {' '.join(tokens)}"
        row["prompt"] = prompt
        row["gold"] = "#".join(Blurb.get_entities(tokens, tags))
        return row

    @staticmethod
    def get_entities(tokens, tags):
        entities = []
        entity = []
        for token, tag in zip(tokens, tags):
            if tag == 1:
                if entity:
                    entities.append(' '.join(entity))
                entity = [token]
            elif tag == 2:
                entity.append(token)
            elif tag == 0 and entity:
                entities.append(' '.join(entity))
                entity = []
        if entity:
            entities.append(' '.join(entity))
        return entities



def format_mcq(question, options):
    """
    Formats a multiple choice question with the given options.
    Uses the format recommended by: https://huggingface.co/blog/evaluating-mmlu-leaderboard

    'Question: What is the capital of France?

    Options:
    A. London
    B. Paris
    C. Berlin
    D. Rome'

    :param question: str, the question
    :param options: list of str, the options
    :return: str, the formatted question
    """
    if not question.endswith('?') and not question.endswith('.'):
        question += '?'
    options_str = '\n'.join([f"{chr(65+i)}. {options[i]}" for i in range(len(options))])
    prompt = 'Question: ' + question + '\n\nOptions:\n' + options_str
    return prompt


def aggregate_datasets(path, subsets, partition='train'):
    """
    Takes as input a Huggingface DatasetDict with subset name as key, and Dataset as value.
    Returns a pd.DataFrame with all subsets concatenated.

    :param subsets: list of str, the subsets of the data to download from the HuggingFace hub.
    :return: pd.DataFrame
    """
    dataframes = []
    for subset in subsets:
        subset_data = load_dataset(os.path.join(path, subset), split=partition)
        subset_df = pd.DataFrame(subset_data.map(lambda x: {'subset': subset, **x}))
        dataframes.append(subset_df)
    aggregate_df  = pd.concat(dataframes, axis=0)
    aggregate = Dataset.from_pandas(aggregate_df)
    if '__index_level_0__' in aggregate.column_names:
        aggregate = aggregate.remove_columns('__index_level_0__')
    return aggregate
