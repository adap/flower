import os
import numpy as np
import torch
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load
from .utils import extract_file, make_classes_counts


class SpeechCommandsV1(Dataset):
    data_name = 'SpeechCommandsV1'
    file = [('https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz',
             '743935421bb51cccdb6bdd152e04c5c70274e935c82119ad7faeec31780d811d')]
    sr = 16000

    def __init__(self, root, split, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        if not check_exists(self.processed_folder):
            self.process()
        self.id, self.data, self.speaker_id, self.utterance_id, self.target = load(
            os.path.join(self.processed_folder, '{}.pt'.format(self.split)), mode='pickle')
        self.classes_counts = make_classes_counts(self.target)
        self.background_noise, self.classes_to_labels, self.target_size = load(
            os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')
        self.background_noise = [torchaudio.load(self.background_noise[i])[0] for i in
                                 range(len(self.background_noise))]

    def __getitem__(self, index):
        id, data, speaker_id, utterance_id, target = torch.tensor(self.id[index]), \
                                                     torchaudio.load(self.data[index])[0], \
                                                     torch.tensor(self.speaker_id[index]), \
                                                     torch.tensor(self.utterance_id[index]), \
                                                     torch.tensor(self.target[index])
        input = {'id': id, 'data': data, 'speaker_id': speaker_id, 'utterance_id': utterance_id, 'target': target}
        with torch.no_grad():
            if self.transform is not None:
                input = self.transform(input)
        return input

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        train_set, valid_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'), mode='pickle')
        save(valid_set, os.path.join(self.processed_folder, 'valid.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'test.pt'), mode='pickle')
        save(meta, os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, hash_prefix) in self.file:
            filename = os.path.basename(url)
            torch.hub.download_url_to_file(url, os.path.join(self.raw_folder, filename), hash_prefix)
            extract_file(os.path.join(self.raw_folder, filename))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.transform.__repr__())
        return fmt_str

    def make_data(self):
        train_raw_data = sorted(str(p) for p in Path(self.raw_folder).glob("*/*.wav"))
        train_raw_data = [f for f in train_raw_data if '_nohash_' in f and '_background_noise_' not in f]
        valid_raw_data = _load_list(self.raw_folder, 'validation_list.txt')
        test_raw_data = _load_list(self.raw_folder, 'testing_list.txt')
        background_noise = sorted(
            str(p) for p in Path(os.path.join(self.raw_folder, '_background_noise_')).glob("*.wav"))
        target_dict = {"yes": 0, "no": 1, "up": 2, "down": 3, "left": 4, "right": 5, "on": 6, "off": 7, "stop": 8,
                       "go": 9, 'silence': 10, 'unknown': 11}
        train_data, train_speaker_id, train_utterance, train_target = filter_target(train_raw_data, target_dict,
                                                                                    background_noise)
        valid_data, valid_speaker_id, valid_utterance, valid_target = filter_target(valid_raw_data, target_dict,
                                                                                    background_noise)
        test_data, test_speaker_id, test_utterance, test_target = filter_target(test_raw_data, target_dict,
                                                                                background_noise)
        speaker_id = train_speaker_id + valid_speaker_id + test_speaker_id
        speaker_id = sorted(list(set(speaker_id)))
        speaker_id_dict = {speaker_id[i]: i for i in range(len(speaker_id))}
        train_speaker_id = [speaker_id_dict[x] for x in train_speaker_id]
        valid_speaker_id = [speaker_id_dict[x] for x in valid_speaker_id]
        test_speaker_id = [speaker_id_dict[x] for x in test_speaker_id]
        train_id, valid_id, test_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(valid_data)).astype(
            np.int64), np.arange(len(test_data)).astype(np.int64)
        classes_to_labels = target_dict
        target_size = len(target_dict.keys())
        return (train_id, train_data, train_speaker_id, train_utterance, train_target), (
            valid_id, valid_data, valid_speaker_id, valid_utterance, valid_target), (
                   test_id, test_data, test_speaker_id, test_utterance, test_target), (
                   background_noise, classes_to_labels, target_size)


class SpeechCommandsV2(Dataset):
    data_name = 'SpeechCommandsV2'
    file = [('https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
             'af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58')]
    sr = 16000

    def __init__(self, root, split, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        if not check_exists(self.processed_folder):
            self.process()
        self.id, self.data, self.speaker_id, self.utterance_id, self.target = load(
            os.path.join(self.processed_folder, '{}.pt'.format(self.split)), mode='pickle')
        self.classes_counts = make_classes_counts(self.target)
        self.background_noise, self.classes_to_labels, self.target_size = load(
            os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')
        self.background_noise = [torchaudio.load(self.background_noise[i])[0] for i in
                                 range(len(self.background_noise))]

    def __getitem__(self, index):
        id, data, speaker_id, utterance_id, target = torch.tensor(self.id[index]), \
                                                     torchaudio.load(self.data[index])[0], \
                                                     torch.tensor(self.speaker_id[index]), \
                                                     torch.tensor(self.utterance_id[index]), \
                                                     torch.tensor(self.target[index])
        input = {'id': id, 'data': data, 'speaker_id': speaker_id, 'utterance_id': utterance_id, 'target': target}
        if self.transform is not None:
            input = self.transform(input)
        return input

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        train_set, valid_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'), mode='pickle')
        save(valid_set, os.path.join(self.processed_folder, 'valid.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'test.pt'), mode='pickle')
        save(meta, os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, hash_prefix) in self.file:
            filename = os.path.basename(url)
            torch.hub.download_url_to_file(url, os.path.join(self.raw_folder, filename), hash_prefix)
            extract_file(os.path.join(self.raw_folder, filename))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.transform.__repr__())
        return fmt_str

    def make_data(self):
        train_raw_data = sorted(str(p) for p in Path(self.raw_folder).glob("*/*.wav"))
        train_raw_data = [f for f in train_raw_data if '_nohash_' in f and '_background_noise_' not in f]
        valid_raw_data = _load_list(self.raw_folder, 'validation_list.txt')
        test_raw_data = _load_list(self.raw_folder, 'testing_list.txt')
        background_noise = sorted(
            str(p) for p in Path(os.path.join(self.raw_folder, '_background_noise_')).glob("*.wav"))
        target_dict = {"yes": 0, "no": 1, "up": 2, "down": 3, "left": 4, "right": 5, "on": 6, "off": 7, "stop": 8,
                       "go": 9, 'silence': 10, 'unknown': 11}
        train_data, train_speaker_id, train_utterance, train_target = filter_target(train_raw_data, target_dict,
                                                                                    background_noise)
        valid_data, valid_speaker_id, valid_utterance, valid_target = filter_target(valid_raw_data, target_dict,
                                                                                    background_noise)
        test_data, test_speaker_id, test_utterance, test_target = filter_target(test_raw_data, target_dict,
                                                                                background_noise)
        speaker_id = train_speaker_id + valid_speaker_id + test_speaker_id
        speaker_id = sorted(list(set(speaker_id)))
        speaker_id_dict = {speaker_id[i]: i for i in range(len(speaker_id))}
        train_speaker_id = [speaker_id_dict[x] for x in train_speaker_id]
        valid_speaker_id = [speaker_id_dict[x] for x in valid_speaker_id]
        test_speaker_id = [speaker_id_dict[x] for x in test_speaker_id]
        train_id, valid_id, test_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(valid_data)).astype(
            np.int64), np.arange(len(test_data)).astype(np.int64)
        classes_to_labels = target_dict
        target_size = len(target_dict.keys())
        return (train_id, train_data, train_speaker_id, train_utterance, train_target), (
            valid_id, valid_data, valid_speaker_id, valid_utterance, valid_target), (
                   test_id, test_data, test_speaker_id, test_utterance, test_target), (
                   background_noise, classes_to_labels, target_size)


def _load_list(root, *filenames):
    output = []
    for filename in filenames:
        filepath = os.path.join(root, filename)
        with open(filepath) as fileobj:
            output += [os.path.normpath(os.path.join(root, line.strip())) for line in fileobj]
    return output


def load_speechcommands_item(path):
    filename = os.path.splitext(os.path.basename(path))[0]
    speaker_id, _, utterance = filename.split('_')
    utterance = int(utterance)
    target = os.path.basename(os.path.dirname(path))
    return speaker_id, utterance, target


def filter_target(raw_data, target_dict, background_noise):
    data, speaker_id, utterance, target = [], [], [], []
    unknown_data, unknown_speaker_id, unknown_utterance, unknown_target = [], [], [], []
    for path in raw_data:
        train_speaker_id_i, train_utterance_i, train_target_i = load_speechcommands_item(path)
        if train_target_i in target_dict:
            data.append(path)
            speaker_id.append(train_speaker_id_i)
            utterance.append(train_utterance_i)
            target.append(target_dict[train_target_i])
        else:
            unknown_data.append(path)
            unknown_speaker_id.append(train_speaker_id_i)
            unknown_utterance.append(train_utterance_i)
            unknown_target.append(target_dict['unknown'])
    data_size = len(data)
    silence_idx = np.random.choice(len(background_noise), int(0.1 * data_size)).tolist()
    data = data + [background_noise[i] for i in silence_idx]
    speaker_id = speaker_id + [os.path.splitext(os.path.basename(background_noise[i]))[0] for i in silence_idx]
    utterance = utterance + [0 for _ in range(len(silence_idx))]
    target = target + [target_dict['silence'] for _ in range(len(silence_idx))]
    unknown_idx = np.random.choice(len(unknown_data), int(0.1 * data_size)).tolist()
    data = data + [unknown_data[i] for i in unknown_idx]
    speaker_id = speaker_id + [unknown_speaker_id[i] for i in unknown_idx]
    utterance = utterance + [unknown_utterance[i] for i in unknown_idx]
    target = target + [unknown_target[i] for i in unknown_idx]
    return data, speaker_id, utterance, target
