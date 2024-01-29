import random
import string
from datetime import datetime, timedelta
from typing import List, Any, Union, Tuple, Callable, Optional

import datasets
import numpy as np
from PIL import Image
from datasets import Features, Value, ClassLabel, DatasetDict
import io


def create_artificial_strings(num_rows: int, num_unique: int, string_length: int) -> \
        List[str]:
    """Create list of strings."""
    unique_strings = set()
    while len(unique_strings) < num_unique:
        random_str = ''.join(
            random.choices(string.ascii_letters + string.digits, k=string_length))
        unique_strings.add(random_str)

    unique_strings_list = list(unique_strings)
    artificial_column = unique_strings_list.copy()
    remaining_to_allocate = num_rows - num_unique
    for _ in range(remaining_to_allocate):
        artificial_column.append(random.choice(unique_strings_list))
    return artificial_column


def create_artificial_categories(num_rows: int, choices: List[Any]) -> List[str]:
    """Create list of strings from given `choices` list."""
    artificial_column = choices.copy()
    remaining_to_allocate = num_rows - len(choices)
    for _ in range(remaining_to_allocate):
        artificial_column.append(random.choice(choices))
    return artificial_column


def generate_random_word(length: int) -> str:
    # Generate a random word of the given length
    return ''.join(random.choices(string.ascii_letters, k=length))


def generate_random_sentence(min_word_length: int, max_word_length: int,
                             min_sentence_length: int, max_sentence_length: int) -> str:
    # Generate a random sentence with words of random lengths
    sentence_length = random.randint(min_sentence_length, max_sentence_length)
    sentence = []
    while len(' '.join(sentence)) < sentence_length:
        word_length = random.randint(min_word_length, max_word_length)
        word = generate_random_word(word_length)
        sentence.append(word)
    return ' '.join(sentence)


def generate_random_text_column(num_rows: int, min_word_length: int,
                                max_word_length: int,
                                min_sentence_length: int, max_sentence_length: int) -> \
        List[str]:
    # Generate a list of random sentences
    return [
        generate_random_sentence(min_word_length, max_word_length, min_sentence_length,
                                 max_sentence_length)
        for _ in range(num_rows)]


def generate_random_date(start_date: datetime, end_date: datetime,
                         date_format: str = "%a %b %d %H:%M:%S %Y",
                         as_string: bool = True) -> Union[str, datetime]:
    # Generate a random date between start_date and end_date
    time_between_dates = end_date - start_date
    random_seconds = random.randint(0, int(time_between_dates.total_seconds()))
    random_date = start_date + timedelta(seconds=random_seconds)

    # Return the date in the specified format
    if as_string:
        return random_date.strftime(date_format)
    else:
        return random_date


def generate_random_date_column(num_rows: int, start_date: datetime, end_date: datetime,
                                date_format: str = "%a %b %d %H:%M:%S %Y",
                                as_string: bool = True) -> List[Union[str, datetime]]:
    # Generate a list of random dates
    return [generate_random_date(start_date, end_date, date_format, as_string)
            for _ in range(num_rows)]


def generate_random_image_column(num_rows: int, image_size: Union[
    Tuple[int, int], Tuple[int, int, int]], simulate_type: str):
    """Simulate the images with the format that is found in HF Hub.

    Directly using `Image.fromarray` does not work because it creates `PIL.Image.Image`.
    """
    # Generate numpy images
    np_images = []
    for _ in range(num_rows):
        np_images.append(np.random.randint(0, 255, size=image_size, dtype=np.uint8))
    # Change the format to the PIL.PngImagePlugin.PngImageFile
    # or the PIL.JpegImagePlugin.JpegImageFile format
    pil_imgs = []
    for np_image in np_images:
        # Convert the NumPy array to a PIL image
        pil_img_beg = Image.fromarray(np_image)

        # Save the image to an in-memory bytes buffer
        in_memory_file = io.BytesIO()
        pil_img_beg.save(in_memory_file, format=simulate_type)
        in_memory_file.seek(0)

        # Reload the image as a PngImageFile
        pil_image_end = Image.open(in_memory_file)
        pil_imgs.append(pil_image_end)
    return pil_imgs


def mock_sentiment140(num_rows: int):
    users = create_artificial_strings(num_rows=num_rows, num_unique=30, string_length=5)
    sentiment = create_artificial_categories(num_rows=num_rows, choices=[0, 4])
    query = ["NO_QUERY"] * num_rows

    # Sentences
    min_word_length = 3
    max_word_length = 8
    min_sentence_length = 20
    max_sentence_length = 60

    text = generate_random_text_column(num_rows, min_word_length,
                                       max_word_length, min_sentence_length,
                                       max_sentence_length)

    start_date = datetime(2009, 1, 1)
    end_date = datetime(2010, 12, 31)
    date_format = "%a %b %d %H:%M:%S %Y"

    # Generate a list of random dates as strings
    date = generate_random_date_column(num_rows, start_date, end_date,
                                       date_format, as_string=True)

    features = Features(
        {'text': Value(dtype='string'),
         'date': Value(dtype='string'),
         'user': Value(dtype='string'),
         'sentiment': Value(dtype='int32'),
         'query': Value(dtype='string')})
    dataset = datasets.Dataset.from_dict(
        {"user": users, "sentiment": sentiment, "query": query, "text": text,
         "date": date}, features=features)
    return dataset


def mock_cifar100(num_rows: int):
    imgs = generate_random_image_column(num_rows, (32, 32, 3), "PNG")
    unique_fine_labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed',
                          'bee',
                          'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus',
                          'butterfly', 'camel', 'can', 'castle', 'caterpillar',
                          'cattle',
                          'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch',
                          'cra', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant',
                          'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
                          'kangaroo',
                          'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
                          'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
                          'mouse',
                          'mushroom', 'oak_tree', 'orange', 'orchid', 'otter',
                          'palm_tree',
                          'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate',
                          'poppy',
                          'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road',
                          'rocket',
                          'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
                          'skyscraper',
                          'snail', 'snake', 'spider', 'squirrel', 'streetcar',
                          'sunflower',
                          'sweet_pepper', 'table', 'tank', 'telephone', 'television',
                          'tiger',
                          'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe',
                          'whale',
                          'willow_tree', 'wolf', 'woman', 'worm']
    fine_label = create_artificial_categories(num_rows, unique_fine_labels)
    unique_coarse_labels = ['aquatic_mammals', 'fish', 'flowers', 'food_containers',
                            'fruit_and_vegetables', 'household_electrical_devices',
                            'household_furniture', 'insects', 'large_carnivores',
                            'large_man-made_outdoor_things',
                            'large_natural_outdoor_scenes',
                            'large_omnivores_and_herbivores', 'medium_mammals',
                            'non-insect_invertebrates', 'people', 'reptiles',
                            'small_mammals',
                            'trees', 'vehicles_1', 'vehicles_2']

    coarse_label = create_artificial_categories(num_rows, unique_coarse_labels)
    features = Features(
        {'img': datasets.Image(decode=True),
         'fine_label': ClassLabel(
             names=unique_fine_labels),
         'coarse_label': ClassLabel(
             names=unique_coarse_labels)}
    )
    dataset = datasets.Dataset.from_dict(
        {"img": imgs, "coarse_label": coarse_label, "fine_label": fine_label},
        features=features)
    return dataset


def mock_svhn_cropped_digits(num_rows: int):
    imgs = generate_random_image_column(num_rows, (32, 32, 3), "PNG")
    unique_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    label = create_artificial_categories(num_rows, unique_labels)
    features = Features(
        {'image': datasets.Image(decode=True),
         'label': ClassLabel(
             names=unique_labels), }
    )
    dataset = datasets.Dataset.from_dict(
        {"image": imgs, "label": label},
        features=features)
    return dataset


def mock_dict_dataset(num_rows: List[int], split_names: List[str], function: Callable):
    dataset_dict = {}
    for params in zip(num_rows, split_names):
        dataset_dict[params[1]] = function(params[0])
    return datasets.DatasetDict(dataset_dict)


dataset_name_to_mock_function = {
    "cifar100": mock_cifar100,
    "sentiment140": mock_sentiment140,
    "svhn_cropped_digits": mock_svhn_cropped_digits
}


def load_mocked_dataset(dataset_name: str, num_rows: List[int], split_names: List[str],
                        subset: Optional[str] = "", ) -> DatasetDict:
    dataset_dict = {}
    name = dataset_name if subset is "" else dataset_name + "_" + subset
    dataset_creation_fnc = dataset_name_to_mock_function[name]
    for params in zip(num_rows, split_names):
        dataset_dict[params[1]] = dataset_creation_fnc(params[0])
    return datasets.DatasetDict(dataset_dict)


if __name__ == "__main__":
    print("mock_cifar100")
    print(mock_dict_dataset([200, 100], ["train", "test"], mock_cifar100))
    print("mock_sentiment140")
    print(mock_dict_dataset([200, ], ["train", ], mock_sentiment140))
    print("mock_svhn_cropped_digits")
    print(mock_dict_dataset([200, 100, 50], ["train", "test", "extra"],
                            mock_svhn_cropped_digits))
