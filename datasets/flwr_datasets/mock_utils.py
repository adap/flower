import random
import string
from datetime import datetime, timedelta
from typing import List, Any, Union

import datasets
from datasets import Features, Value


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


if __name__ == "__main__":
    # def mock_sentiment140(num_rows):
    num_rows = 100
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
    print(dataset)
    print(dataset[0])
    print(dataset.features)

    dataset_dict = datasets.DatasetDict({"train": dataset})
    print(dataset_dict)
