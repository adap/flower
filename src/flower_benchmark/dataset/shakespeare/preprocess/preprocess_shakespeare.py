"""Preprocesses the Shakespeare dataset for federated training.
Copyright 2017 Google Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
To run:
  python preprocess_shakespeare.py path/to/raw/shakespeare.txt output_directory/
The raw data can be downloaded from:
  http://www.gutenberg.org/cache/epub/100/pg100.txt
(The Plain Text UTF-8 file format, md5sum: 036d0f9cf7296f41165c2e6da1e52a0e)
Note that The Comedy of Errors has a incorrect indentation compared to all the
other plays in the file. The code below reflects that issue. To make the code
cleaner, you could fix the indentation in the raw shakespeare file and remove
the special casing for that play in the code below.
Authors: loeki@google.com, mcmahan@google.com
Disclaimer: This is not an official Google product.
"""
import collections
import json
import os
import random
import re
import sys
RANDOM_SEED = 1234
# Regular expression to capture an actors name, and line continuation
CHARACTER_RE = re.compile(r'^  ([a-zA-Z][a-zA-Z ]*)\. (.*)')
CONT_RE = re.compile(r'^    (.*)')
# The Comedy of Errors has errors in its indentation so we need to use
# different regular expressions.
COE_CHARACTER_RE = re.compile(r'^([a-zA-Z][a-zA-Z ]*)\. (.*)')
COE_CONT_RE = re.compile(r'^(.*)')

def _match_character_regex(line, comedy_of_errors=False):
    return (COE_CHARACTER_RE.match(line) if comedy_of_errors
            else CHARACTER_RE.match(line))

def _match_continuation_regex(line, comedy_of_errors=False):
    return (
        COE_CONT_RE.match(line) if comedy_of_errors else CONT_RE.match(line))

def _split_into_plays(shakespeare_full):
    """Splits the full data by play."""
    # List of tuples (play_name, dict from character to list of lines)
    plays = []
    discarded_lines = []  # Track discarded lines.
    slines = shakespeare_full.splitlines(True)[1:]

    # skip contents, the sonnets, and all's well that ends well
    author_count = 0
    start_i = 0
    for i, l in enumerate(slines):
        if 'by William Shakespeare' in l:
            author_count += 1
        if author_count == 2:
            start_i = i - 5
            break
    slines = slines[start_i:]

    current_character = None
    comedy_of_errors = False
    for i, line in enumerate(slines):
        # This marks the end of the plays in the file.
        if i > 124195 - start_i:
            break
        # This is a pretty good heuristic for detecting the start of a new play:
        if 'by William Shakespeare' in line:
            current_character = None
            characters = collections.defaultdict(list)
            # The title will be 2, 3, 4, 5, 6, or 7 lines above "by William Shakespeare".
            if slines[i - 2].strip():
                title = slines[i - 2]
            elif slines[i - 3].strip():
                title = slines[i - 3]
            elif slines[i - 4].strip():
                title = slines[i - 4]
            elif slines[i - 5].strip():
                title = slines[i - 5]
            elif slines[i - 6].strip():
                title = slines[i - 6]
            else:
                title = slines[i - 7]
            title = title.strip()

            assert title, (
                'Parsing error on line %d. Expecting title 2 or 3 lines above.' %
                i)
            comedy_of_errors = (title == 'THE COMEDY OF ERRORS')
            # Degenerate plays are removed at the end of the method.
            plays.append((title, characters))
            continue
        match = _match_character_regex(line, comedy_of_errors)
        if match:
            character, snippet = match.group(1), match.group(2)
            # Some character names are written with multiple casings, e.g., SIR_Toby
            # and SIR_TOBY. To normalize the character names, we uppercase each name.
            # Note that this was not done in the original preprocessing and is a
            # recent fix.
            character = character.upper()
            if not (comedy_of_errors and character.startswith('ACT ')):
                characters[character].append(snippet)
                current_character = character
                continue
            else:
                current_character = None
                continue
        elif current_character:
            match = _match_continuation_regex(line, comedy_of_errors)
            if match:
                if comedy_of_errors and match.group(1).startswith('<'):
                    current_character = None
                    continue
                else:
                    characters[current_character].append(match.group(1))
                    continue
        # Didn't consume the line.
        line = line.strip()
        if line and i > 2646:
            # Before 2646 are the sonnets, which we expect to discard.
            discarded_lines.append('%d:%s' % (i, line))
    # Remove degenerate "plays".
    return [play for play in plays if len(play[1]) > 1], discarded_lines

def _remove_nonalphanumerics(filename):
    return re.sub('\\W+', '_', filename)

def play_and_character(play, character):
    return _remove_nonalphanumerics((play + '_' + character).replace(' ', '_'))

def _get_train_test_by_character(plays, test_fraction=0.2):
    """
      Splits character data into train and test sets.
      if test_fraction <= 0, returns {} for all_test_examples
      plays := list of (play, dict) tuples where play is a string and dict
      is a dictionary with character names as keys
    """
    skipped_characters = 0
    all_train_examples = collections.defaultdict(list)
    all_test_examples = collections.defaultdict(list)

    def add_examples(example_dict, example_tuple_list):
        for play, character, sound_bite in example_tuple_list:
            example_dict[play_and_character(
                play, character)].append(sound_bite)

    users_and_plays = {}
    for play, characters in plays:
        curr_characters = list(characters.keys())
        for c in curr_characters:
            users_and_plays[play_and_character(play, c)] = play
        for character, sound_bites in characters.items():
            examples = [(play, character, sound_bite)
                        for sound_bite in sound_bites]
            if len(examples) <= 2:
                skipped_characters += 1
                # Skip characters with fewer than 2 lines since we need at least one
                # train and one test line.
                continue
            train_examples = examples
            if test_fraction > 0:
                num_test = max(int(len(examples) * test_fraction), 1)
                train_examples = examples[:-num_test]
                test_examples = examples[-num_test:]
                assert len(test_examples) == num_test
                assert len(train_examples) >= len(test_examples)
                add_examples(all_test_examples, test_examples)
            add_examples(all_train_examples, train_examples)
    return users_and_plays, all_train_examples, all_test_examples

def _write_data_by_character(examples, output_directory):
    """Writes a collection of data files by play & character."""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for character_name, sound_bites in examples.items():
        filename = os.path.join(output_directory, character_name + '.txt')
        with open(filename, 'w') as output:
            for sound_bite in sound_bites:
                output.write(sound_bite + '\n')

def main(argv):
    print('Splitting .txt data between users')
    input_filename = argv[0]
    with open(input_filename, 'r') as input_file:
        shakespeare_full = input_file.read()
    plays, discarded_lines = _split_into_plays(shakespeare_full)
    print('Discarded %d lines' % len(discarded_lines))
    users_and_plays, all_examples, _ = _get_train_test_by_character(plays, test_fraction=-1.0)
    output_directory = argv[1]
    with open(os.path.join(output_directory, 'users_and_plays.json'), 'w') as ouf:
        json.dump(users_and_plays, ouf)
    _write_data_by_character(all_examples,
                             os.path.join(output_directory,
                                          'by_play_and_character/'))

if __name__ == '__main__':
    main(sys.argv[1:])