import argparse
import json
import os

from shake_utils import parse_data_in

parser = argparse.ArgumentParser()

parser.add_argument(
    "--raw",
    help="include users' raw .txt data in respective .json files",
    action="store_true",
)

parser.set_defaults(raw=False)

args = parser.parse_args()

parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

users_and_plays_path = os.path.join(
    parent_path, "data", "raw_data", "users_and_plays.json"
)
txt_dir = os.path.join(parent_path, "data", "raw_data", "by_play_and_character")
json_data = parse_data_in(txt_dir, users_and_plays_path, args.raw)
json_path = os.path.join(parent_path, "data", "all_data", "all_data.json")
with open(json_path, "w") as outfile:
    json.dump(json_data, outfile)
