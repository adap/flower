import pathlib
import re
import sys

import yaml


# Load the YAML configuration
with (pathlib.Path(__file__).parent.resolve() / "changelog.yml").open("r") as file:
    config = yaml.safe_load(file)

# Extract types, project, and scope from the config
types = "|".join(config["types"])
project = "|".join(config["project"]) + "|\\*"
scope = config["scope"]

# Construct the pattern
pattern_template = config["pattern_template"]
pattern = pattern_template.format(types=types, project=project, scope=scope)

# Check for the pattern in the first argument given to the script
if re.search(pattern, sys.argv[1]):
    print("PR title is valid")
    sys.exit(0)
else:
    print(
        "PR title format is invalid, it should be of the form: <PR_TYPE>(<PR_SCOPE>) "
        f"<PR_SUBJECT> with <PR_TYPE> in {types}, and "
        f"<PR_SCOPE> in {project} where '*' is used when modifying multiple projects),"
        "and <PR_SUBJECT> starting with "
        "a capitalized verb in the imperative mood and without a dot at the end.\n"
        "A valid example is: `feat(framework) Add flwr build CLI command`"
    )
    sys.exit(1)
