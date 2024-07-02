

def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}


def change_target(x):
    if 'positive' in x or 'Positive' in x:
        return 'positive'
    elif 'negative' in x or 'Negative' in x:
        return 'negative'
    else:
        return 'neutral'


# A dictionary to store various prompt templates.
template_dict = {
    'default': 'Instruction: {instruction}\nInput: {input}\nAnswer: '
}


def test_mapping(args, feature):
    """
    Generate a mapping for testing purposes by constructing a prompt based on given instructions and input.

    Args:
    args (Namespace): A namespace object that holds various configurations, including the instruction template.
    feature (dict): A dictionary containing 'instruction' and 'input' fields used to construct the prompt.

    Returns:
    dict: A dictionary containing the generated prompt.

    Raises:
    ValueError: If 'instruction' or 'input' are not provided in the feature dictionary.
    """
    # Ensure 'instruction' and 'input' are present in the feature dictionary.
    if 'instruction' not in feature or 'input' not in feature:
        raise ValueError("Both 'instruction' and 'input' need to be provided in the feature dictionary.")

    # Construct the prompt using the provided instruction and input.
    prompt = get_prompt(
        args.instruct_template,
        feature['instruction'],
        feature['input']
    )

    return {
        "prompt": prompt,
    }


def get_prompt(template, instruction, input_text):
    """
    Generates a prompt based on a predefined template, instruction, and input.

    Args:
    template (str): The key to select the prompt template from the predefined dictionary.
    instruction (str): The instruction text to be included in the prompt.
    input_text (str): The input text to be included in the prompt.

    Returns:
    str: The generated prompt.

    Raises:
    KeyError: If the provided template key is not found in the template dictionary.
    """
    if not instruction:
        return input_text

    if template not in template_dict:
        raise KeyError(f"Template '{template}' not found. Available templates: {', '.join(template_dict.keys())}")

    return template_dict[template].format(instruction=instruction, input=input_text)
