import numpy as np
import json


def create_prompt(dataset_name, dataset, example_per_class=1, example_selection="random", **kwargs):
    """
    Create prompt for label function generation
    """
    if dataset_name in ["youtube", "sms"]:
        task = "spam classification"
        class_info = "0 for non-spam, 1 for spam"
    elif dataset_name in ["imdb", "yelp"]:
        task = "sentiment analysis"
        class_info = "0 for negative, 1 for positive"
    elif dataset_name in ["chemprot"]:
        task = "biomedical relation extraction"
        class_info = "0 for chemical B is part of chemical A, 1 for chemical B is the regulator of chemical A. " \
                     "2 for chemical B is the upregulator of chemical A. 3 for chemical B is the downregulator of chemical A." \
                     "4 for chemical B is the agnoist of chemical A. 5 for chemical B is the antagonist of chemical A." \
                     "6 for chemical B is the antagonist of chemical A. 7 for chemical B is the modulator of chemical A." \
                     "8 for chemical B is the substrate or product of chemical A. 9 for none of the above."
    elif dataset_name in ["cdr"]:
        task = "chemical disease relation extraction"
        class_info = "0 for the chemical does not cause the disease, 1 for the chemical causes the disease"
    else:
        raise ValueError("dataset task not identified.")

    if dataset_name == "youtube":
        task_info = "In each iteration, the user will provide a comment for a video. Please decide whether the comment is a spam."
    elif dataset_name == "sms":
        task_info = "In each iteration, the user will provide a text message. Please decide whether the message is a spam."
    elif dataset_name == "imdb":
        task_info = "In each iteration, the user will provide a movie review. Please decide whether the review is positive or negative."
    elif dataset_name == "yelp":
        task_info = "In each iteration, the user will provide a product review. Please decide whether the review is positive or negative."
    elif dataset_name == "chemprot":
        task_info = "In each iteration, the user will provide a biomedical statement, followed with two chemicals occured in that statement." \
                    "Please decide the relationship between the two chemicals based on the statement."
    elif dataset_name == "cdr":
        task_info = "In each iteration, the user will provide a biomedical statement, followed with a chemical and a disease occured in " \
                    "that statement. Please decide whether there is a causal relationship between the chemical and the disease based on the statement."

    if "expert_role" in kwargs and kwargs["expert_role"]:
        system_role = f"{task} expert"
    else:
        system_role = "helpful assistant"

    if "explanation" in kwargs and kwargs["explanation"]:
        interaction_format = """
After the user provides input, explain your reason process step by step. Then identify a list of keywords that helps
making prediction. Finally, provide the class label for the input. The interaction format is as follows. Replace the 
text in brackets when you respond to user query.
User: 
[Input text]
Response:
EXPLANATION: <Explain the reason process step by step>
KEYWORDS: <List of keywords>
LABEL: <Predicted label>
"""
    else:
        interaction_format = """
After the user provides input, identify a list of keywords that helps making prediction. Then provide the class label 
for the input. The interaction format is as follows. Replace the text in brackets when you respond to user query.
User: 
[Input text]
Response:
KEYWORDS: <List of keywords>
LABEL: <Predicted label>
"""

    if example_per_class > 0:
        example_string = ""
        with open("examples.json") as json_file:
            example_dict = json.load(json_file)

        examples = example_dict[dataset_name]
        example_labels = []
        example_indices = []  # example indices in example file. NOT the original indices in validation set.
        for e in examples:
            example_labels.append(e["label"])

        n_class = dataset.n_class
        for c in range(n_class):
            active_indices = np.nonzero(np.array(example_labels) == c)[0]
            if example_selection == "random":
                assert len(active_indices) >= example_per_class
                selected_indices = np.random.choice(active_indices, example_per_class)
                example_indices += selected_indices.tolist()
            else:
                raise ValueError("Example selection method not supported.")

        for idx in example_indices:
            user_input = examples[idx]["data"]
            label = examples[idx]["label"]
            keywords = examples[idx]["keywords"]
            explanation = examples[idx]["explanation"]
            if "explanation" in kwargs and kwargs["explanation"]:
                example = "User:{}\nResponse:\nExplanation: {}\nKEYWORDS: {}\nLABEL: {}\n".format(
                    user_input, explanation, keywords, label)
            else:
                example = "User:{}\nResponse:\nKEYWORDS: {}\nLABEL: {}\n".format(user_input, keywords, label)
            example_string += example
    else:
        example_string = ""

    if "dp_aware" in kwargs and kwargs["dp_aware"]:
        task_prompt = """
TASK DESCRIPTION: 
You are a {} that helps users design label functions in a data programming task for {}. The label functions are composed
 of a keyword and a label, such that the keyword is indicative of the corresponding label. {} ({})
INTERACTION FORMAT: {}""".format(system_role, task, task_info, class_info, interaction_format)

    else:
        task_prompt = """
TASK DESCRIPTION: 
You are a {} that helps users in a {} task. {} ({})
INTERACTION FORMAT: {}""".format(system_role, task, task_info, class_info, interaction_format)

    return task_prompt, example_string