import numpy as np
import json


def create_prompt(dataset_name, dataset, example_per_class=1, example_selection="random", **kwargs):
    """
    Create prompt for label function generation
    """
    if dataset_name == "youtube":
        task = "spam classification"
        task_info = "In each iteration, the user will provide a comment for a video. Please decide whether the comment is a spam."
        class_info = "0 for non-spam, 1 for spam"
    elif dataset_name == "sms":
        task = "spam classification"
        task_info = "In each iteration, the user will provide a text message. Please decide whether the message is a spam. Hint: promotional " \
                    "messages should also be considered as spam messages."
        class_info = "0 for non-spam, 1 for spam"
    elif dataset_name == "imdb":
        task = "sentiment analysis"
        task_info = "In each iteration, the user will provide a movie review. Please decide whether the review is positive or negative."
        class_info = "0 for negative, 1 for positive"
    elif dataset_name == "yelp":
        task = "sentiment analysis"
        task_info = "In each iteration, the user will provide a restaurant review. Please decide whether the review is positive or negative."
        class_info = "0 for negative, 1 for positive"
    elif dataset_name == "chemprot":
        task = "biomedical relation extraction"
        task_info = "In each iteration, the user will provide a biomedical statement, followed by a question asking the relationship between two chemicals occured in that statement." \
                    "Please decide the relationship between the two chemicals based on the statement."
        class_info = "0 for chemical B (or A) is part of chemical A (or B), 1 for chemical B (or A) is the regulator of chemical A (or B). " \
                     "2 for chemical B (or A) is the upregulator of chemical A (or B). 3 for chemical B (or A) is the downregulator of chemical A (or B)." \
                     "4 for chemical B (or A) is the agnoist of chemical A (or B). 5 for chemical B (or A) is the antagonist of chemical A (or B)." \
                     "6 for chemical B (or A) is the modulator of chemical A (or B). 7 for chemical B (or A) is the cofactor of chemical A (or B)." \
                     "8 for chemical B (or A) is the substrate or product of chemical A (or B). 9 for the relationship between chemical A and chemical B is not listed above."
    elif dataset_name == "cdr":
        task = "chemical disease relation extraction"
        task_info = "In each iteration, the user will provide a biomedical passage, followed by a question asking whether a chemical causes " \
                    "a disease. Please decide whether the chemical causes the disease based on the passage. Hint: please be rigorous when making" \
                    "the causal claim, that is, only return 1 if the passage explictly states that the chemical causes the disease, and return 0" \
                    "when it only indicate a possibility of causal relationship."
        class_info = "0 for the chemical does not cause the disease, 1 for the chemical causes the disease"
    elif dataset_name == "agnews":
        task = "topic classification"
        task_info = "In each iteration, the user will provide a piece of news. Please classify the topic of the news into following categories."
        class_info = "0 for world news, 1 for sports news, 2 for business news, 3 for science or high technology news."
    elif dataset_name == "trec":
        task = "topic classification"
        task_info = "In each iteration, the user will provide a question. Please classify the topic of the question into following categories."
        class_info = "0 for questions asking for description and abstract concept, 1 for questions asking for an entity (animal, plant, color, etc.), " \
                     "2 for questions asking about a person or a group of persons, 3 for questions asking for an abbreviation," \
                     "4 for questions asking for a location, 5 for questions asking for a number (data, postcode, etc.)."
    elif dataset_name == "medical_abstract":
        task = "topic classification"
        task_info = "In each iteration, the user will provide a medical abstract. Please classify the topic of the abstract based on the disease it focuses on."
        class_info = "0 for neoplasms diseases, 1 for digestive system diseases, 2 for nervous system diseases, 3 for cardiovascular diseases, 4 for general pathological conditions."

    if "lf_type" in kwargs:
        lf_type = kwargs["lf_type"]
    else:
        lf_type = "keyword"

    if "explanation" in kwargs:
        explanation = kwargs["explanation"]
    else:
        explanation = False

    if lf_type == "keyword":
        if explanation:
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
    elif lf_type == "regex":
        if dataset_name == "cdr":
            regex_instruction = "In the regular expression, use {{A}} to represent the chemical and {{B}} to represent the disease that" \
                                " occur in the user's query. Use [SEP] to seperate multiple regular expressions."
        elif dataset_name == "chemprot":
            regex_instruction = "In the regular expression, use {{A}} to represent the first chemical and {{B}} to represent the second " \
                                "chemical that occur in the user's query. Use [SEP] to seperate multiple regular expressions."
        else:
            regex_instruction = ""

        if explanation:
            interaction_format = """
        After the user provides input, explain your reason process step by step. Then provide a regular expression such that
        if a passage matches the regex, it is likely to have the same label with the current input.{} If no regular expression
        can be identified, return NONE for regular expression. Finally, provide the class label for the input. The interaction 
        format is as follows. Replace the text in brackets when you respond to user query.
        User: 
        [Input text]
        Response:
        EXPLANATION: <Explain the reason process step by step>
        REGEX: <List of regular expressions>
        LABEL: <Predicted label>
        """.format(regex_instruction)
        else:
            interaction_format = """
        After the user provides input, provide a regular expression such that if a passage matches the regex, it is likely to 
        have the same label with the current input. {} If no regular expression can be identified, return NONE for regular expression. 
        Finally, provide the class label for the input. The interaction format is as follows. Replace the text in brackets when 
        you respond to user query.
        User: 
        [Input text]
        Response:
        REGEX: <List of regular expressions>
        LABEL: <Predicted label>
        """.format(regex_instruction)
    else:
        raise NotImplementedError(f"LF type {lf_type} not supported.")

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
                raise NotImplementedError("Example selection method not supported.")

        for idx in example_indices:
            user_input = examples[idx]["data"]
            label = examples[idx]["label"]

            if lf_type == "keyword":
                keywords = examples[idx]["keywords"]
                if explanation:
                    explanation = examples[idx]["explanation"]
                    example = "User:{}\nResponse:\nExplanation: {}\nKEYWORDS: {}\nLABEL: {}\n".format(
                        user_input, explanation, keywords, label)
                else:
                    example = "User:{}\nResponse:\nKEYWORDS: {}\nLABEL: {}\n".format(user_input, keywords, label)
            elif lf_type == "regex":
                regex = examples[idx]["regex"]
                if explanation:
                    explanation = examples[idx]["explanation"]
                    example = "User:{}\nResponse:\nExplanation: {}\nREGEX: {}\nLABEL: {}\n".format(
                        user_input, explanation, regex, label)
                else:
                    example = "User:{}\nResponse:\nREGEX: {}\nLABEL: {}\n".format(user_input, regex, label)

            example_string += example
    else:
        example_string = ""

    system_role = "helpful assistant"
    if "dp_aware" in kwargs and kwargs["dp_aware"]:
        task_prompt = """
TASK DESCRIPTION: 
You are a {} who helps users design label functions in a data programming task for {}. The label functions are composed
 of a keyword and a label, such that the keyword is indicative of the corresponding label. {} ({})
INTERACTION FORMAT: {}""".format(system_role, task, task_info, class_info, interaction_format)

    else:
        task_prompt = """
TASK DESCRIPTION: 
You are a {} who helps users in a {} task. {} ({})
INTERACTION FORMAT: {}""".format(system_role, task, task_info, class_info, interaction_format)

    return task_prompt, example_string