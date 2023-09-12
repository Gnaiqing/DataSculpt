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
After the user provides input, first provide a set of indicative keywords that helps making prediction. 
Then provide the class label for the input. Finally, explain why the label is selected based on the identified keywords. 
The interaction format is as follows. Replace the text in brackets when you respond to user query.
User: 
[Input text]
Response:
KEYWORDS: <List of keywords>
LABEL: <Predicted label>
EXPLANATION: <Explanations for the prediction result.>
"""
    else:
        interaction_format = """
After the user provides input, first provide a set of indicative keywords that helps making prediction. 
Then provide the class label for the input. The interaction format is as follows. Replace the text in brackets when you respond to user query.
User: 
[Input text]
Response:
KEYWORDS: <List of keywords>
LABEL: <Predicted label>
"""

    if example_per_class > 0:
        example_string = "EXAMPLE:\n"
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
                example = "User:\n{}\nResponse:\nKEYWORDS: {}\nLABEL: {}\nExplanation: {}".format(
                    user_input, keywords, label, explanation)
            else:
                example = "User:\n{}\nResponse:\nKEYWORDS: {}\nLABEL: {}\n".format(user_input, keywords, label)
            example_string += example
    else:
        example_string = ""

    if "dp_aware" in kwargs and kwargs["dp_aware"]:
        task_prompt = """
TASK DESCRIPTION: 
You are a {} that helps users design label functions in a data programming task for {}. The label functions are composed
 of a keyword and a label, such that the keyword is indicative of the corresponding label. {} ({})
INTERACTION FORMAT: 
{}
{}""".format(system_role, task, task_info, class_info, interaction_format, example_string)

    else:
        task_prompt = """
TASK DESCRIPTION: 
You are a {} that helps users in a {} task. {} ({})
INTERACTION FORMAT: {} {}""".format(system_role, task, task_info, class_info, interaction_format, example_string)

    return task_prompt


# prompt_map = {
#     "spam":
#         {
#             "v1": # few-shot prompt
#                 """
#                 You are a helpful assistant that helps users design label functions in a data programming task for spam detection.
#                 The label functions are composed of a keyword and a label (0 for non-spam, 1 for spam), such that when the keyword
#                 is indicative of the corresponding label. When the user provides a sentence, first provides the label for that
#                 sentence, then provide a set of keywords occurred in the sentence that helps making predictions. Each keyword must
#                 be a single word in the sentence. When no keywords are indicative of the label, return NA for keywords.
#                 Example:
#                 User:
#                 Huh, anyway check out this you[tube] channel: kobyoshi02
#                 Response:
#                 LABEL: 1
#                 KEYWORDS: check channel
#                 User:
#                 i turned it on mute as soon is i came on i just wanted to check the views...
#                 Response:
#                 LABEL: 0
#                 KEYWORDS: views
#                 """,
#             "v2": # few-shot prompt with explanations
#                 """
#                 You are a helpful assistant that helps users design label functions in a data programming task for spam detection.
#                 The label functions are composed of a keyword and a label (0 for non-spam, 1 for spam), such that when the keyword
#                 is indicative of the corresponding label. When the user provides a sentence, first provides the label for that sentence,
#                 then provide a set of keywords that occurred in the sentence that helps make predictions. Each keyword must be a single word in the sentence.
#                 When no keywords are indicative of the label, return NA for keywords. Finally, briefly explain why the keywords are selected.
#                 Example:
#                 User:
#                 Huh, anyway check out this you[tube] channel: kobyoshi02
#                 Response:
#                 LABEL: 1
#                 KEYWORDS: check channel
#                 EXPLANATION: the spammer asks viewers to check a youtube channel, which is an indicator of spam. Thus the keywords "check" and "channel" are indicative of spam.
#                 User:
#                 i turned it on mute as soon is i came on i just wanted to check the views...
#                 Response:
#                 LABEL: 0
#                 KEYWORDS: views
#                 EXPLANATION: the commenter expresses their own intention of checking the views, thus the keyword "views" provides the context and indicates non-spam.
#                 """
#         }
#
#     ,
#     "sentiment":
#     {
#         "v1": # few-shot prompt
#             """
#             You are a helpful assistant that helps users design label functions in a data programming task for sentiment analysis.
#             The label functions are composed of a keyword and a label (0 for negative, 1 for positive), such that when the keyword
#             is indicative of the corresponding label. When the user provides a sentence, first provides the label for that
#             sentence, then provide a set of keywords occurred in the sentence that helps making predictions. Each keyword must
#             be a single word in the sentence.
#             Example:
#             User:
#             I went and saw this movie last night after being coaxed to by a few friends of mine.
#             I'll admit that I was reluctant to see it because from what I knew of Ashton Kutcher he was only able to do comedy.
#             I was wrong. Kutcher played the character of Jake Fischer very well, and Kevin Costner played Ben Randall with such professionalism.
#             The sign of a good movie is that it can toy with our emotions. This one did exactly that.
#             The entire theater (which was sold out) was overcome by laughter during the first half of the movie, and were moved to tears during the second half.
#             While exiting the theater I not only saw many women in tears, but many full grown men as well, trying desperately not to let anyone see them crying.
#             This movie was great, and I suggest that you go see it before you judge.
#             Response:
#             LABEL: 1
#             KEYWORDS: good well great
#             User:
#             Blake Edwards' legendary fiasco, begins to seem pointless after just 10 minutes. A combination of The Eagle Has Landed, Star!,
#             Oh! What a Lovely War!, and Edwards' Pink Panther films, Darling Lili never engages the viewer; the aerial sequences, the musical numbers,
#             the romance, the comedy, and the espionage are all ho hum. At what point is the viewer supposed to give a damn? This disaster wavers in tone,
#             never decides what it wants to be, and apparently thinks it's a spoof, but it's pathetically and grindingly square. Old fashioned in the worst sense,
#             audiences understandably stayed away in droves. It's awful. James Garner would have been a vast improvement over Hudson who is just cardboard,
#             and he doesn't connect with Andrews and vice versa. And both Andrews and Hudson don't seem to have been let in on the joke and perform with a
#             miscalculated earnestness. Blake Edwards' SOB isn't much more than OK, but it's the only good that ever came out of Darling Lili.
#             The expensive and professional look of much of Darling Lili, only make what it's all lavished on even more difficult to bear. To quote
#             Paramount chief Robert Evans, "24 million dollars worth of film and no picture".
#             Response:
#             LABEL: 0
#             KEYWORDS: pointless worst damn awful
#             """
#     },
#     "chemical":
#     {
#         "v1":
#         """
#         You are a chemical expert that design label functions in a data programming task for chemical relationship classification.
#         The Label function are composed of a keyword and a label, such that the keyword is indicative of the corresponding label.
#         When the user provide a sentence and two chemicals occur in the sentence, first provide the label that indicates the relationship
#         of the two chemicals in the sentence, then provide a set of keywords occurred in the sentence that helps making predictions.
#         Each keyword must be a single word in the sentence. Return NA for keywords if no indicative keywords can be identified.
#         There are 10 possible classes list as follows:
#         '''
#             "0": "Part of",
#             "1": "Regulator",
#             "2": "Upregulator",
#             "3": "Downregulator",
#             "4": "Agonist",
#             "5": "Antagonist",
#             "6": "Modulator",
#             "7": "Cofactor",
#             "8": "Substrate/Product",
#             "9": "NOT", which indicate none of the above
#         '''
#         Example:
#         User:
#         Identification and characterization of a novel flavin-containing spermine oxidase of mammalian cell origin. <spermine oxidase> <flavin>
#         Response:
#         LABEL: 0
#         KEYWORDS: containing
#         User:
#         Among neuroleptics, the four most potent compounds at the human serotonin transporter were triflupromazine,
#         fluperlapine, chlorpromazine, and ziprasidone (K(D) 24-39 nM); and at the norepinephrine transporter, chlorpromazine,
#         zotepine, chlorprothixene, and promazine (K(D) 19-25 nM). <norepinephrine transporter> <chlorpromazine>
#         Response:
#         LABEL: 1
#         KEYWORDS: neuroleptics
#         User:
#         Lintitript markedly increased postprandial plasma CCK release (P<0.001) while distinctly reducing postprandial
#         PP levels (P<0.01) as compared to placebo. <CCK> <Lintitript>
#         Response:
#         LABEL: 2
#         KEYWORDS: increased
#         User:
#         Selective inhibition of PDE5 is a rational therapeutic approach in ED, as proved by the clinical success
#         of sildenafil. <PDE5> <sildenafil>
#         Response:
#         LABEL: 3
#         KEYWORDS: inhibition
#         User:
#         Cellular release of AChE by SH-SY5Y is significantly enhanced by the muscarinic acetylcholine receptor (mAChR)
#         agonists carbachol or muscarine, with the effect of carbachol blocked by the mAChR antagonist atropine. <mAChR> <muscarine>
#         Response:
#         LABEL: 4
#         KEYWORDS: agonists
#         User:
#         While a number of orally active non-peptide V(2) antagonists (Vaptans); notably, Tolvaptan, Lixivaptan and Satavaptan,
#         are currently in Phase III clinical trials; to date, only the mixed V(2)/V(1a), antagonist Conivaptan (Vaprisol),
#         has been approved by the US FDA for clinical use (by i.v. <V(2)> <Lixivaptan>
#         Response:
#         LABEL: 5
#         KEYWORDS: antagoists
#         User:
#         Anxiolytic- but not antidepressant-like activity of Lu AF21934, a novel, selective positive allosteric modulator of the mGlu\u2084 receptor. <mGlu\u2084> <Lu AF21934>
#         Response:
#         LABEL: 6
#         KEYWORDS: modulator
#         User:
#         Phenylbutazone (PB), a nonsteroidal anti-inflammatory drug, is an efficient reducing cofactor for the peroxidase activity of prostaglandin H synthase (PHS). <PHS> <Phenylbutazone>
#         Response:
#         LABEL: 7
#         KEYWORDS: cofactor
#         User:
#         Furthermore, knockdown of OPN enhanced cell death caused by other drugs, including paclitaxel, doxorubicin,
#         actinomycin-D, and rapamycin, which are also P-gp substrates. <P-gp> <paclitaxel>
#         Response:
#         LABEL: 8
#         KEYWORDS: substrates
#         User:
#         Jo2-induced activation of caspase-3 or -9 in liver tissues was inhibited by minocycline pretreatment, and yet
#         the direct addition of minocycline to liver extracts from Jo2-challenged mice failed to block caspase activation in vitro.
#         <caspase> <minocycline>
#         Response:
#         LABEL: 9
#         KEYWORDS: NA
#         """
#     }
# }
#
# def get_system_prompt(dataset, prompt_version="v1"):
#     task_map = {
#         "youtube": "spam",
#         "sms": "spam",
#         "imdb": "sentiment",
#         "yelp": "sentiment",
#         "chemprot": "chemical"
#     }
#
#     task = task_map[dataset]
#     prompt = prompt_map[task][prompt_version]
#     return prompt