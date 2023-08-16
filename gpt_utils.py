task_map = {
    "youtube": "spam",
    "sms": "spam",
    "AmazonReview": "sentiment",
    "IMDB": "sentiment",
    "Yelp": "sentiment"
}


prompt_map = {
    "spam":
        {
            "v1": # few-shot prompt
                """
                You are a helpful assistant that helps users design label functions in a data programming task for spam detection.
                The label functions are composed of a keyword and a label (0 for non-spam, 1 for spam), such that when the keyword
                is indicative of the corresponding label. When the user provides a sentence, first provides the label for that 
                sentence, then provide a set of keywords occurred in the sentence that helps making predictions. Each keyword must 
                be a single word in the sentence. When no keywords are indicative of the label, return NA for keywords.
                Example:
                User: 
                Huh, anyway check out this you[tube] channel: kobyoshi02
                Response:
                LABEL: 1
                KEYWORDS: check channel
                User:
                i turned it on mute as soon is i came on i just wanted to check the views...
                Response:
                LABEL: 0
                KEYWORDS: views
                """,
            "v2": # few-shot prompt with explanations
                """
                You are a helpful assistant that helps users design label functions in a data programming task for spam detection. 
                The label functions are composed of a keyword and a label (0 for non-spam, 1 for spam), such that when the keyword
                is indicative of the corresponding label. When the user provides a sentence, first provides the label for that sentence, 
                then provide a set of keywords that occurred in the sentence that helps make predictions. Each keyword must be a single word in the sentence.
                When no keywords are indicative of the label, return NA for keywords. Finally, briefly explain why the keywords are selected.
                Example:
                User: 
                Huh, anyway check out this you[tube] channel: kobyoshi02
                Response:
                LABEL: 1
                KEYWORDS: check channel
                EXPLANATION: the spammer asks viewers to check a youtube channel, which is an indicator of spam. Thus the keywords "check" and "channel" are indicative of spam.
                User:
                i turned it on mute as soon is i came on i just wanted to check the views...
                Response:
                LABEL: 0
                KEYWORDS: views
                EXPLANATION: the commenter expresses their own intention of checking the views, thus the keyword "views" provides the context and indicates non-spam.
                """
        }

    ,
    "sentiment":
    {
        "v1": # few-shot prompt
            """
            You are a helpful assistant that helps users design label functions in a data programming task for sentiment analysis.
            The label functions are composed of a keyword and a label (0 for negative, 1 for positive), such that when the keyword
            is indicative of the corresponding label. When the user provides a sentence, first provides the label for that 
            sentence, then provide a set of keywords occurred in the sentence that helps making predictions. Each keyword must 
            be a single word in the sentence. 
            Example:
            User: 
            I went and saw this movie last night after being coaxed to by a few friends of mine. 
            I'll admit that I was reluctant to see it because from what I knew of Ashton Kutcher he was only able to do comedy. 
            I was wrong. Kutcher played the character of Jake Fischer very well, and Kevin Costner played Ben Randall with such professionalism. 
            The sign of a good movie is that it can toy with our emotions. This one did exactly that. 
            The entire theater (which was sold out) was overcome by laughter during the first half of the movie, and were moved to tears during the second half. 
            While exiting the theater I not only saw many women in tears, but many full grown men as well, trying desperately not to let anyone see them crying. 
            This movie was great, and I suggest that you go see it before you judge.
            Response:
            LABEL: 1
            KEYWORDS: good well great
            User:
            Blake Edwards' legendary fiasco, begins to seem pointless after just 10 minutes. A combination of The Eagle Has Landed, Star!, 
            Oh! What a Lovely War!, and Edwards' Pink Panther films, Darling Lili never engages the viewer; the aerial sequences, the musical numbers, 
            the romance, the comedy, and the espionage are all ho hum. At what point is the viewer supposed to give a damn? This disaster wavers in tone, 
            never decides what it wants to be, and apparently thinks it's a spoof, but it's pathetically and grindingly square. Old fashioned in the worst sense, 
            audiences understandably stayed away in droves. It's awful. James Garner would have been a vast improvement over Hudson who is just cardboard, 
            and he doesn't connect with Andrews and vice versa. And both Andrews and Hudson don't seem to have been let in on the joke and perform with a 
            miscalculated earnestness. Blake Edwards' SOB isn't much more than OK, but it's the only good that ever came out of Darling Lili. 
            The expensive and professional look of much of Darling Lili, only make what it's all lavished on even more difficult to bear. To quote
            Paramount chief Robert Evans, "24 million dollars worth of film and no picture".
            Response:
            LABEL: 0
            KEYWORDS: pointless worst damn awful
            """
    }
}

def get_system_prompt(dataset, prompt_version="v1"):
    task = task_map[dataset]
    prompt = prompt_map[task][prompt_version]
    return prompt