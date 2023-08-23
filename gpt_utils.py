task_map = {
    "youtube": "spam",
    "sms": "spam",
    "AmazonReview": "sentiment",
    "IMDB": "sentiment",
    "imdb": "sentiment",
    "Yelp": "sentiment",
    "yelp": "sentiment"
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
    },
    "chemical":
    {
        """
        You are a chemical expert that design label functions in a data programming task for chemical relationship classification.
        The Label function are composed of a keyword and a label, such that the keyword is indicative of the corresponding label. 
        When the user provide a sentence and two chemicals occur in the sentence, first provide the label that indicates the relationship
        of the two chemicals in the sentence, then provide a set of keywords occurred in the sentence that helps making predictions. 
        Each keyword must be a single word in the sentence. Return NA for keywords if no indicative keywords can be identified.
        There are 10 possible classes list as follows:
        '''
            "0": "Part of", 
            "1": "Regulator",
            "2": "Upregulator",
            "3": "Downregulator",
            "4": "Agonist",
            "5": "Antagonist",
            "6": "Modulator",
            "7": "Cofactor",
            "8": "Substrate/Product",
            "9": "NOT", which indicate none of the above
        '''
        Example:
        User: 
        Identification and characterization of a novel flavin-containing spermine oxidase of mammalian cell origin. <spermine oxidase> <flavin>
        Response:
        LABEL: 0
        KEYWORDS: containing
        User: 
        Among neuroleptics, the four most potent compounds at the human serotonin transporter were triflupromazine, 
        fluperlapine, chlorpromazine, and ziprasidone (K(D) 24-39 nM); and at the norepinephrine transporter, chlorpromazine, 
        zotepine, chlorprothixene, and promazine (K(D) 19-25 nM). <norepinephrine transporter> <chlorpromazine>
        Response:
        LABEL: 1
        KEYWORDS: neuroleptics
        User:
        Lintitript markedly increased postprandial plasma CCK release (P<0.001) while distinctly reducing postprandial 
        PP levels (P<0.01) as compared to placebo. <CCK> <Lintitript>
        Response:
        LABEL: 2
        KEYWORDS: increased
        User: 
        Selective inhibition of PDE5 is a rational therapeutic approach in ED, as proved by the clinical success 
        of sildenafil. <PDE5> <sildenafil>
        Response:
        LABEL: 3
        KEYWORDS: inhibition
        User: 
        Cellular release of AChE by SH-SY5Y is significantly enhanced by the muscarinic acetylcholine receptor (mAChR) 
        agonists carbachol or muscarine, with the effect of carbachol blocked by the mAChR antagonist atropine. <mAChR> <muscarine>
        Response:
        LABEL: 4
        KEYWORDS: agonists
        User: 
        While a number of orally active non-peptide V(2) antagonists (Vaptans); notably, Tolvaptan, Lixivaptan and Satavaptan, 
        are currently in Phase III clinical trials; to date, only the mixed V(2)/V(1a), antagonist Conivaptan (Vaprisol), 
        has been approved by the US FDA for clinical use (by i.v. <V(2)> <Lixivaptan>
        Response:
        LABEL: 5
        KEYWORDS: antagoists
        User:
        Anxiolytic- but not antidepressant-like activity of Lu AF21934, a novel, selective positive allosteric modulator of the mGlu\u2084 receptor. <mGlu\u2084> <Lu AF21934>
        Response:
        LABEL: 6
        KEYWORDS: modulator
        User: 
        Phenylbutazone (PB), a nonsteroidal anti-inflammatory drug, is an efficient reducing cofactor for the peroxidase activity of prostaglandin H synthase (PHS). <PHS> <Phenylbutazone>
        Response:
        LABEL: 7
        KEYWORDS: cofactor
        User: 
        Furthermore, knockdown of OPN enhanced cell death caused by other drugs, including paclitaxel, doxorubicin, 
        actinomycin-D, and rapamycin, which are also P-gp substrates. <P-gp> <paclitaxel>
        Response:
        LABEL: 8
        KEYWORDS: substrates  
        User: 
        Jo2-induced activation of caspase-3 or -9 in liver tissues was inhibited by minocycline pretreatment, and yet 
        the direct addition of minocycline to liver extracts from Jo2-challenged mice failed to block caspase activation in vitro.
        <caspase> <minocycline>
        Response:
        LABEL: 9
        KEYWORDS: NA   
        """
    }
}

def get_system_prompt(dataset, prompt_version="v1"):
    task = task_map[dataset]
    prompt = prompt_map[task][prompt_version]
    return prompt