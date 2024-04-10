# The following taggers are available:
#     - "group" to tag deprived groups 
#     - "cta" to tag calls to action 
#     - "problem" to tag problem frames
#     - "1psg" to tag first person singular pronouns
#     - "discourse" to tag discourse markers
#     - "sentencemod" to tag sentence modifiers
#     - "all" to tag all of the above 
#     - "null" to tag nothing 

### Dependencies ###
from codebase.rules import Rules
from codebase.tokenize import generate_text


def tagger(df, wordlists, args):
    
    ### GROUP ###
    # Tag deprived groups
    # if 'group' in args.features or 'all' in args.features:
    #     for index, line in df.iterrows():
    #         cache = Rules.tag_content(line['tokenized_content'],
    #                                   wordlists['deprived_group'], '+GROUP')
    #         df.at[index, 'tokenized_content'] = cache
    if 'group' in args.features or 'all' in args.features:
        for index, line in df.iterrows():
            cache = Rules.tag_content(line['tokenized_content'],
                                    wordlists['deprived_group'], '+GRUPO')
            df.at[index, 'tokenized_content'] = cache

    ### CALL TO ACTION ###
    # # Tag calls to action
    # if 'cta' in args.features or 'all' in args.features:
    #     for index, line in df.iterrows():
    #         tokenized_content_list = line['tokenized_content']
    #         cache = Rules.tag_CTA(tokenized_content_list, '+CTA')
    #         df.at[index, 'tokenized_content'] = cache
    if 'cta' in args.features or 'all' in args.features:
        for index, line in df.iterrows():
            tokenized_content_list = line['tokenized_content']
            cache = Rules.tag_CTA(tokenized_content_list, '+ACCION')
            df.at[index, 'tokenized_content'] = cache

    ### PROBLEM ###
    # Tag problem frames
    # if 'problem' in args.features or 'all' in args.features:
    #     for index, line in df.iterrows():
    #         cache = Rules.tag_content(line['tokenized_content'], wordlists['problem_frame'], '+PROBLEM')
    #         df.at[index, 'tokenized_content'] = cache

        # Tag problem frames
    if 'problem' in args.features or 'all' in args.features:
        for index, line in df.iterrows():
            cache = Rules.tag_content(line['tokenized_content'], wordlists['problem_frame'], '+PROBLEMA')
            df.at[index, 'tokenized_content'] = cache

    ### FIRST PERSON SINGULAR ###
    # Tag first person singular pronouns
    if '1psg' in args.features or 'all' in args.features:
        for index, line in df.iterrows():
            tokenized_content_list = line['tokenized_content']
            cache = Rules.tag_pronouns(tokenized_content_list)
            df.at[index, 'tokenized_content'] = cache

    ### DISCOURSE MARKERS ###
    # Tag discourse markers
    # if 'discourse' in args.features or 'all' in args.features:
    #     # Tag adversative conjunctions
    #     for index, line in df.iterrows():
    #         cache = Rules.tag_content(line['tokenized_content'],wordlists['adversative_conjunctions_discourse'] , '+DISCOURSE')
    #         df.at[index, 'tokenized_content'] = cache

    #     # Tag causal
    #     for index, line in df.iterrows():
    #         cache = Rules.tag_content(line['tokenized_content'],wordlists['causal_discourse'] , '+DISCOURSE')
    #         df.at[index, 'tokenized_content'] = cache

    #     # Tag consecutive
    #     for index, line in df.iterrows():
    #         cache = Rules.tag_content(line['tokenized_content'],wordlists['consecutive_discourse'] , '+DISCOURSE')
    #         df.at[index, 'tokenized_content'] = cache   

    if 'discourse' in args.features or 'all' in args.features:
        # Tag adversative conjunctions
        for index, line in df.iterrows():
            cache = Rules.tag_content(line['tokenized_content'],wordlists['adversative_conjunctions_discourse'] , '+DISCURSO')
            df.at[index, 'tokenized_content'] = cache

        # Tag causal
        for index, line in df.iterrows():
            cache = Rules.tag_content(line['tokenized_content'],wordlists['causal_discourse'] , '+DISCURSO')
            df.at[index, 'tokenized_content'] = cache

        # Tag consecutive
        for index, line in df.iterrows():
            cache = Rules.tag_content(line['tokenized_content'],wordlists['consecutive_discourse'] , '+DISCURSO')
            df.at[index, 'tokenized_content'] = cache  

    ### SENTENCE MODIFIER ###
    # Tag sentence modifiers
    # if 'sentencemod' in args.features or 'all' in args.features:
    #     # Tag intensifiers
    #     for index, line in df.iterrows():
    #         cache = Rules.tag_content(line['tokenized_content'], wordlists['sentence_mod_intensifier'], '+MODIFIER')
    #         df.at[index, 'tokenized_content'] = cache

    #     # Tag negation
    #     for index, line in df.iterrows():
    #         cache = Rules.tag_content(line['tokenized_content'], wordlists['sentence_mod_negation'], '+MODIFIER')
    #         df.at[index, 'tokenized_content'] = cache

    #     # Tag restriction
    #     for index, line in df.iterrows():
    #         cache = Rules.tag_content(line['tokenized_content'], wordlists['sentence_mod_restriction'], '+MODIFIER')
    #         df.at[index, 'tokenized_content'] = cache

        # Tag sentence modifiers
    if 'sentencemod' in args.features or 'all' in args.features:
        # Tag intensifiers
        for index, line in df.iterrows():
            cache = Rules.tag_content(line['tokenized_content'], wordlists['sentence_mod_intensifier'], '+MODIFICADOR')
            df.at[index, 'tokenized_content'] = cache

        # Tag negation
        for index, line in df.iterrows():
            cache = Rules.tag_content(line['tokenized_content'], wordlists['sentence_mod_negation'], '+MODIFICADOR')
            df.at[index, 'tokenized_content'] = cache

        # Tag restriction
        for index, line in df.iterrows():
            cache = Rules.tag_content(line['tokenized_content'], wordlists['sentence_mod_restriction'], '+MODIFICADOR')
            df.at[index, 'tokenized_content'] = cache

    result_df = generate_text(df, 'tokenized_content', args)

    return result_df