### RULES FOR TAGGING###

# This class contains methods for tagging different parts of speech in a text.

class Rules:
    # This method tags content words (nouns, verbs, adjectives, adverbs).
    # It takes as input a list of tokens (cell), a list of words to tag (wordlist), and a tag to apply.
    @staticmethod
    def tag_content(cell, wordlist, tag):
        # The switch variable is used to determine whether to match words or lemmas.
        switch='word'
        
        # Initialize 'tag' as an empty list if it doesn't exist for each token.
        for token in cell:
            if token.get('tag') is None:
                token['tag'] = []
        
        # Iterate over the words to tag.
        for word in wordlist:
            # If the word is a verb, match lemmas instead of words.
            if word[1] == 'V':
                switch = 'lemma'
            else:
                switch = 'word'
            
            # Iterate over the tokens in the cell.
            for count, token in enumerate(cell):
                # If the word is a multi-word expression,
                if len(word[0].split(' ')) > 1:
                    multi_word = word[0].split(' ')
                    cache = []
                    # Iterate over the words in the multi-word expression.
                    for j, single_word in enumerate(multi_word):
                        id = count + j
                        if id < len(cell):
                            # If the word matches the current token, add it to the cache.
                            if single_word == cell[id][switch]:
                                cache.append(cell[id])
                            else:
                                break
                        else: 
                            break
                    # If all words in the multi-word expression match, add the tag to each token.
                    if len(cache) == len(word[0].split(' ')):
                        for k, item in enumerate(cache):
                            id = count + k
                            if item['upos'] == word[1]:
                                cell[id]['tag'].append([tag])
                # If the word is a single word and it matches the current token, add the tag.
                elif token[switch] == word[0] and token['upos'] == word[1]:
                    token['tag'].append([tag])
        return cell

    # This method tags calls to action (CTA).
    # It takes as input a list of tokens (cell) and a tag to apply.
    @staticmethod
    def tag_CTA(cell, tag):
        # Initialize 'tag' as an empty list if it doesn't exist for each token.
        for token in cell:
            if token.get('tag') is None:
                token['tag'] = []
        
        # Iterate over the tokens in the cell.
        for count, token in enumerate(cell):
            # If the current token is a CTA verb followed by another verb, add the tag.
            if token['lemma'] == 'deber'and cell[count+1]['upos'] == 'VERB':
                cell[count]['tag'].append([tag])
            # If the current token is 'tener' followed by 'que' and another verb, add the tag.
            elif token['lemma'] == 'tener'and cell[count+1]['lemma'] == 'que' and cell[count+2]['upos'] == 'VERB':
                cell[count]['tag'].append([tag])
            # If the current token is 'hay' followed by 'que' and another verb, add the tag.
            elif token['word'] == 'hay'and cell[count+1]['word'] == 'que' and cell[count+2]['upos'] == 'VERB':
                cell[count]['tag'].append([tag])
            #elif token['upos'] == 'VERB' and token['feats'].get('Mood') == 'Imp':
            #    cell[count]['tag'].append([tag])
        return cell

    # This method tags pronouns.
    # It takes as input a list of tokens (cell).
    @staticmethod
    def tag_pronouns(cell):
        # Initialize 'tag' as an empty list if it doesn't exist for each token.
        for token in cell:
            if token.get('tag') is None:
                token['tag'] = []
        
        # Iterate over the tokens in the cell.
        for token in cell:
            # If the current token is a pronoun,
            if token['upos'] == 'PRON':
                # If the pronoun has number and person features
                if token['feats'].get('Number') != None and token['feats'].get('Person') != None:
                    # If the pronoun is first person singular, add the tag.
                    if token['feats']['Number'] == 'Sing' and token['feats']['Person'] == '1':
                        token['tag'].append(['+' + token['feats']['Person'] + 'SG'])

        return cell