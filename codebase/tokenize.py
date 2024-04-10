### Import Settings ###
from settings import BASE_DIR
from statistics import mean, median

### Import Dependencies ###
import gc
import itertools

import pandas as pd

import stanza

### Import Functions ###
from codebase.tools import string_to_dict, is_punctuation, tag_mode_selector

### Stanza ###
def process_text(text):
    gc.collect()
    #download stanza pipeline for tokenize, POS, lemma, NER, sentiment analysis
    nlp = stanza.Pipeline(lang='es', model_dir=str(BASE_DIR / 'models'), use_gpu=False, download_method=None, processors='tokenize, pos, lemma, ner, sentiment', verbose=False)
    doc = nlp(text)

    #calculate sentiment not for sentence but across item that consists of multiple sentences
    tokenized_content = []
    sentiment = 0
    sentiment_count = 0
    word_count_per_sentence_list =[]
    total_word_count = 0

    #look for doc-information (e.g., sentence-count for item)
    
    for sentence in doc.sentences:
        sentiment += sentence.sentiment
        sentiment_count += 1
        word_count_per_sentence = 0

        for word, entity in itertools.zip_longest(sentence.words, sentence.ents):
            entity_type = "" if entity else None

            #new column containing information
            tokenized_content.append({
                'word': word.text,
                'lemma': word.lemma,
                'upos': word.upos,
                'feats': string_to_dict(word.feats,'|') if word.feats != None else None,
                'ner': entity_type
            })
            ##get the word count per sentence and per item
            word_count_per_sentence +=1
            total_word_count += 1

        word_count_per_sentence_list.append(word_count_per_sentence)
    
        #calculate mean of sentiment across sentences in cell
    if sentiment_count == 0: 
        sentiment = None
    else: 
        sentiment = sentiment / sentiment_count

    ##get info on corpus      
    #get the count of sentences in the cell 
    num_sentences = len(doc.sentences)

    return tokenized_content, sentiment, num_sentences, total_word_count,  word_count_per_sentence_list


### Generating text ###
# input: dataframe and the analysed_content as column_name
# output: intermediate dataframe of processed_text
def generate_text(df, column_name, args):
    processed_data = []
    for index, row in df.iterrows():
        cache = ''
       # total_sentences += row['num_sentences']
        #for count in row['word_count_per_sentence']:
        #    word_count += count
        for word in row[column_name]:
            tags = ''
            if word.get('tag'):
                for tag in word['tag']:
                    tags += str(tag[0])
            else:
                pass
                
            if is_punctuation(word['word']):
                cache += str(word['word'])
            else:
                cache += ' ' + str(word['word']) + str(tags)

        if 'sentiment' in args.features or 'all' in args.features:
            negative_list = ['negativo', 'NEGATIVO', row['sentiment']]
            positive_list = ['positivo', 'POSITIVO', row['sentiment']]
            neutral_list = ['neutro', 'NEUTRO', row['sentiment']]

            # tag-mode selector
            mode_selector = tag_mode_selector(args)

            if row['sentiment'] < 0.66 and row['sentiment'] >= 0.0:
                sentiment_variable = negative_list[mode_selector]
            elif row['sentiment'] > 0.66 and row['sentiment'] < 1.33:
                sentiment_variable = neutral_list[mode_selector]
            elif row['sentiment'] > 1.33 and row['sentiment'] <=2.0:
                sentiment_variable = positive_list[mode_selector]
            else: 
                raise ValueError('Sentiment value out of range')
            
            cache += ' +' + str(sentiment_variable)

        processed_data.append({'tagged_content': cache})

   
    processed_df = pd.DataFrame(processed_data)
    return pd.concat([df, processed_df], axis=1)
