### Import Dependencies ###
import re

import pandas as pd

### Import Functions ###
from codebase.tools import default_normalize
from codebase.feature_importance import BasicImportance

### FUNCTIONS ###
def vectorize_dense(df, args):
    #initialize cache
    cache = []
    dummy_vec = []

    #populate dummy_vec
    if 'group' in args.features or 'all' in args.features:
        dummy_vec.append('group')
    if 'cta' in args.features or 'all' in args.features:
        dummy_vec.append('cta')
    if 'problem' in args.features or 'all' in args.features:
        dummy_vec.append('problem')
    if '1psg' in args.features or 'all' in args.features:
        dummy_vec.append('1psg')
    if 'discourse' in args.features or 'all' in args.features:
        dummy_vec.append('discourse')
    if 'sentencemod' in args.features or 'all' in args.features:
        dummy_vec.append('sentencemod')
    if 'sentiment' in args.features or 'all' in args.features:
        dummy_vec.append('sentiment')
    
    #count tags in tagged df
    for index, row in df.iterrows():
        num_sentences = row['num_sentences']
        vector =  []
        # if 'group' in args.features or 'all' in args.features:
        #     vector.append(default_normalize(len(re.findall(r'(\+GROUP)', str(row['tagged_content']), re.IGNORECASE)), num_sentences))
        # if 'cta' in args.features or 'all' in args.features:
        #     vector.append(default_normalize(len(re.findall(r'(\+CTA)', str(row['tagged_content']), re.IGNORECASE)), num_sentences))
        # if 'problem' in args.features or 'all' in args.features:
        #     vector.append(default_normalize(len(re.findall(r'(\+PROBLEM)', str(row['tagged_content']), re.IGNORECASE)), num_sentences))
        # if '1psg' in args.features or 'all' in args.features:
        #     vector.append(default_normalize(len(re.findall(r'(\+1SG)', str(row['tagged_content']), re.IGNORECASE)), num_sentences))
        # if 'discourse' in args.features or 'all' in args.features:
        #     vector.append(default_normalize(len(re.findall(r'(\+DISCOURSE)', str(row['tagged_content']), re.IGNORECASE)), num_sentences))
        # if 'sentencemod' in args.features or 'all' in args.features:
        #     vector.append(default_normalize(len(re.findall(r'(\+MODIFIER)', str(row['tagged_content']), re.IGNORECASE)), num_sentences))
        # if 'sentiment' in args.features or 'all' in args.features:
        #     vector.append(row['sentiment'])
        
        if 'group' in args.features or 'all' in args.features:
            vector.append(default_normalize(len(re.findall(r'(\+GRUPO)', str(row['tagged_content']), re.IGNORECASE)), num_sentences))
        if 'cta' in args.features or 'all' in args.features:
            vector.append(default_normalize(len(re.findall(r'(\+ACCION)', str(row['tagged_content']), re.IGNORECASE)), num_sentences))
        if 'problem' in args.features or 'all' in args.features:
            vector.append(default_normalize(len(re.findall(r'(\+PROBLEMA)', str(row['tagged_content']), re.IGNORECASE)), num_sentences))
        if '1psg' in args.features or 'all' in args.features:
            vector.append(default_normalize(len(re.findall(r'(\+1SG)', str(row['tagged_content']), re.IGNORECASE)), num_sentences))
        if 'discourse' in args.features or 'all' in args.features:
            vector.append(default_normalize(len(re.findall(r'(\+DISCURSO)', str(row['tagged_content']), re.IGNORECASE)), num_sentences))
        if 'sentencemod' in args.features or 'all' in args.features:
            vector.append(default_normalize(len(re.findall(r'(\+MODIFICADOR)', str(row['tagged_content']), re.IGNORECASE)), num_sentences))
        if 'sentiment' in args.features or 'all' in args.features:
            vector.append(row['sentiment'])
        grievance = row['grievance']
        # Append the vector and grievance to the DataFrame
        cache.append({'grievance': grievance, 'vector': vector})

    # Convert the cache to a DataFrame
    vectorized_df = pd.DataFrame(cache)

    if args.scaler:
        vectorized_df = BasicImportance.scale_x(vectorized_df, 'vector')

    return vectorized_df, dummy_vec

def vectorize_sparse(df, args):
    #initialize cache
    cache = []
    
    #count tags in tagged df
    for index, row in df.iterrows():
        num_sentences = row['num_sentences']
        vector =  [0, 0, 0, 0, 0, 0, 0]
        dummy_vec = ['group', 'cta', 'problem', '1psg', 'discourse', 'sentencemod', 'sentiment']
        if 'group' in args.features or 'all' in args.features:
            vector[0] = default_normalize(len(re.findall(r'(\+GRUPO)', str(row['tagged_content']), re.IGNORECASE)), num_sentences)
        if 'cta' in args.features or 'all' in args.features:
            vector[1] = default_normalize(len(re.findall(r'(\+ACCION', str(row['tagged_content']), re.IGNORECASE)), num_sentences)
        if 'problem' in args.features or 'all' in args.features:
            vector[2] = default_normalize(len(re.findall(r'(\+PROBLEMA)', str(row['tagged_content']), re.IGNORECASE)), num_sentences)
        if '1psg' in args.features or 'all' in args.features:
            vector[3] = default_normalize(len(re.findall(r'(\+1SG)', str(row['tagged_content']), re.IGNORECASE)), num_sentences)
        if 'discourse' in args.features or 'all' in args.features:
            vector[4] = default_normalize(len(re.findall(r'(\+DISCURSO)', str(row['tagged_content']), re.IGNORECASE)), num_sentences)
        if 'sentencemod' in args.features or 'all' in args.features:
            vector[5] = default_normalize(len(re.findall(r'(\+MODIFICADOR)', str(row['tagged_content']), re.IGNORECASE)), num_sentences)
        if 'sentiment' in args.features or 'all' in args.features:
            vector[6] = row['sentiment']
        
        grievance = row['grievance']
        #print(vector)

        # Append the vector and grievance to the DataFrame
        cache.append({'grievance': grievance, 'vector': vector})

    # Convert the cache to a DataFrame
    vectorized_df = pd.DataFrame(cache)

    if args.scaler:
        vectorized_df = BasicImportance.scale_x(vectorized_df, 'vector')
    
    return vectorized_df, dummy_vec