### Description ###
# This file contains utility functions that are used in the main script.

### Dependencies ###
from statistics import mean, median
import json
import argparse

# Check if input is a punctuation
def is_punctuation(i):
    return i in {'.', ',', '?', '!'}


#turn input string with variable delimiter into dictionary 
def string_to_dict(input_string, delim):
    # Split the string into key-value pairs
    try:
        pairs = input_string.split(delim)
    
        # Create a dictionary to store the key-value pairs
        result_dict = {}
    
        # Iterate over each pair and split it into key and value
        for pair in pairs:
            key, value = pair.split('=')
            result_dict[key] = value

    except:
        key, value = input_string.split('=')
        result_dict[key] = value

    return result_dict


# Write data to a JSON file
def write_to_json_file(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


# Read data from a JSON file
def read_from_json_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def str2bool(i):
    if isinstance(i, bool):
       return i
    if i.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif i.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Convert a string to a boolean value
def multiselect(i):
    if i.lower() in ('yes', 'true', 'all', '1'):
        return ['all']
    elif i.lower() in ('no', 'false', 'none', 'null', '0'):
        return False
    elif ',' in i:
        selection = i.split(',')
        if not isinstance(selection, list):
            return [selection]
        else:
            return selection
    else:
        return str(i)
    
# Normalize a number by dividing it by another number
def default_normalize(num, denom): 
    if num >= 0 and denom > 0: 
        return (num/denom) #normalized count
    elif num < 0: 
        raise TypeError('Input cannot be negative') 
    elif denom <= 0: 
        raise TypeError('Count of Sentences cannot be zero or negative')
    else: 
        return 
    
# Check if something is a list of lists
def check_lioli(i):
    if isinstance(i, list):
        if any(isinstance(j, list) for j in i):
            return True
        else:
            return False
    else:
        return None
    
def get_corpus_info(df):
    total_sentences_sum = 0
    total_word_count = 0
    word_count_per_sentence_stats = []

    for index, row in df.iterrows():
        num_sentences = row['num_sentences']
        total_word_count_item = row['total_word_count']
        word_count_per_sentence_list = row['word_count_per_sentence']

        total_sentences_sum += num_sentences
        total_word_count += total_word_count_item
        word_count_per_sentence_stats.extend(word_count_per_sentence_list)

    # Calculate mean, min, and max of word count per sentence across the entire DataFrame
    if word_count_per_sentence_stats:
        mean_word_count_per_sentence = mean(word_count_per_sentence_stats)
        min_word_count_per_sentence = min(word_count_per_sentence_stats)
        max_word_count_per_sentence = max(word_count_per_sentence_stats)
    else:
        mean_word_count_per_sentence = min_word_count_per_sentence = max_word_count_per_sentence = None

    return total_sentences_sum, total_word_count, mean_word_count_per_sentence, min_word_count_per_sentence, max_word_count_per_sentence


def tag_mode_selector(args):
    if 'lower' in args.tag_mode:
        mode_selector = 0
    elif 'upper' in args.tag_mode:
        mode_selector = 1
    elif 'value' in args.tag_mode:
        mode_selector = 2
    else:
        raise ValueError('Tag mode not recognized')
    return mode_selector