### Import Settings ###
from settings import BASE_DIR, TIME, TOKEN

### Import Dependencies ###
import os
import sys
import argparse
import time
import gc

import pandas as pd

import stanza

### Import Functions ###
from codebase.tools import read_from_json_file, str2bool, multiselect, get_corpus_info, tag_mode_selector
from codebase.sanitizers import sanitize_df
from codebase.multi_proc import run_in_parallel
from codebase.tokenize import process_text
from codebase.tagger import tagger

### Import Models ###
from codebase.log_reg_main import log_reg_main
from codebase.gzip_main import gzip_main
from codebase.spanberta_main import spanberta_main

def main(args):
    start_time = time.time()

    print('Starting the pipeline...')
    print(f'Running the pipeline at {TIME}\nwith token: {TOKEN}...')
    print('___________________________\n')
    
    print('Checking Python version...')
    print('Python version:', sys.version)
    if sys.version_info < (3, 10):
        raise Exception("Please use Python version 3.10 or higher.")
    
    print('\n___________________________\n')

    print(f'Creating output directory at {BASE_DIR / args.output}...')
    output_dir = BASE_DIR / args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f'Creating data directory at {BASE_DIR / args.data}...')
    data_dir = BASE_DIR / args.data
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    print('Reading wordlists...')
    files = os.listdir(BASE_DIR / args.wordlists)
    wordlists = {}
    for file in files:
        if file.endswith('.json'):
            variable_name = file.split('.')[0]
            wordlists[variable_name] = read_from_json_file(BASE_DIR / args.wordlists / file)

    print('\n___________________________\n')

    #set pandas options
    #pd.set_option("display.max_colwidth", None)

    #load data and preprocess
    print("Loading and preprocessing data...")
    if args.tokenize == False and os.path.exists(BASE_DIR / data_dir / "df.pkl") == True:
        print("Tokenization skipped. Using existing data...")
        df = pd.read_pickle(BASE_DIR / data_dir / "df.pkl")
        print("Dataframe in use: \n")
        print(df.head())
    
    else:
        print("Reading data...")
        df = pd.read_excel(BASE_DIR / args.input / args.corpus)

        print("Sanitizing data...")
        df = sanitize_df(df)

        print("Dataframe in use: \n")
        print(df.head())

        print("Tokenizing data...")

        ### Models ###
        #check if the models folder is present
        if os.path.exists(BASE_DIR / "models") == False:
            print("No models folder found. Creating folder...")
            os.makedirs(BASE_DIR / "models")

        # Download the Spanish model
        print("Downloading the Spanish model...")
        stanza.download("es", model_dir=str(BASE_DIR / "models"))
        
        #tokenize with the Spanish stanza model
        print("Processing text...")
        results = run_in_parallel(process_text, df['content'], args.threads, use_thread=False)

        #unpack results
        print("Unpacking results...")
        tokenized_content, sentiment, num_sentences, total_word_count, word_count_per_sentence = zip(*results)
        df['tokenized_content'] = tokenized_content
        df['sentiment'] = sentiment
        df['num_sentences'] = num_sentences
        df['total_word_count'] = total_word_count
        df['word_count_per_sentence'] = word_count_per_sentence

        print("Saving data...")
        df.to_pickle(BASE_DIR / data_dir / "df.pkl")

    print('\n___________________________\n')

    #tag data
    if args.features == False:
        print("Features skipped. Nothing to do...")
    else:
        print("Tagging data...")
        df = tagger(df, wordlists, args)

    print('\n___________________________\n')

    #use tag mode selector
    if tag_mode_selector(args) == 0:
        print("Using lower tag mode...")
        df['tagged_content'] = df['tagged_content'].str.lower()
    elif tag_mode_selector(args) == 1:
        print("Using upper tag mode...")
        print("Nothing to do...")
    elif tag_mode_selector(args) == 2:
        print("Using value tag mode...")
        print("Tags will be in UPPERCASE mode. Sentiment will be in value mode.")
    else:
        raise Exception("No tag mode selected or tag_mode_selector returned wrong value. Exiting...")

    #save data to excel
    filename = f'{TIME}-{TOKEN}-checkpoint-pre-classify.xlsx'
    print(f'Saving preprocessed data to {BASE_DIR / data_dir / filename}...')
    df.to_excel(BASE_DIR / data_dir / filename, index=False)

    print('\n___________________________\n')

    # Now you can print or use these values as needed
    total_sentences_sum, total_word_count, mean_word_count_per_sentence, min_word_count_per_sentence, max_word_count_per_sentence = get_corpus_info(df)
    print("Total sentences sum:", total_sentences_sum)
    print("Total word count:", total_word_count)
    print("Mean word count per sentence:", mean_word_count_per_sentence)
    print("Min word count per sentence:", min_word_count_per_sentence)
    print("Max word count per sentence:", max_word_count_per_sentence)

    print('\n___________________________\n')

    if args.logreg == False and args.gzip == False and args.bert == False:
        print("No classification approach selected. Exiting...")
        sys.exit()

    print("Starting classification...")
    if args.logreg == True:
        log_reg_main(df, args)
    
    if args.gzip == True:
        gzip_main(df, args)

    if args.bert == True:
        spanberta_main(df, args)

    print('\n___________________________\n')

    print("Pipeline completed. Thank you for trusting us with your data! :)")
    print(f'Output stored in {BASE_DIR / args.output} with token: {TOKEN}.')
    print(f'Data and checkpoints stored in {BASE_DIR / args.data}.')

    print('\n___________________________\n')

    end_time = time.time()
    time_taken = end_time - start_time
    hours, remainder = divmod(time_taken, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f'Speedup was: {int(hours)}:{int(minutes)}:{seconds:.2f} (Hours:Minutes:Seconds).')

    print('\n___________________________\n')

    print('Cleaning up...')
    gc.collect()

    print('Goodbye!')

    sys.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This is a pipeline to classify grievances using different models', formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument_group('Folders and Data:')
    parser.add_argument('--input', type=str, default='input', help='The folder containing the input data. Default is input.')
    parser.add_argument('--corpus', type=str, default='corpus.xlsx', help='The name of the corpus file. Default is corpus.xlsx.')
    parser.add_argument('--wordlists', type=str, default='wordlists', help='The folder containing the wordlists. Default is wordlists.')
    parser.add_argument('--data', type=str, default='data', help='The folder containing the pickled or intermediate data. Default is data.')
    parser.add_argument('--output', type=str, default=f'OUTPUT-{TIME}-{TOKEN}', help='The folder to store the output in. Default is output.')

    parser.add_argument_group('Preprocessing:')
    parser.add_argument('--tokenize', type=str2bool, default=True, help='A boolean value indicating whether to tokenize the text. Default is True. Otherwise skip tokenization and use existing data.')
    parser.add_argument('--features', type=multiselect, default='all', help='''A string/boolean value indicating whether to tag the text. Default is all. 
Allowed values are "true", "false", "none", "all", "null" and "" or one or more features separated by commas.
The following features are available:
    - "group" to tag deprived groups 
    - "cta" to tag calls to action 
    - "problem" to tag problem frames
    - "1psg" to tag first person singular pronouns
    - "discourse" to tag discourse markers
    - "sentencemod" to tag sentence modifiers
    - "sentiment" to tag sentiment
    - "all" to tag all of the above 
    - "null" to tag nothing 
Example: --features group,cta,pronouns''')
    parser.add_argument('--tag_mode', type=str, default='upper', help='''
                        The sentiment mode to use. Default is upper.
                        Allowed values are "upper": +NEGATIVE, 
                        "lower": +negative
                        and "value": +0.1352376
                        ''')
    
    parser.add_argument_group('Performance')
    parser.add_argument('--threads', type=int, default=8, help='The number of threads to use for parallel processing. Default is 8.')

    parser.add_argument_group('Classification approach:')
    parser.add_argument('--resample', type=str2bool, default=False, help='A boolean value indicating whether to resample the data. Default is False.')
    parser.add_argument('--random_state', type=int, default=42, help='The random state to use for all occasions. Default is 42.')
    parser.add_argument('--kfold', type=str2bool, default=True, help='A boolean value indicating whether to use stratified k-fold cross validation. Default is True. Otherwise use a simple train-test split.')
    parser.add_argument('--logreg', type=str2bool, default=False, help='A boolean value indicating whether to use logistic regression. Default is False.')
    parser.add_argument('--gzip', type=str2bool, default=False, help='A boolean value indicating whether to use gzip. Default is False.')
    parser.add_argument('--bert', type=str2bool, default=False, help='A boolean value indicating whether to use BERT. Default is False.')
    parser.add_argument('--hybrid', type=str2bool, default=False, help='''
                        A boolean value indicating whether to use a hybrid mode. Default is False.
                        The hybrid approach tags the rule-based features in the text and uses either gzip or BERT for classification.''')
    
    parser.add_argument_group('Stratified k-fold cross validation parameters:')
    parser.add_argument('--folds', type=int, default=5, help='The number of folds. Default is 5.')

    parser.add_argument_group('Splitting parameters (for simple train-test splitting only!):')
    parser.add_argument('--min_grievances', type=int, default=20, help='The minimum number of grievances to include in the training set. Default is 20. Has no effect if "resampling" is set to True.')
    parser.add_argument('--train_size', type=float, default=0.7, help='A number between 0 and 1 that indicates the size of the training set. Default is 0.7.')

    parser.add_argument_group('Logistic regression parameters:')
    parser.add_argument('--iterations', type=int, default=100, help='The maximum number of iterations to use for training the logistic regression model. Default is 100.')
    parser.add_argument('--vector_dense', type=str2bool, default=True, help='A boolean defining if the vector for logreg should be dense (True) or sparse (False). Default is True.')
    parser.add_argument('--scaler', type=str2bool, default=True, help='A boolean defining if the vectorized data should be scaled. Default is True. Hint: Scaling is crucial for calculating the feature importance!')

    parser.add_argument_group('GZIP parameters:')
    parser.add_argument('--k', type=int, default=3, help='The number of nearest neighbors to use for classification. Default is 3.')

    parser.add_argument_group('BERT parameters:')
    parser.add_argument('--batch_size', type=int, default=8, help='The batch size to use for training and evaluation. Default is 8.')
    parser.add_argument('--epoch', type=int, default=3, help='The number of epochs to use for training. Default is 3.')
    parser.add_argument('--lr', type=float, default=1e-5, help='The learning rate to use for training. Default is 1e-5.')
    parser.add_argument('--print_model', type=str2bool, default=False, help='A boolean value indicating whether to print the model. Default is False.')

    parser.add_argument_group('Statistics:')
    parser.add_argument('--count_features', type=str2bool, default=False, help='A boolean value indicating whether to count the tags in the tagged dataframe. Default is True.')
    parser.add_argument('--feature_normal', type=str2bool, default=False, help='A boolean value indicating whether to normalize the features. Default is False.')

    parser.add_argument_group('Save Modes:')
    parser.add_argument('--save_dataframe', type=str2bool, default=True, help='A boolean value indicating whether to save the dataframe. Default is True.')
    parser.add_argument('--save_results', type=str2bool, default=True, help='A boolean value indicating whether to save the results. Default is True.')
    parser.add_argument('--save_callbacks', type=str2bool, default=True, help='A boolean value indicating whether to save the callbacks. Default is True.')

    args = parser.parse_args()
    main(args)