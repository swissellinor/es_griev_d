### Import Settings ###
from settings import BASE_DIR

### Import Dependencies ###
import pandas as pd

### Import Functions ###
from codebase.spanberta_codebase import spanberta_tokenize, spanberta_load, spanberta_train, spanberta_predictor_classify
from codebase.split import custom_split, kfold_validation
from codebase.evaluation import evaluate_results

### CODE ###
def spanberta_train_predict(X_train, y_train, X_test, y_test, args):

    context = {}
	
    # Convert X_train and y_train to DataFrames
    X_train_df = pd.DataFrame(X_train, columns=['content'])
    y_train_df = pd.DataFrame(y_train, columns=['labels'])

    # Concatenate X_train_df and y_train_df into a single DataFrame
    train_set = pd.concat([X_train_df, y_train_df], axis=1)

    # Convert X_test and y_test to DataFrames
    X_test_df = pd.DataFrame(X_test, columns=['content'])
    y_test_df = pd.DataFrame(y_test, columns=['labels'])

    # Concatenate X_test_df and y_test_df into a single DataFrame
    test_set = pd.concat([X_test_df, y_test_df], axis=1)

    train_set, test_set = spanberta_tokenize(train_set, test_set)
	
    # Load the model
    model = spanberta_load(args)
	
    # Train the model
    trainer, train_loss, eval_loss = spanberta_train(train_set, test_set, model, args)

    context['train_loss'] = train_loss
    context['eval_loss'] = eval_loss
	
    # Predict label X_test with the model
    y_pred = spanberta_predictor_classify(trainer, test_set)

    return model, y_test, y_pred, context

def spanberta_main(df, args):
    print("Using SPANBERTA model...")
    if args.hybrid == True:
        print("Using hybrid model...")
        model_name = "SPANBERTA Hybrid"
    else:
        print("Using content only...")
        model_name = "SPANBERTA"

    if args.kfold == True:
        print("Using K-Fold validation...")
		
        if args.hybrid:
            df.drop(columns = ['tagged_content'], axis=1, inplace = True)
            df.rename(columns = {'tagged_content': 'content'}, inplace = True)
		
        df.rename(columns = {'grievance': 'labels'}, inplace = True)
        X = 'content'
        y = 'labels'

        model, y_test, y_pred, y_df, context_dict = kfold_validation(df, X, y, spanberta_train_predict, args)

        df = pd.concat([df, y_df], axis=1)
        
        evaluate_results(df, y_test, y_pred, f'{model_name} K-Fold', context_dict,  args)
		
    else:
        print("Splitting data...")
        train_df, test_df = custom_split(df, args)
		
        if args.hybrid:
            train_df['content'] = train_df['tagged_content']
            test_df['content'] = test_df['tagged_content']

        train_df.rename(columns = {'grievance': 'labels'}, inplace = True)
        test_df.rename(columns = {'grievance': 'labels'}, inplace = True)
        X_train = train_df['content']
        y_train = train_df['labels']
        X_test = test_df['content']
        y_test = test_df['labels']

        print("Training model...")
        model, y_test, y_pred, context = spanberta_train_predict(X_train, y_train, X_test, y_test, args)

        if context is not None:
            context_dict = {key: [] for key in context.keys()}
            for key in context.keys():
                context_dict[key].append(context[key])
        else:
            context_dict = None

        test_df['y_test'] = y_test
        test_df['y_pred'] = y_pred

        if 'y_proba' in context_dict.keys():
            test_df['y_proba'] = pd.Series(context_dict['y_proba'])

        evaluate_results(test_df, y_test, y_pred, f'{model_name} Single', context_dict, args)

    return