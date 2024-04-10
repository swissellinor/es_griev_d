### Import Settings ###
from settings import BASE_DIR

### Import Dependencies ###
import gzip

import pandas as pd
import numpy as np

### Import Functions ###
from codebase.split import custom_split, kfold_validation
from codebase.evaluation import evaluate_results

### CODE ###
def npc_gzip(X_train, y_train, X_test, y_test, args):
    # Convert the dataframe to a list
    if isinstance(X_train, pd.DataFrame):
        X_train_list = X_train.iloc[:, 0].tolist()
    else:  # X_train is a Series
        X_train_list = X_train.tolist()

    if isinstance(y_train, pd.DataFrame):
        y_train_list = y_train.iloc[:, 0].tolist()
    else:  # y_train is a Series
        y_train_list = y_train.tolist()

    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.iloc[:, 0].tolist()
    else:  # X_test is a Series
        X_test = X_test.tolist()

    # Check if y_train_list and X_train_list have the same length
    if len(y_train_list) != len(X_train_list):
        raise ValueError("y_train_list and X_train_list must have the same length.")
    else:
        # Combine the lists into a list of tuples
        training_set = list(zip(y_train_list, X_train_list))

    predict_list = []
    for x1 in X_test:
        Cx1 = len(gzip.compress(x1.encode())
        )
        distance_from_x1 = []
        for (_, x2) in training_set:
            Cx2 = len(gzip.compress(x2. encode()))
            x1x2 = " ".join([x1, x2])
            Cx1x2 = len(gzip.compress(x1x2. encode()))
            ncd = (Cx1x2 - min(Cx1,Cx2)) / max(Cx1, Cx2)
            distance_from_x1.append(ncd) 
        sorted_idx = np.argsort(np.array(distance_from_x1))
        top_k_class = [training_set[i][0] for i in sorted_idx[:args.k]]
        predict_class = max(set(top_k_class), key=top_k_class.count)
        predict_list.append(predict_class)

    # Turn predict list into np array
    y_pred = np.array(predict_list)

    # Check if y_test and y_pred have the same length
    if len(y_test) != len(y_pred):
        raise ValueError("y_test and y_pred must have the same length.")
    else:
        return None, y_test, y_pred, None

def gzip_main(df, args):
    print("Using GZIP model...")
    if args.hybrid == True:
        print("Using hybrid model...")
        column_selector = 'tagged_content'
        model_name = "GZIP Hybrid"
    else:
        print("Using content only...")
        column_selector = 'content'
        model_name = "GZIP"

    if args.kfold == True:
        print("Using K-Fold validation...")
        X = column_selector
        y = 'grievance'

        model, y_test, y_pred, y_df, context_dict = kfold_validation(df, X, y, npc_gzip, args)

        df = pd.concat([df, y_df], axis=1)

        evaluate_results(df, y_test, y_pred, f'{model_name} K-Fold', context_dict, args)

    else:
        print("Splitting data...")
        train_df, test_df = custom_split(df, args)

        X_train = train_df[column_selector]
        y_train = train_df['grievance']
        X_test = test_df[column_selector]
        y_test = test_df['grievance']

        print("Training model...")
        model, y_test, y_pred, context = npc_gzip(X_train, y_train, X_test, y_test, args)

        if context is not None:
            context_dict = {key: [] for key in context.keys()}
            for key in context.keys():
                context_dict[key].append(context[key])
        else:
            context_dict = None

        test_df['y_test'] = y_test
        test_df['y_pred'] = y_pred

        evaluate_results(test_df, y_test, y_pred, f'{model_name} Single', context_dict, args)

    return