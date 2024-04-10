### Import Settings ###
from settings import BASE_DIR

### Import Dependencies ###
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import resample

### Splitting ###
def upsampling(df, args):

    if 'grievance' in df.columns:
        df_grievances = df[df['grievance'] == 1]
        df_non_grievances = df[df['grievance'] == 0]
    elif 'labels' in df.columns:
        df_grievances = df[df['labels'] == 1]
        df_non_grievances = df[df['labels'] == 0]

    else:
        raise ValueError('df is not containing a grievance or labels column, could not upsample.')

    # Upsample grievances in the training set if they are fewer than non-grievances
    if len(df_grievances) < len(df_non_grievances):
        df_grievances = resample(df_grievances, replace=True, n_samples=len(df_non_grievances), random_state=args.random_state)

    df = pd.concat([df_grievances, df_non_grievances])

    df.reset_index(drop=True, inplace=True)

    return df

def custom_split(df, args):
    # Resampling grievences
    if args.resample == True:
        print("Splitting data with a train size of {}...".format(args.train_size))

        # Split the dataframe with grievances into training and test sets
        train_df, test_df = train_test_split(df, train_size=args.train_size, random_state=args.random_state)

        train_df = upsampling(train_df, args)

        print("Produced {} training samples and {} test samples.".format(len(train_df), len(test_df)))
        print("The training set contains {} grievances and {} non-grievances.".format(len(train_df[train_df['grievance'] == 1]), len(train_df[train_df['grievance'] == 0])))
    
    # Using minimum number of grievances
    else:
        print("Splitting data with a train size of {} and a minimum of {} grievances...".format(args.train_size, args.min_grievances))
        # Separate the dataframe into two: one with grievances and one without
        df_grievances = df[df['grievance'] == 1]
        df_non_grievances = df[df['grievance'] == 0]

        # Calculate the train size for the dataframe with grievances
        train_size_grievances = max(args.min_grievances / len(df_grievances), args.train_size) if len(df_grievances) > args.min_grievances else args.train_size
        if len(df_grievances) < args.min_grievances:
            print("The number of grievances ({}) is less than the minimum number of grievances ({}).".format(len(df_grievances), args.min_grievances))
            print("The train size for the dataframe with grievances is set to {}.".format(train_size_grievances))

        # Split the dataframe with grievances into training and test sets
        train_grievances, test_grievances = train_test_split(df_grievances, train_size=train_size_grievances, random_state=args.random_state)

        # Split the dataframe without grievances into training and test sets
        train_non_grievances, test_non_grievances = train_test_split(df_non_grievances, train_size=args.train_size, random_state=args.random_state)

        # Concatenate the training sets and test sets to get the final training and test sets
        train_df = pd.concat([train_grievances, train_non_grievances])
        test_df = pd.concat([test_grievances, test_non_grievances])

        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        print("Produced {} training samples and {} test samples.".format(len(train_df), len(test_df)))
        print("The training set contains {} grievances and {} non-grievances.".format(len(train_df[train_df['grievance'] == 1]), len(train_df[train_df['grievance'] == 0])))

    return train_df, test_df


#function that takes dataframe and applies stratified k-fold cross validation
#input: df (dataframe) x = content, y= labels (strings), train_pred_function (function) and arguments
#return: model (logreg, compression, spanberta), lists with actual label, predicted label (list), context_dict (None for logreg and gzip, Loss for Spanberta)
def kfold_validation(df, X, y, train_pred_function, args):
    # Create a StratifiedKFold object
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.random_state)

    y_pred_list = []
    y_test_list = []
    models = []
    context_dict = None

    #skf split needs actual numpy arrays with content
    #gives back indices
    X_pseudo = df.drop(columns=[y])
    y_pseudo = df[y]

    y_df = pd.DataFrame(index=df.index, columns=['y_test', 'y_pred'])

    #take df with rows that skf.split provided
    for train_index, test_index in skf.split(X_pseudo, y_pseudo):
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]

        #upsample train-df
        if args.resample == True:
            print("Resampling the training set...")
            train_df = upsampling(train_df, args)

        #take train_df[content] and map it to x_train arrays
        X_train, X_test = train_df[X], test_df[X]
        y_train, y_test = train_df[y], test_df[y]
        
        # Train the model using the provided function
        model, y_test, y_pred, context = train_pred_function(X_train, y_train, X_test, y_test, args)

        # Add these series to the DataFrame
        y_df.loc[test_index, 'y_test'] = pd.Series(y_test, index=test_index)
        y_df.loc[test_index, 'y_pred'] = pd.Series(y_pred, index=test_index)

        # Store the results
        y_test_list.append(y_test)
        y_pred_list.append(y_pred)
        models.append(model)

        if context is not None:

            if 'y_proba' in context.keys():
                y_df.loc[test_index, 'y_proba'] = pd.Series(context['y_proba'], index=test_index)

            if context_dict is None:
                context_dict = {key: [] for key in context.keys()}
                
            for key in context.keys():
                context_dict[key].append(context[key])
        else:
            context_dict = None

    return models, y_test_list, y_pred_list, y_df, context_dict