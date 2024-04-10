### Import Settings ###
from settings import BASE_DIR

### Import Dependencies ###
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


### Import Functions ###
from codebase.vectorize import vectorize_dense, vectorize_sparse
from codebase.split import custom_split, kfold_validation
from codebase.evaluation import evaluate_results
from codebase.feature_importance import BasicImportance

### CODE ###
def linear_logistic_classify(X_train, y_train, X_test, y_test, args):

    context = {}

    # Convert the lists to numpy arrays
    X_train = X_train.tolist()
    X_test = X_test.tolist()

    # Train the model
    ##penalty: 'l1' or 'l2', default='l2'
    ##solver: 'liblinear for using both penalties and small datasets, lbfgs for 'l2' penalty
    model = LogisticRegression(penalty = 'l2', solver='liblinear', random_state=args.random_state, max_iter=args.iterations).fit(X_train, y_train)
    # Predict the test set
    y_pred = model.predict(X_test)

    ##get the probabilities of the test set
    y_pred_proba = ["; ".join(f"{label}: {prob}" for label, prob in enumerate(probs)) for probs in model.predict_proba(X_test)]  

    # append probabilities to context
    context['y_proba'] = y_pred_proba

    #get the loss
    eval_loss = log_loss(y_test, y_pred)
    context['eval_loss'] = eval_loss
    
    return model, y_test, y_pred, context #None can be context, e.g., dictionary


def log_reg_main(df, args):
    if args.kfold == True:
        print("Vectorizing data...")
        if args.vector_dense == True:
            df_v, dummy_vec = vectorize_dense(df, args)
        else:
            df_v, dummy_vec = vectorize_sparse(df, args)

        X = 'vector'
        y = 'grievance'

        model, y_test, y_pred, y_df, context_dict = kfold_validation(df_v, X, y, linear_logistic_classify, args)

        df = pd.concat([df, y_df], axis=1)

        context_dict['feature_importance'] = BasicImportance.feature_importance(model, dummy_vec, args)

        evaluate_results(df, y_test, y_pred, "Logistic Regression K-Fold", context_dict, args)

    else:
        print("Splitting data...")
        train_df, test_df = custom_split(df, args)

        if args.vector_dense == True:
            train_df_v, dummy_vec = vectorize_dense(train_df, args)
            test_df_v, dummy_vec = vectorize_dense(test_df, args)
        else:
            train_df_v, dummy_vec = vectorize_sparse(train_df, args)
            test_df_v, dummy_vec = vectorize_sparse(test_df, args)

        X_train = train_df_v['vector']
        y_train = train_df_v['grievance']
        X_test = test_df_v['vector']
        y_test = test_df_v['grievance']

        print("Training model...")
        model, y_test, y_pred, context = linear_logistic_classify(X_train, y_train, X_test, y_test, args)

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

        context_dict['feature_importance'] = BasicImportance.feature_importance(model, dummy_vec, args)

        evaluate_results(test_df, y_test, y_pred, "Logistic Regression Single", context_dict, args)

    return