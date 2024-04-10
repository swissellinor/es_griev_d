### Importing necessary libraries
import os
import datetime
import numpy as np
import pandas as pd
import torch
import gc
import lime.lime_text
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from transformers import RobertaTokenizerFast

from sklearn.preprocessing import StandardScaler

### Import Settings ###
from settings import BASE_DIR

class LimeImportance:
    # Specific function to predict roberta model for lime explainer
    @staticmethod
    def spanberta_predictor(text, model):
        def tokenizer(text, padding=True, truncation=True, return_tensors="pt"):
            tokenizer = RobertaTokenizerFast.from_pretrained("skimai/spanberta-base-cased", max_length = 512, cache_dir = str(BASE_DIR / "models"))
            encoding = tokenizer(text, padding=padding, truncation=truncation, return_tensors=return_tensors)
            return encoding["input_ids"], encoding["attention_mask"]
        input_ids, attention_mask = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return torch.softmax(outputs.logits, dim=1).detach().numpy()

    @staticmethod
    def feature_importance(model, X_test, args):
        X_test = X_test.reset_index(drop=True)
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        filename = f'{now}-LIME-Feature-Importance.pdf'
        explainer = lime.lime_text.LimeTextExplainer(class_names=['non-grievance', 'grievance'])
        with PdfPages(os.path.join(BASE_DIR, args.output, filename)) as pdf:
            for idx, content in X_test.items():
                print(f'Calculating feature importance for sample {idx}...')
                print(f'Content: {content}')
                exp = explainer.explain_instance(content, lambda x: spanberta_predictor(x, model), num_features=8, num_samples=500)
                print('Creating figure...')
                fig = exp.as_pyplot_figure(label=1)
                print('Saving figure...')
                pdf.savefig(fig)
                plt.close(fig)
                print('Figure saved.')
                print('Running garbage collection...')
                gc.collect()
                print('Garbage collection complete.')
                print('Done.')
                print('______________________________')
                print('\n')

    

class BasicImportance:

    ### Scaling the X values ###
    # input: df (DataFrame), column (string)
    # output: df (DataFrame)
    @staticmethod
    def scale_x(df, column):
        print('Initializing StandardScaler...')
        scaler = StandardScaler()

        print('Scaling...')
        vectors = df[column].to_list()
        scaler.fit(vectors)
        df[column] = df[column].apply(lambda x: scaler.transform([x])[0])

        return df

    ### Feature importance calculation ###
    # input: models (list of models), features (list of features), args (arguments)
    # output: stats (dictionary of mean and standard deviation of coefficients)
    # output format: {feature: [mean, std]}
    @staticmethod
    def feature_importance(models, features, args):
        print('Calculating feature importance...')

        # Check if models is a list
        if not isinstance(models, list):
            models = [models]

        # Initialize an empty DataFrame with the desired columns
        df = pd.DataFrame(columns=features)

        for model in models:
            # Create a new row for this model
            row = {}
            for idy, coef in enumerate(model.coef_[0]):
                # Add the coefficient to the corresponding feature
                row[features[idy]] = coef

            # Append the new row to the DataFrame
            df = pd.concat([df, pd.DataFrame(row, index=[0])])

        # Calculate the mean of the coefficients
        stats = {}
        if len(models) > 1:
            for feature in df.columns:
                mean = df[feature].mean()
                std = df[feature].std()

                stats[feature] = [mean, std]
        elif len(models) == 1:
            for feature in df.columns:
                stats[feature] = [df[feature].mean(), 0]
        else:
            raise ValueError('No models found.')
        
        print('______________________________')
        print ('Feature importance calculated:')
        print(stats)
        print('______________________________')

        return stats

        
        
            