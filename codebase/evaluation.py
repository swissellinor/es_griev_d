### Import Settings ###
from settings import BASE_DIR, TIME, TOKEN

### Import Dependencies ###
import os

import numpy as np
import pandas as pd

import re

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import textwrap

from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

### Import Functions ###
from codebase.tools import default_normalize

def evaluate_results(master_df, y_test, y_pred, model_name, context_dict, args):

    # Create the output filename
    features = '-'.join([f.replace(' ', '_') for f in args.features])
    filename_pdf = f'{TIME}-{TOKEN}-{model_name}-{features}.pdf'
    filename_xlsx = f'{TIME}-{TOKEN}-{model_name}-{features}.xlsx'

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.join(BASE_DIR, args.output), exist_ok=True)

    if args.kfold == True:
        print('y_test and y_pred are both lists of lists. Iterating through and taking mean...')
        
        # Initialize a dictionary to store metrics for each fold
        metrics_dict = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'cm': [], 'classification_rep': []}

        # Calculate metrics for each fold
        for y_true_fold, y_pred_fold in zip(y_test, y_pred):
            metrics_dict['accuracy'].append(metrics.accuracy_score(y_true_fold, y_pred_fold))
            metrics_dict['precision'].append(metrics.precision_score(y_true_fold, y_pred_fold, zero_division=0))
            metrics_dict['recall'].append(metrics.recall_score(y_true_fold, y_pred_fold))
            metrics_dict['f1'].append(metrics.f1_score(y_true_fold, y_pred_fold))
            metrics_dict['cm'].append(metrics.confusion_matrix(y_true_fold, y_pred_fold))
            metrics_dict['classification_rep'].append(classification_report(y_true_fold, y_pred_fold, output_dict=True))

        # Calculate mean values
        mean_metrics = {metric: np.mean(values) for metric, values in metrics_dict.items() if metric != 'cm' and metric != 'classification_rep'}
        std_metrics = {metric: np.std(values) for metric, values in metrics_dict.items() if metric != 'cm' and metric != 'classification_rep'}

        # Find index of best and worst fold
        best_fold_idx = np.argmax(metrics_dict['accuracy'])
        worst_fold_idx = np.argmin(metrics_dict['accuracy'])

        # Print mean values
        print('\n')
        print('______________________________')
        print('Mean Metrics:')
        for metric, mean_value in mean_metrics.items():
            print(f'{metric.capitalize()}: {mean_value} ± {std_metrics[metric]}')
        print('______________________________')
        print('\n')

        # Calculate mean confusion matrix
        mean_cm = np.mean(metrics_dict['cm'], axis=0)

        # Create a figure and axes object for plotting
        fig, axs = plt.subplots(4, 3, figsize=(18, 20))

        # # Add a title to the plot
        fig.suptitle(f'{model_name} - {TIME} - {TOKEN}')

        plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95, wspace=0.2, hspace=0.5)

        # Plot the confusion matrix for the best and worst fold, and the mean confusion matrix
        for idx, title, ax in zip([best_fold_idx, worst_fold_idx, None], ['Best', 'Worst', 'Mean'], axs.flatten()[:3]):
            cm = metrics_dict['cm'][idx] if idx is not None else mean_cm
            ConfusionMatrixDisplay(cm, display_labels=['non-grievance', 'grievance']).plot(ax=ax)
            ax.set_title(f'Confusion Matrix for {title} Fold')

        # Combine all metrics boxplots into one larger boxplot
        metrics_data = [metrics_dict[metric] for metric in ['accuracy', 'precision', 'recall', 'f1']]
        labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        sns.boxplot(data=metrics_data, ax=axs[1, 0], palette = "tab20c")
        sns.despine()
        axs[1, 0].set_title('Metrics Distribution Across Folds')

        # Set x-ticks and add labels under x-axis
        axs[1, 0].set_xticks(range(len(labels)))
        axs[1, 0].set_xticklabels(labels, rotation=45, ha='right')

        # Create a graph for the training loss
        if context_dict is not None and 'train_loss' in context_dict:
            for idx, loss_fn in enumerate(context_dict['train_loss']):
                axs[1, 1].plot(loss_fn, label=f'Fold {idx+1}')
            axs[1, 1].set_title('Training Loss')
            axs[1, 1].set_xlabel('Epoch')
            axs[1, 1].set_ylabel('Loss')

        # Create a graph for the evaluation loss
        if context_dict is not None and 'eval_loss' in context_dict:
            for idx, loss_fn in enumerate(context_dict['eval_loss']):
                axs[1, 2].plot(loss_fn, label=f'Fold {idx+1}')
            axs[1, 2].set_title('Evaluation Loss')
            axs[1, 2].set_xlabel('Epoch')
            axs[1, 2].set_ylabel('Loss')

        # Create a table for mean metrics with standard deviations
        mean_std_df = pd.DataFrame({'Mean': mean_metrics, 'Standard Deviation': std_metrics}).round(3)
        table1 = axs[2, 0].table(cellText=mean_std_df.values,
                                colLabels=mean_std_df.columns,
                                rowLabels=mean_std_df.index,
                                loc='center')
        table1.auto_set_font_size(False)
        table1.set_fontsize(10)

        # Add a title to the table
        axs[2, 0].set_title('Mean Metrics with Standard Deviations')

        # Hide axes
        axs[2, 0].axis('off')

        # Concatenate all the true and predicted values from each fold
        y_true_all = np.concatenate(y_test)
        y_pred_all = np.concatenate(y_pred)

        # Generate a classification report for the mean values
        mean_classification_rep = classification_report(y_true_all, y_pred_all, output_dict=True)

        # Extract values from the classification report dictionary
        report_data = []
        for label, scores in mean_classification_rep.items():
            if isinstance(scores, dict):
                report_data.append([label, round(scores['precision'], 3), round(scores['recall'], 3), round(scores['f1-score'], 3), scores['support']])
            else:
                report_data.append([label, round(scores, 3), "-", "-", "-"])

        # Create a DataFrame for the classification report
        columns = ['Label', 'Precision', 'Recall', 'F1 Score', 'Support']
        classification_report_df = pd.DataFrame(report_data, columns=columns)

        # Create a table for the classification report
        table2 = axs[2, 1].table(cellText=classification_report_df.values,
                                colLabels=classification_report_df.columns,
                                loc='center')
        table2.auto_set_font_size(False)
        table2.set_fontsize(10)

        # Add a title to the table
        axs[2, 1].set_title('Average Classification Report')

        # Hide axes
        axs[2, 1].axis('off')

        # Create a new subplot that spans two fields
        gs = gridspec.GridSpec(4, 3)
        ax = plt.subplot(gs[2:, 2])

        # Create a DataFrame from args
        args_df = pd.DataFrame(list(vars(args).items()), columns=['Key', 'Value'])

        # Add line breaks to the 'Value' column
        args_df['Value'] = args_df['Value'].apply(lambda x: textwrap.fill(str(x), width=60))

        # Create a table for the arguments
        table3 = ax.table(cellText=args_df.values,
                        colLabels=args_df.columns,
                        loc='center')
        table3.auto_set_font_size(False)
        table3.set_fontsize(8)
        table3.auto_set_column_width(col=list(range(len(args_df.columns))))
        table3.scale(1, 1.5)
        ax.set_title('Arguments')
        ax.axis('off')
        axs[2, 2].axis('off')
        axs[3, 2].axis('off')

        # Create a bar plot for the feature importance
        if context_dict is not None and 'feature_importance' in context_dict:
            feature_importance = context_dict['feature_importance']
            feature_importance_df = pd.DataFrame(feature_importance).transpose()
            feature_importance_df.columns = ['Mean', 'Standard Deviation']

            # Sort the feature importance values by the mean
            feature_importance_df = feature_importance_df.sort_values(by='Mean', ascending=False)

            # Plot the feature importance with error bars for the standard deviation
            sns.barplot(x='Mean', y=feature_importance_df.index, data=feature_importance_df, ax=axs[3, 0], color='blue', xerr=feature_importance_df['Standard Deviation'].values)
            axs[3, 0].set_title('Feature Importance')
            axs[3, 0].set_xlabel('Mean Coefficient')
            axs[3, 0].set_ylabel('Feature')

            # Create a DataFrame from feature_importance
            feature_importance_df = pd.DataFrame(feature_importance).transpose()
            feature_importance_df.columns = ['Mean', 'Standard Deviation']

            # Round the values up to 3 decimal places
            feature_importance_df = feature_importance_df.round(3)

            # Create a table for the feature importance
            ax = axs[3, 1]
            table4 = ax.table(cellText=feature_importance_df.values,
                            colLabels=feature_importance_df.columns,
                            rowLabels=feature_importance_df.index,
                            loc='center')
            table4.auto_set_font_size(False)
            table4.set_fontsize(8)
            table4.auto_set_column_width(col=list(range(len(feature_importance_df.columns))))
            table4.scale(1, 1.5)
            ax.set_title('Feature Importance')
            ax.axis('off')

        # Adjust layout
        plt.tight_layout()

        print(f'Saving plot to {os.path.join(BASE_DIR, args.output, filename_pdf)}...')
        plt.savefig(os.path.join(BASE_DIR, args.output, filename_pdf))
        plt.close()


    else:
        print('y_test and y_pred are not lists of lists. Using as is...')
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred, zero_division=0)
        recall = metrics.recall_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)
        cm = metrics.confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)

        print('\n')
        print('______________________________')
        print('Metrics:')
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        print('______________________________')
        print('\n')
        print('Classification report:')
        print(classification_rep)

        # Generate the classification report
        classification_rep = classification_report(y_test, y_pred, output_dict=True)

        # Convert the classification report to a DataFrame
        classification_rep_df = pd.DataFrame(classification_rep).transpose()

        # Round the values to 3 decimal places
        classification_rep_df = classification_rep_df.round(3)

        # Create a figure and a set of subplots
        fig, axs = plt.subplots(9, figsize=(12, 24))

        # Adjust the position of the subplots to give more space to the table
        plt.subplots_adjust(left=0.2, bottom=0.05, right=0.8, top=0.95, wspace=0.2, hspace=0.5)

        # Add a title to the plot
        fig.suptitle(f'{model_name} - {TIME} - {TOKEN}')

        # Plot the metrics
        axs[0].bar(['Accuracy', 'Precision', 'Recall', 'F1'], [accuracy, precision, recall, f1])
        axs[0].set_ylabel('Score')
        axs[0].set_ylim([0, 1])

        # Plot the confusion matrix
        ConfusionMatrixDisplay(cm, display_labels=['non-grievance', 'grievance']).plot(ax=axs[1])

        # Create a graph for the training loss
        if context_dict is not None and 'train_loss' in context_dict:
            for idx, loss_fn in enumerate(context_dict['train_loss']):
                print("train loss", loss_fn)
                axs[2].plot(loss_fn, label=f'Fold {idx+1}')
            axs[2].set_title('Training Loss')
            axs[2].set_xlabel('Epoch')
            axs[2].set_ylabel('Loss')


        # Create a graph for the evaluation loss
        if context_dict is not None and 'eval_loss' in context_dict:
            for idx, loss_fn in enumerate(context_dict['eval_loss']):
                print("eval loss", loss_fn)
                axs[3].plot(loss_fn, label=f'Fold {idx+1}')
            axs[3].set_title('Evaluation Loss')
            axs[3].set_xlabel('Epoch')
            axs[3].set_ylabel('Loss')

        # Add the classification report as a table
        axs[4].axis('tight')
        axs[4].axis('off')
        table = axs[4].table(cellText=classification_rep_df.values, colLabels=classification_rep_df.columns, rowLabels=classification_rep_df.index, cellLoc = 'center', loc='center')

        # Adjust the font size of the table to make it smaller
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        # Create a new subplot that spans two fields
        gs = gridspec.GridSpec(9, 1)
        ax = plt.subplot(gs[5:7, 0])

        # Create a DataFrame from args
        args_df = pd.DataFrame(list(vars(args).items()), columns=['Key', 'Value'])

        # Add line breaks to the 'Value' column
        args_df['Value'] = args_df['Value'].apply(lambda x: textwrap.fill(str(x), width=80))

        # Create a table for the arguments
        table3 = ax.table(cellText=args_df.values,
                          colLabels=args_df.columns,
                          loc='center')
        table3.auto_set_font_size(False)
        table3.set_fontsize(8)
        table3.auto_set_column_width(col=list(range(len(args_df.columns))))
        table3.scale(1, 1)
        ax.set_title('Arguments', y=1.13)
        ax.axis('off')
        axs[5].axis('off')
        axs[6].axis('off')

        # Create a bar plot for the feature importance
        if context_dict is not None and 'feature_importance' in context_dict:
            feature_importance = context_dict['feature_importance']
            feature_importance_df = pd.DataFrame(feature_importance).transpose()
            feature_importance_df.columns = ['Mean', 'Standard Deviation']

            # Sort the feature importance values by the mean
            feature_importance_df = feature_importance_df.sort_values(by='Mean', ascending=False)

            # Plot the feature importance with error bars for the standard deviation
            sns.barplot(x='Mean', y=feature_importance_df.index, data=feature_importance_df, ax=axs[7], color='blue', xerr=feature_importance_df['Standard Deviation'].values)
            axs[7].set_title('Feature Importance')
            axs[7].set_xlabel('Mean Coefficient')
            axs[7].set_ylabel('Feature')

            # Create a DataFrame from feature_importance
            feature_importance_df = pd.DataFrame(feature_importance).transpose()
            feature_importance_df.columns = ['Mean', 'Standard Deviation']

            # Round the values up to 3 decimal places
            feature_importance_df = feature_importance_df.round(3)

            # Create a table for the feature importance
            table4 = axs[8].table(cellText=feature_importance_df.values,
                                colLabels=feature_importance_df.columns,
                                rowLabels=feature_importance_df.index,
                                loc='center')
            table4.auto_set_font_size(False)
            table4.set_fontsize(10)
            axs[8].set_title('Feature Importance')
            axs[8].axis('off')

        print(f'Saving plot to {os.path.join(BASE_DIR, args.output, filename_pdf)}...')
        plt.savefig(os.path.join(BASE_DIR, args.output, filename_pdf))
        plt.close()


    if args.count_features:
        features = []
        tags = []

        if 'group' in args.features or 'all' in args.features:
            features.append('group')
            tags.append('+GRUPO')
        if 'cta' in args.features or 'all' in args.features:
            features.append('cta')
            tags.append('+ACCION')
        if 'problem' in args.features or 'all' in args.features:
            features.append('problem')
            tags.append('+PROBLEMA')
        if '1psg' in args.features or 'all' in args.features:
            features.append('1psg')
            tags.append('+1SG')
        if 'discourse' in args.features or 'all' in args.features:
            features.append('discourse')
            tags.append('+DISURSO')
        if 'sentencemod' in args.features or 'all' in args.features:
            features.append('sentencemod')
            tags.append('+MODIFICADOR')

        if args.feature_normal:
            for feature, tag in zip(features, tags):
                if feature in args.features or 'all' in args.features:
                    x_tag = re.escape(tag)
                    master_df[feature] = master_df.apply(lambda row: default_normalize(len(re.findall(x_tag, str(row['tagged_content']), re.IGNORECASE)), row['num_sentences']), axis=1)

        else:
            for feature, tag in zip(features, tags):
                if feature in args.features or 'all' in args.features:
                    x_tag = re.escape(tag)
                    master_df[feature] = master_df.apply(lambda row: len(re.findall(x_tag, str(row['tagged_content']), re.IGNORECASE)), axis=1)


    if args.save_dataframe:
        print("Saving results to excel...")
        master_df.to_excel(BASE_DIR / args.output / filename_xlsx, index = False)

    if args.save_callbacks and context_dict is not None:
        print("Saving callbacks to excel...")
        callbacks_df = pd.DataFrame()
        store_df = False
        if context_dict.get('eval_loss'):
            callbacks_df['eval_loss'] = pd.Series(context_dict.get('eval_loss') if isinstance(context_dict.get('eval_loss'), list) else [context_dict.get('eval_loss')])
            store_df = True
        if context_dict.get('train_loss'):
            callbacks_df['train_loss'] = pd.Series(context_dict.get('train_loss') if isinstance(context_dict.get('train_loss'), list) else [context_dict.get('train_loss')])
            store_df = True
        if store_df:
            callbacks_df.to_excel(BASE_DIR / args.output / f'{TIME}-{TOKEN}-{model_name}-callbacks.xlsx', index = False)

    return