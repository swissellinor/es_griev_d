# Masters Thesis: Automatically Identifying/ Classifying Grievance Frames in Latin-American Elite Discourse

This repository was created for my master's project. Three models are evaluated on a latin-american grievances-corpus. 

## Pipeline
This is a pipeline to classify grievances using different models

- The pipeline requires the dependencies from requirements.txt
- More information on the rule-based features / dictionary are provided here: https://github.com/swissellinor/thesis/tree/main/alte_readme

- Three models are tested: 
1. logisitic regression based on above-mentioned rule based features
2. compression approach (npc-gzip, adapted from https://github.com/bazingagin/npc_gzip)
3. spanish pre-trained RoBERTa (https://github.com/chriskhanhtran/spanish-bert)

## logistic regression on rule-based features

Best performing with: 
- k-fold cross validation (5 folds)
- upsampling of the minority group in the training sets
- features: group, cta, problem, sentiment, discourse, sentence-modifier, 1p sg. 

  - The results were calculated by calculating the metrics for each fold separately and taking the mean and sd of these values: 

    |    | mean | standard deviation |
    | -- | ----- | ----------------- |
    |accuracy | 0.819 | 0.034 |
    |precision | 0.403 | 0.062 |
    | recall | 0.0729 | 0.056 |
    | f1 | 0.515 | 0.052 |

  
- These results were calculated on the concatenated predictions of all folds:
 
    |  |precision | recall | f1-score | support |
    | -| ------   | ------ | -------- | ------  |
    | 0| 0.954     | 0.833   | 0.889     | 617     |
    | 1| 0.394     | 0.728   | 0.511     | 92      |
    |  |          |        |          |         |
    |accuracy |   |        |  0.819   | 709     |
    |macro avg|0.674| 0.781  | 0.7     | 709     |
    |weighted avg| 0.881| 0.819| 0.84   | 709    |
    
    | Average confusion matrix |                      |
    | ---------------------  | -------------------- |
    | 100  (True negatives)  |  21 (False positives)|
    | 5    (False negatives) | 13 (True positives)  |



## compression approach

- Best performing with single split (80% training set and 20% test set) without upsampling.
    
    |  |precision | recall | f1-score | support |
    | -| ------   | ------ | -------- | ------  |
    | 0| 0.905     | 0.919   | 0.912    | 124     |
    | 1| 0.412    | 0.368   | 0.389     | 19      |
    |  |          |        |          |         |
    |accuracy |   |        |  0.846    | 143     |
    |macro avg|0.658| 0.644  | 0.65     | 143     |
    |weighted avg| 0.839| 0.846| 0.842   | 143     | 


    | Average Confustion Matrix |            |
    | ---------------   |  ----------------  |
    | 114  (True negatives)   |  10 (False positives)  |
    | 12    (False negatives) | 7 (True positives)     |

    
- Hybrid model: with group, cta, problem, sentiment

    |  |precision | recall | f1-score | support |
    | -| ------   | ------ | -------- | ------  |
    | 0| 0.938    | 0.976   | 0.957    | 124     |
    | 1| 0.786    | 0.579   | 0.667     | 19      |
    |  |          |        |          |         |
    |accuracy |   |        |  0.8923    | 143     |
    |macro avg|0.862| 0.777  | 0.812     | 143     |
    |weighted avg| 0.918| 0.923| 0.918   | 143     | 


    | Average Confusion Matrix  |            |
    | ---------------   |  ----------------  |
    | 121 (True negatives) |  3 (False positives) |
    | 8  (False negatives) | 11 (True positives) |



## Spanish trained RoBERTa "spanberta" 
(if RoBERTa is not approproate, an alternative could be spanish trained BERT (beto: https://github.com/dccuchile/beto) )

preliminary results spanberta with
- batch size = 16 
(update: forgot sanitizing and lowercasing): 

    results model_2024-01-30_10-23-49
    |  |precision | recall | f1-score | support |
    | -| ------   | ------ | -------- | ------  |
    | 0| 0.96     | 0.94   | 0.95      | 125   |
    | 1| 0.63    | 0.71   | 0.67    | 17     |
    |  |          |        |          |         |
    |accuracy |   |        |  0.92   | 142     |
    |macro avg|0.80| 0.82  | 0.81  | 142     |
    |weighted avg| 0.92| 0.92| 0.92   | 142     | 

preliminary results spanberta batch size = 16 with sanitizing and lowercasing: 
    
    results model_2024-01-30_10-43-40
    |  |precision | recall | f1-score | support |
    | -| ------   | ------ | -------- | ------  |
    | 0| 0.97     | 0.94   | 0.96     | 125     |
    | 1| 0.64     | 0.82   | 0.72      | 17     |
    |  |          |        |          |         |
    |accuracy |   |        |  0.92    | 142     |
    |macro avg|0.81| 0.82  | 0.88     | 142     |
    |weighted avg| 0.93| 0.92| 0.93   | 142     | 

 

## Corpora
- Corpora consisting of Latin-American Manifestos of which 93 are grievances
    - manifestos_speech_complete: 713 items 
        - 93 grievances (13%) and 620 non-grievances (87%)
    - manifestos_speech_dev: 20 items for code testing-purposes
    - train.csv, 80% of the data
         - 75 grievances (ca 13%) 492 non-grievances (ca 87%)
    - test.csv, 20% of the data
        - 17 grievances (ca 13%) and 125 non-grievances (87%)


## TBD: 

- Model selection
- Solver and penalty for log-reg model
- when using another BERT-model, I get the following error: ValueError: Expected input batch_size (4096) to match target batch_size (8). Could be due to loss, needs to be checked. ATM, I am using xlm-roberta-large, while spanberta is uncommented. This would need to be changed in order for the code to work.

## LOG changes
- tried using another RoBERTa model for comparison
- implemented a pipeline: Model and feature selection now happening in command line 
- implemented kfold, worked on upsampling, hybrid models now available
- 13.02.24: found a bug in code which was messing with the structure of the dataset
- 27.02.24: worked on spanberta, plotting results in ./results
    - To-Do: clean up spanberta-codebase and main.py
- 26.01.24: worked on spanberta
- 20.01.24
    - implemented gzip
    - implemented feature importance for logreg
- first try gzip
-  07.01.24: multiple changes: 
    - got rid of jupyter script as it was not working reliably. 
    - changed the structure to one main.py and multiple dependencies for a better overview.
    - implemented command-line interface in which parameters can be adjusted accordingly. 
    - adapted feature tags since only few of them are signficanct. These are now the only ones that are tagged as default.
    - adjusted corpora: "_dev" contains the first 20 items for better testing, "unseen_data" contain the last 100 items and remain for later purposes
- 29.12.23: R-script: sentiment and number of sentences as possible features?
- 29.12.23: fixed problem framing and conditional, started R-script for feature-analysis
- 28.12.23: worked on corpus_analysis.ipynb, added normalized feature counts
- 26.12.23: worked on corpus_analysis.ipynb, added some statistical counts
- 24.12.23: added corpus_analysis.ipynb
- 18.12.23: worked on rule-based pipeline, implemented pronouns. Need to check CTAs
- 15.12.23: worked on rule-based pipeline, implemented problem framing and CTA
- 12.12.23: worked on rule-based pipeline, implemented the features that should be detected and tagged in the corpus.

## Scripts and Servers

### KIRSCHSAFTSCHORLE

- 24x 3,5 gHz INTEL XEON CPU
- 96GB RAM
- 4TB+ INTEL Enterprise SSD-Storage

### the start script

#### Usage:

In Terminal type:
./start-kirschsaftschorle.sh

The script automatically connects to kirschsaftschorle via a reverse ssh shell and tunnels port _8888_ to the local host while starting a new jupyter lab instance.

The script is not yet set up to trap any _SIGTERM_ from the local host.
**So to avoid orphan kernels it is best practice to shut down all Kernels under "Kernel" -> "Shut Down All Kernels" in the browser menu after/befor working.**

Jupyter-Lab is now available under http://localhost:8888/

To close jupyther lab and the ssh tunnel press _control_ + _c_ in the terminal that hosts the jupyter lab session.
No extra steps required.

#### Caveats

There is only one reverse ssh shell allowed per session because of how the internet works.

For accessing the server, there is another script called **login-kirschsaftschorle.sh**

Because of the missing _SIGTERM_ there is a possibility that the garbage collector is not clearing out the ram, which could cause a critical system failure. Shutting down all kernels once in a while should fix the issue.

#### STANZA

max-workers: ca 20

### login-holle.sh

In Terminal type:
./login-holle.sh

jupyter lab --no-browser --ip="0.0.0.0" .
http://192.168.2.2:8888/

ping 192.168.2.2
