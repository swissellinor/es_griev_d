# Masters Thesis: Automatically Identifying/ Classifying Grievance Frames in Latin-American Elite Discourse

This repository was created for my master's project. Three models are evaluated on a latin-american grievances-corpus. 

## Pipeline
This is a pipeline to classify grievances using different models

- The pipeline requires the dependencies from requirements.txt
- More information on the rule-based features / dictionary are provided here: https://github.com/swissellinor/thesis/tree/main/alte_readme

- Three models are tested: 
1. logistic regression based on above-mentioned rule based features
2. compression approach (npc-gzip, adapted from https://github.com/bazingagin/npc_gzip)
3. spanish pre-trained RoBERTa (https://github.com/chriskhanhtran/spanish-bert)

### FEATURES

Based on a qualitative analysis of the corpus, the multiple features were found as indicators of grievance frames and five of them turned out to be significant: 


**sentiment**
- sentiment calculated with STANZA 0 - negative, 1- neutral, 2 - positive
- Grievances are significantly more negative. Therefore, the higher the sentiment, the more probable it is a non-grievance

**deprived_group** (Tag: +GRUPO): 
- wordlist here: https://github.com/swissellinor/es_griev_d/blob/main/wordlists/deprived_group.json
- the higher the count of deprived_group, the more probable the item is a grievance

**Problem framing:** (Tag: +PROBLEMA): 
- wordlist here: https://github.com/swissellinor/es_griev_d/blob/main/wordlists/problem_frame.json
- the higher the count of problem_frames, the more probable it is a grievance


| sentence modifier (Tag: +MODIFICADOR) | negation | restriction | intensifier |
|----- | ---- | ----|---|
| | no | casi sin| muy |
| | sin (prep) | menor | demasiado |
| | nunca | minoria | realmente |
| | jamas | solo | bastante |
| | faltar (V) | restringir (V) | mas |
| | excluyir (V) | limitar (V) | menos |
| | ausencia de | poco| tan |
| |tampoco| unicamente | mucho|
| |ni siquiera| | super|
- The higher the count of modifier, the more probable it is a grievance

| Pronouns, Possessives, Reflexives | Singular (Tag: +SG) | Plural (Tag +PL) |
|----------| ------------------- | ---------------- |
| First person (Tag: +1) | yo, mí, me, mío, mía, míos, mías | nosotros, nosotras, nuestro, nuestras, nos |
| Second person (Tag: +2) | tú, te, tí, tuyo, tuya, tuyos, tuyas, se | ustedes, os, los, las, vuestro, vuestra, vuestros, vuestras, se |
| Third person (Tag: +3) |  él, ella, usted, lo, la, le, suyo, suya, se | ellos, ellas, ustedes, los, las, les, suyos, suyas, se |
- For pronouns, only the first person singular turned out to be significant. 
- The higher the count of 1P SG, the less probable it is a grievance.


**Call to Actions** (Tag: +CTA)
- deber + verb
- hay que + verb
- imperative verb forms
- the higher the count of CTAs, the less probable it is a grievance. This could be due to an erroneous rule.

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
