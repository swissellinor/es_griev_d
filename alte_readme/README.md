## Statistical classifier that uses logistic-regression to classify grievances based on feature-based rules.

Pipeline that helps annotate features that are present in grievance frames in latin-american elite discourse (manifestos). In the next step the feature-counts for each item are transformed into vectors and given into a logistic regression model. 


### FEATURES

Based on a qualitative analysis of the corpus, the multiple features were found as indicators of grievance frames and seven of them turned out to be significant: 


**sentiment**
- sentiment calculated with STANZA 0 - negative, 1- neutral, 2 - positive
- Grievances are significantly more negative. Therefore, the higher the sentiment, the more probable it is a non-grievance

**deprived_group** (Tag: +GRUPO): 
- wordlist here: https://github.com/swissellinor/thesis/blob/main/esgrievd/wordlists/deprived_group.jsondeprived_group.json
- the higher the count of deprived_group, the more probable the item is a grievance

**Problem framing:** (Tag: +PROBLEMA): 
- wordlist here: https://github.com/swissellinor/thesis/blob/main/esgrievd/wordlists/problem_frame.json
- the higher the count of problem_frames, the more probable it is a grievance

**Call to Actions** (Tag: +CTA)
- deber + verb
- hay que + verb
- the higher the count of CTAs, the less probable it is a grievance. This could be due to an erroneous rule.


|Discourse relational features (Tag: +DISCURSO) | adversative conjunctions | causal | consecutive |
| ----- | ---- | ----- | ---- |
| | pero| porque | |
| | aunque | pues | indudablamende |
| | sin embargo | ya que | ciertamente |
| | no obstante | debido a | por supuesto |
| | pese a (que)| por eso | claro que si |
| | a pesar de (que) | así que | en efecto |
| | mientras que | por (lo) tanto | como consecuencia de |
| | no obstante (que) | como resultado 
| |por otro lado | de ahí que |
| | aun así | |
| | |siendo eso la razón porque|
| | | por cuya razón |
| | |la principal razón por la que
| | |por tal motivo|
| | | por ello |
- Discourse relational features can be found in the wordlists too. 
- The higher the count of features, the more probable it is a grievance

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
