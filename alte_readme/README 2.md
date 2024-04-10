# Fine-tuned classifier based on spanBERTa

This is a classifier, based on the spanish trained RoBERTa model. 
It consists of: 
- main.py (main function)
- data.py: sanitizes the dataset, splits into train and test-set (with transformers load_dataset function), shows the label distribution among datasets and gets the true item labels
- model.py: loads model (without cache, 2 labels to be classified), tokenizes with spanBERTa-specific tokenizer, sets loggins steps (still need to work on that!) and training arguments, evaluates and predicts for the test set

 Corpus: 
        - Training set: 492 items with label 0, 75 items with label 1
        - Test set: 125 items with label 0, 17 items with label 1

1.  Model 2024-01-31-21-44-47

- Model Architecture: 

  [\[Title\] (model architecture)
    ](<Screenshot 2024-01-31 at 22.15.52.png>)

### Tried to freeze the already pre-trained parameters and only train the classifier-layers. Seems to be a common practice to prevent overfitting, did not work for me. 

- Parameters (specified in codebase/model.py)
    - 3 Epochs
    - Training batch size 8, Evaluation batch size 32 (default?)
    - learning rate 1e-9 for all parameters
    - arguments: 
        output_dir='./results',          # output directory
		num_train_epochs=2,              # total number of training epochs
		evaluation_strategy='epoch',     # evaluate the model after each epoch
		logging_strategy='epoch',
		logging_steps =num_train_steps_per_epoch,
		do_train = True,
		do_eval = True,
		per_device_train_batch_size=train_batch_size,  # batch size per device during training
		per_device_eval_batch_size=32,   # batch size for evaluation
		warmup_steps=100,                # number of warmup steps for learning rate
		weight_decay=0.01,               # strength of weight decay
		logging_dir='./logs',  

3. Results: 

    results model 2024-01-31-21-44-47
    |  |precision | recall | f1-score | support |
    | -| ------   | ------ | -------- | ------  |
    | 0| 0.98     | 0.95   | 0.96     | 125     |
    | 1| 0.70     | 0.82   | 0.76     | 17      |
    |  |          |        |          |         |
    |accuracy |   |        |  0.94    | 142     |
    |macro avg|0.84| 0.89  | 0.86     | 142     |
    |weighted avg| 0.94| 0.94| 0.94   | 142     |

     
    Loss: 
    [\[Title\](results/plot_2024-01-31_21-45-01.pdf) 
    ](plot_2024-01-31_21-45-01.pdf)

    [\[Title\](confusion matrix)
    ](confusion_matrix_2024-01-31_21-45-18.png)

2. Model 2024-01-31_23-16-52

    - 3 Epochs
    - Training batch size 8, Evaluation batch size 8
    - learning rate 1e-6 for all parameters
    - arguments: 
        output_dir='./results',          # output directory
		num_train_epochs=3,              # total number of training epochs
		evaluation_strategy='epoch',     # evaluate the model after each epoch
		logging_strategy='epoch',
		logging_steps =num_train_steps_per_epoch,
		do_train = True,
		do_eval = True,
		per_device_train_batch_size=train_batch_size,  # batch size per device during training
		per_device_eval_batch_size=8,   # batch size for evaluation
		warmup_steps=0,                # number of warmup steps for learning rate
		weight_decay=0.01,               # strength of weight decay
		logging_dir='./logs',  

- Results
    results model 2024-01-31_23-16-52
    |  |precision | recall | f1-score | support |
    | -| ------   | ------ | -------- | ------  |
    | 0| 0.95     | 0.97   | 0.96     | 125     |
    | 1| 0.73     | 0.65   | 0.69    | 17      |
    |  |          |        |          |         |
    |accuracy |   |        |  0.93    | 142     |
    |macro avg|0.84| 0.81  | 0.82     | 142     |
    |weighted avg| 0.93| 0.34| 0.93   | 142     |

### Model 2024-02-01-19-46

--> Check again!"

-   Parameters: 
    - all layers
    - learning rate 1e-9
    - 3 epochs
    - Training and evaluation batch size 16
    training_args = TrainingArguments(
		output_dir='./results',          # output directory
		num_train_epochs=3,              # total number of training epochs
		evaluation_strategy='epoch',     # evaluate the model after each epoch
		logging_strategy='epoch',
		logging_steps =num_train_steps_per_epoch,
		do_train = True,
		do_eval = True,
		per_device_train_batch_size=train_batch_size,  # batch size per device during training
		per_device_eval_batch_size=16,   # batch size for evaluation
		warmup_steps=100,                # number of warmup steps for learning rate
		weight_decay=0.01,               # strength of weight decay
		logging_dir='./logs',            
		)


- Results: 
    results model 2024-02-01-19-46
    |  |precision | recall | f1-score | support |
    | -| ------   | ------ | -------- | ------  |
    | 0| 0.99     | 0.87   | 0.93     | 125     |
    | 1| 0.50     | 0.94   | 0.65     | 17      |
    |  |          |        |          |         |
    |accuracy |   |        | 0.88     | 142     |
    |macro avg|0.75| 0.91  | 0.79     | 142     |
    |weighted avg| 0.93| 0.88| 0.89   | 142     |

