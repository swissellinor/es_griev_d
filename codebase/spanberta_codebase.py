### Import Settings ###
from settings import BASE_DIR

### Import Dependencies ###
import datetime
import torch

#from torch.optim import AdamW
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datasets import Dataset as ds

### CODE ###
class LossCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.eval_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            # logs should contain the training loss
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])
            # logs should contain the evaluation loss
            if 'eval_loss' in logs:
                self.eval_losses.append(logs['eval_loss'])


def spanberta_tokenize(train_set, test_set):

	def tokenizer(batched_text, padding=True, truncation=True):
		#tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', max_length = 512, cache_dir = str(BASE_DIR / "models"))
		tokenizer = RobertaTokenizerFast.from_pretrained("skimai/spanberta-base-cased", max_length = 512, cache_dir = str(BASE_DIR / "models"))
		return tokenizer(batched_text['content'], padding=padding, truncation=truncation)
	
	train_set = ds.from_pandas(train_set)
	test_set = ds.from_pandas(test_set)
    
    # Map train and test set
	train_set = train_set.map(tokenizer, batched=True, batch_size=len(train_set))
	test_set = test_set.map(tokenizer, batched=True, batch_size=len(test_set))


    # Set column names and types
	train_set.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
	test_set.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
	
	return train_set, test_set

def spanberta_load(args):

	N = 6
	
	#load the model without using cache, specifies the number of labels to be classified
	model = RobertaForSequenceClassification.from_pretrained("skimai/spanberta-base-cased", num_labels = 2, hidden_dropout_prob = 0.1, attention_probs_dropout_prob = 0.1, cache_dir = str(BASE_DIR / "models"))
	
	# print the model on each epoch
	if args.print_model:
		print(model)
      
	# for i in range(N):
	# 	for param in model.roberta.encoder.layer[i].parameters():
	# 		param.requires_grad = False

	# for i in range(N, len(model.roberta.encoder.layer)):
	# 	for param in model.roberta.encoder.layer[i].parameters():
	# 		param.requires_grad = True

	for param in model.parameters(): 
		param.requires_grad = False
	
	for param in model.roberta.encoder.layer[-2:].parameters():
		param.requires_grad = True
		
	optimizer = torch.optim.AdamW([{"params": model.roberta.parameters(), "lr": args.lr}])

	return model

def set_logging_steps(train_data, args):
	#set logging steps to the number of steps in my epoch
	num_train_examples = len(train_data)
	train_batch_size = args.batch_size
	num_train_steps_per_epoch = num_train_examples // train_batch_size

	return train_batch_size, num_train_steps_per_epoch

def setup_training_args(train_batch_size, num_train_steps_per_epoch, args):
	now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	model_name = f"model_{now}"
	training_args = TrainingArguments(
		output_dir=str(BASE_DIR / args.output),          # output directory
		num_train_epochs=args.epoch,              # total number of training epochs
		evaluation_strategy='steps',     # evaluate the model after certain number of steps
		logging_strategy='steps',
		logging_steps =num_train_steps_per_epoch,
		#save_strategy = 'steps',
		#save_steps = num_train_steps_per_epoch,
		do_train = True,
		do_eval = True,
		per_device_train_batch_size=args.batch_size,  # batch size per device during training
		per_device_eval_batch_size=args.batch_size,   # batch size for evaluation
		warmup_steps=100,                # number of warmup steps for learning rate
		weight_decay=0.01,               # strength of weight decay
		logging_dir=str(BASE_DIR / 'logs' / model_name),
		#load_best_model_at_end=True            
		)
	return training_args

def compute_metrics(pred):
	labels = pred.label_ids
	preds = pred.predictions.argmax(-1)
	precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
	acc = accuracy_score(labels, preds)
	return {
		'accuracy': acc,
		'f1': f1,
		'precision': precision,
		'recall': recall
	}

def spanberta_train(train_set, test_set, model, args): 

    train_batch_size, num_train_steps_per_epoch = set_logging_steps(train_set, args)
    
    print("Train model...\n")
    loss_callback = LossCallback()
    training_args = setup_training_args(train_batch_size, num_train_steps_per_epoch, args)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=test_set,
        compute_metrics=compute_metrics,
        callbacks=[loss_callback] #EarlyStoppingCallback(early_stopping_patience=)]
    )

    # Train the model
    trainer.train()

    train_loss = loss_callback.train_losses #loss_callback.train_losses[-1][-1] if loss_callback.train_losses else None
    eval_loss = loss_callback.eval_losses #loss_callback.eval_losses[-1][-1] if loss_callback.eval_losses else None

    return trainer, train_loss, eval_loss


def spanberta_predictor_classify(trainer, test_data):
	
	print('Predict Labels...')
	predictions = trainer.predict(test_data)
	predicted_labels = predictions.predictions.argmax(-1) #convert them from logits to labels

	return predicted_labels

