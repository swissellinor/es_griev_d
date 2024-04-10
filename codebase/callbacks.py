### Import Settings ###
from settings import BASE_DIR

### Import Dependencies ###
from transformers import TrainerCallback

### CODE ###
# Callback to store the training and evaluation loss after each logging step
class LossCallback(TrainerCallback): 
	def __init__(self):
		super().__init__()
		self.train_losses = []
		self.eval_losses = []
		
	def on_train_begin(self, args, state, control, **kwargs):
		self.start_fold()
	
	def start_fold(self):
		self.train_losses.append([])
		self.eval_losses.append([])
	
	def on_log(self, args, state, control, logs=None, **kwargs):
		"Store the training and evaluation loss after each logging step"
		if state.is_local_process_zero:
			if 'loss' in logs.keys():
				self.train_losses[-1].append(logs['loss'])
			if 'eval_loss' in logs.keys():
				self.eval_losses[-1].append(logs['eval_loss'])