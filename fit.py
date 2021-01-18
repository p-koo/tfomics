import sys, time
import numpy as np
import tensorflow as tf
from . import metrics


#------------------------------------------------------------------------------------------
# Custom fits
#------------------------------------------------------------------------------------------

def fit_lr_decay(model, loss, optimizer, x_train, y_train, validation_data, verbose=True,  
                 metrics=['auroc', 'aupr'], num_epochs=100, batch_size=128, shuffle=True, 
                 es_patience=10, es_metric='auroc', criterion='max',
                 lr_decay=0.3, lr_patience=3, lr_metric='auroc'):


  # create tensorflow dataset
  trainset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  validset = tf.data.Dataset.from_tensor_slices(validation_data)

  # create trainer class
  trainer = Trainer(model, loss, optimizer, metrics)

  # set up learning rate decay
  lrdecay = trainer.set_lr_decay(decay_rate=lr_decay, patience=lr_patience, metric=lr_metric)

  # train model
  for epoch in range(num_epochs):  
    sys.stdout.write("\rEpoch %d \n"%(epoch+1))
    
    # train over epoch
    trainer.train_epoch(trainset, batch_size=batch_size, shuffle=shuffle, verbose=False)

    # validation performance
    trainer.evaluate('valid', validset, batch_size=batch_size, verbose=verbose)

    # check learning rate decay
    trainer.check_lr_decay()
   
    # check early stopping
    if trainer.early_stopping(es_metric, patience=es_patience, criterion=criterion):
      print("Patience ran out... Early stopping.")
      break
  
  history = trainer.get_metrics('train')
  history = trainer.get_metrics('valid', history)

  return trainer



#------------------------------------------------------------------------------------------
# Trainer class
#------------------------------------------------------------------------------------------


class Trainer():
  """Custom training loop from scratch"""

  def __init__(self, model, loss, optimizer, metrics):
    self.model = model
    self.loss = loss
    self.optimizer = optimizer

    metric_names = []
    for metric in metrics:
        metric_names.append(metric)

    self.metrics = {}
    self.metrics['train'] = MonitorMetrics(metric_names, 'train')
    self.metrics['valid'] = MonitorMetrics(metric_names, 'valid')
    self.metrics['test'] = MonitorMetrics(metric_names, 'test')

  @tf.function
  def train_step(self, x, y):
    with tf.GradientTape() as tape:
      preds = self.model(x, training=True)
      loss = self.loss(y, preds)
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
    return loss, preds


  @tf.function
  def test_step(self, x, y):
    preds = self.model(x, training=False)
    loss = self.loss(y, preds)
    return loss, preds


  def train_epoch(self, trainset, batch_size=128, shuffle=True, verbose=True):
    if shuffle:
      batch_dataset = trainset.shuffle(buffer_size=batch_size).batch(batch_size)
    num_batches = len(list(batch_dataset))

    start_time = time.time()
    for i, (x, y) in enumerate(batch_dataset):      
      loss_batch, pred = self.train_step(x, y)
      self.metrics['train'].update_running_loss_metric(loss_batch, y, pred)
      progress_bar(i+1, num_batches, start_time, bar_length=30, loss=np.mean(self.metrics['train'].running_loss))
    if verbose:
      self.metrics['train'].update_print()
    else:
      self.metrics['train'].update()


  def evaluate(self, name, dataset, batch_size=128, verbose=True):
    batch_dataset = dataset.batch(batch_size)
    num_batches = len(list(batch_dataset))
    for i, (x, y) in enumerate(batch_dataset):   
      loss_batch, pred = self.test_step(x, y)
      self.metrics[name].update_running_loss_metric(loss_batch, y, pred)
    if verbose:
      self.metrics[name].update_print()
    else:
      self.metrics[name].update()   
    

  def predict(self, x, batch_size=128):
    pred = self.model.predict(x, batch_size=batch_size)  
    return pred


  def early_stopping(self, metric, patience, criterion='min'):
    """check if validation loss is not improving and stop after patience
       runs out. This is an expensive early stopping. """

    status = False
    vals = self.metrics['valid'].get(metric)
    if criterion == 'min':
      index = np.argmin(vals)
    else:
      index = np.argmax(vals)
    if patience - (len(vals) - index - 1) <= 0:
      status = True
    return status


  def set_lr_decay(self, decay_rate, patience, metric):
    if metric == 'loss':
      criterion = 'min'
    else: 
      criterion = 'max'
    self.lr_decay = LRDecay(optimizer=self.optimizer, decay_rate=decay_rate, 
                            patience=patience, metric=metric, criterion=criterion)

  def check_lr_decay(self, name='valid'):
    self.lr_decay.check(self.metrics[name].get(self.lr_decay.metric)[-1])


  def get_metrics(self, name, metrics=None):
    if metrics is None:
      metrics = {}
    metrics[name+'_loss'] = self.loss
    for metric_name in self.metrics[name].metric_names:
      metrics[name+'_'+metric_name] = self.metrics[name].get(metric_name)
    return metrics


#------------------------------------------------------------------------------------------
# Helper classes
#------------------------------------------------------------------------------------------


class LRDecay():
  def __init__(self, optimizer, decay_rate=0.3, patience=10, metric='loss', criterion='min'):

    self.optimizer = optimizer
    self.lr = optimizer.lr
    self.decay_rate = tf.constant(decay_rate)
    self.patience = patience
    self.metric = metric
    self.criterion = criterion
    self.index = 0
    self.initialize()

  def initialize(self):
    if self.criterion == 'min':
      self.best_val = 1e10
      self.sign = 1
    else:
      self.best_val = -1e10
      self.sign = -1

  def status(self, val):
    """check if validation loss is not improving and stop after patience
       runs out"""  
    status = False
    if self.sign*val < self.sign*self.best_val:
      self.best_val = val
      self.index = 0
    else:
      self.index += 1
      if self.index == self.patience:
        self.index = 0
        status = True
    return status

  def decay_learning_rate(self):
    self.lr = self.lr * self.decay_rate
    self.optimizer.learning_rate.assign(self.lr)


  def check(self, val):
    if self.status(val):
      self.decay_learning_rate()
      print('  Decaying learning rate to %.6f'%(self.lr))


      

#----------------------------------------------------------------------


class MonitorMetrics():
  """class to monitor metrics during training"""
  def __init__(self, metric_names, name):
    self.name = name
    self.loss = []
    self.running_loss = []

    self.metric_update = {}
    self.metric = {}
    self.metric_names = metric_names
    self.initialize_metrics(metric_names)
    
  def initialize_metrics(self, metric_names):
    """metric names can be list or dict"""
    if 'acc' in metric_names:
      self.metric_update['acc'] = tf.keras.metrics.BinaryAccuracy()
      self.metric['acc'] = []
    if 'auroc' in metric_names:
      self.metric_update['auroc'] = tf.keras.metrics.AUC(curve='ROC')
      self.metric['auroc'] = []
    if 'aupr' in metric_names:
      self.metric_update['aupr'] = tf.keras.metrics.AUC(curve='PR')
      self.metric['aupr'] = []
    if 'cosine' in metric_names:
      self.metric_update['cosine'] = tf.keras.metrics.CosineSimilarity()
      self.metric['cosine'] = []
    if 'kld' in metric_names:
      self.metric_update['kld'] = tf.keras.metrics.KLDivergence()
      self.metric['kld'] = []
    if 'mse' in metric_names:
      self.metric_update['mse'] = tf.keras.metrics.MeanSquaredError()
      self.metric['mse'] = []

  def update_running_loss(self, running_loss):
    self.running_loss.append(running_loss)  
    return np.mean(self.running_loss)  

  def update_running_metrics(self, y, preds):    
    #  update metric dictionary
    for metric_name in self.metric_names:
      self.metric_update[metric_name].update_state(y, preds)

  def update_running_loss_metric(self, running_loss, y, preds):
    self.update_running_loss(running_loss)
    self.update_running_metrics(y, preds)

  def reset(self):
    for metric_name in self.metric_names:
      self.metric_update[metric_name].reset_states()

  def update(self):
    self.loss.append(np.mean(self.running_loss))
    self.running_loss = []    
    for metric_name in self.metric_names:
      self.metric[metric_name].append(np.mean(self.metric_update[metric_name].result()))
    self.reset()

  def update_print(self):
    self.update()
    self.print()

  def print(self):
    if self.loss:
      print('  %s loss:   %.4f'%(self.name, self.loss[-1]))
    for metric_name in self.metric_names:
      print("  " + self.name + " "+ metric_name+":\t{:.5f}".format(self.metric[metric_name][-1]))

  def get(self, name):
    if name == 'loss':
      return self.loss
    else:
      return self.metric[name]



#------------------------------------------------------------------------------
# Useful functions
#------------------------------------------------------------------------------


def progress_bar(iter, num_batches, start_time, bar_length=30, **kwargs):
  """plots a progress bar to show remaining time for a full epoch. 
     (inspired by keras)"""

  # calculate progress bar 
  percent = iter/num_batches
  progress = '='*int(round(percent*bar_length))
  spaces = ' '*int(bar_length-round(percent*bar_length))

  # setup text to output
  if iter == num_batches:   # if last batch, then output total elapsed time
    output_text = "\r[%s] %.1f%% -- elapsed time=%.1fs"
    elapsed_time = time.time()-start_time
    output_vals = [progress+spaces, percent*100, elapsed_time]
  else:
    output_text = "\r[%s] %.1f%%  -- remaining time=%.1fs"
    remaining_time = (time.time()-start_time)*(num_batches-(iter+1))/(iter+1)
    output_vals = [progress+spaces, percent*100, remaining_time]

  # add performance metrics if included in kwargs
  if 'loss' in kwargs:
    output_text += " -- loss=%.5f"
    output_vals.append(kwargs['loss'])
  if 'acc' in kwargs:
    output_text += " -- acc=%.5f"
    output_vals.append(kwargs['acc'])
  if 'auroc' in kwargs:
    output_text += " -- auroc=%.5f"
    output_vals.append(kwargs['auroc'])
  if 'aupr' in kwargs:
    output_text += " -- aupr=%.5f"
    output_vals.append(kwargs['aupr'])
  if 'pearsonr' in kwargs:
    output_text += " -- pearsonr=%.5f"
    output_vals.append(kwargs['pearsonr'])
  if 'mcc' in kwargs:
    output_text += " -- mcc=%.5f"
    output_vals.append(kwargs['mcc'])
  if 'mse' in kwargs:
    output_text += " -- mse=%.5f"
    output_vals.append(kwargs['mse'])

  # set new line when finished
  if iter == num_batches:
    output_text += "\n"
  
  # output stats
  sys.stdout.write(output_text%tuple(output_vals))
   
   


