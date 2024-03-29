import sys, time
import numpy as np
import tensorflow as tf


#------------------------------------------------------------------------------------------
# Custom fits
#------------------------------------------------------------------------------------------

def fit(model, loss, optimizer, x_train, y_train, validation_data, verbose=True,  
                 metrics=['auroc', 'aupr'], num_epochs=100, batch_size=128, shuffle=True, 
                 es_patience=10, es_metric='auroc', es_criterion='max'):


  # create tensorflow dataset
  trainset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  validset = tf.data.Dataset.from_tensor_slices(validation_data)

  # create trainer class
  trainer = Trainer(model, loss, optimizer, metrics)
  trainer.set_early_stopping(patience=es_patience, metric=es_metric, criterion=es_criterion)

  # train model
  for epoch in range(num_epochs):  
    sys.stdout.write("\rEpoch %d \n"%(epoch+1))
    
    # train over epoch
    trainer.train_epoch(trainset, batch_size=batch_size, shuffle=shuffle, verbose=False)

    # validation performance
    trainer.evaluate('valid', validset, batch_size=batch_size, verbose=verbose)
   
    # check early stopping
    if trainer.check_early_stopping('valid'):
      print("Patience ran out... Early stopping.")
      break
  
  # compile history
  history = trainer.get_metrics('train')
  history = trainer.get_metrics('valid', history)

  return history, trainer


def fit_lr_decay(model, loss, optimizer, x_train, y_train, validation_data, verbose=True,  
                 metrics=['auroc', 'aupr'], num_epochs=100, batch_size=128, shuffle=True, 
                 es_patience=10, es_metric='auroc', es_criterion='max',
                 lr_decay=0.3, lr_patience=3, lr_metric='auroc', lr_criterion='max'):


  # create tensorflow dataset
  trainset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  validset = tf.data.Dataset.from_tensor_slices(validation_data)

  # create trainer class
  trainer = Trainer(model, loss, optimizer, metrics)
  trainer.set_lr_decay(decay_rate=lr_decay, patience=lr_patience, metric=lr_metric, criterion=lr_criterion)
  trainer.set_early_stopping(patience=es_patience, metric=es_metric, criterion=es_criterion)

  # train model
  for epoch in range(num_epochs):  
    sys.stdout.write("\rEpoch %d \n"%(epoch+1))
    
    # train over epoch
    trainer.train_epoch(trainset, batch_size=batch_size, shuffle=shuffle, verbose=False)

    # validation performance
    trainer.evaluate('valid', validset, batch_size=batch_size, verbose=verbose)

    # check learning rate decay
    trainer.check_lr_decay('valid')
   
    # check early stopping
    if trainer.check_early_stopping('valid'):
      print("Patience ran out... Early stopping.")
      break
  
  # compile history
  history = trainer.get_metrics('train')
  history = trainer.get_metrics('valid', history)

  return history, trainer



def fit_lr_schedule(model, loss, optimizer, x_train, y_train, validation_data, verbose=True,  
                    metrics=['auroc', 'aupr'], num_epochs=100, batch_size=128, shuffle=True, 
                    es_patience=10, es_metric='auroc', es_criterion='max',
                    lr_schedule=None):


  # create tensorflow dataset
  trainset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  validset = tf.data.Dataset.from_tensor_slices(validation_data)

  # create trainer class
  trainer = Trainer(model, loss, optimizer, metrics)
  trainer.set_early_stopping(patience=es_patience, metric=es_metric, criterion=es_criterion)

  # train model
  for epoch in range(num_epochs):  
    sys.stdout.write("\rEpoch %d \n"%(epoch+1))

    # check learning rate scheduler
    if epoch in lr_schedule:
      trainer.set_learning_rate(lr_schedule[epoch])
      print("  Changed learning rate to: %.6f"%(lr_schedule[epoch]))    

    # train over epoch
    trainer.train_epoch(trainset, batch_size=batch_size, shuffle=shuffle, verbose=False)

    # validation performance
    trainer.evaluate('valid', validset, batch_size=batch_size, verbose=verbose)

    # check early stopping
    if trainer.check_early_stopping('valid'):
      print("Patience ran out... Early stopping.")
      break
  
  # compile history
  history = trainer.get_metrics('train')
  history = trainer.get_metrics('valid', history)

  return history, trainer


def fit_robust(model, loss, optimizer, attacker, x_train, y_train, validation_data, 
               num_epochs=200, batch_size=16, shuffle=True, metrics=['auroc','aupr'], 
               clean_epoch=10, mix_epoch=50,  es_start_epoch=50, 
               es_patience=20, es_metric='auroc', es_criterion='max',
               lr_decay=0.3, lr_patience=10, lr_metric='auroc', lr_criterion='max'):

  # create tensorflow dataset
  trainset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  validset = tf.data.Dataset.from_tensor_slices(validation_data)

  # create trainer class
  trainer = RobustTrainer(attacker, model, loss, optimizer, metrics)

  # set up learning rate decay
  trainer.set_lr_decay(decay_rate=lr_decay, patience=lr_patience, metric=lr_metric, criterion=lr_criterion)
  trainer.set_early_stopping(patience=es_patience, metric=es_metric, criterion=es_criterion)

  for i in range(clean_epoch):
    trainer.train_epoch(trainset, batch_size, shuffle, store=False)

  # train model
  for epoch in range(num_epochs):  
    sys.stdout.write("\rEpoch %d \n"%(epoch+1))
    
    # train over epoch
    if epoch < mix_epoch:
      trainer.robust_train_epoch(trainset, batch_size, shuffle, mix=True)
    else:
      trainer.robust_train_epoch(trainset, batch_size, shuffle, mix=False)

    # validation performance
    trainer.evaluate_attack('valid', validset, batch_size)
    
    # check early stopping
    if epoch >= es_start_epoch:
      
      # check learning rate decay
      trainer.check_lr_decay('valid')

      if trainer.check_early_stopping('valid'):
        print("Patience ran out... Early stopping.")
        break
  
  # compile history
  history = trainer.get_metrics('train')
  history = trainer.get_metrics('valid', history)

  return history, trainer



#------------------------------------------------------------------------------------------
# Trainer class
#------------------------------------------------------------------------------------------


class Trainer():
  """Custom training loop from scratch"""

  def __init__(self, model, loss, optimizer, metrics):
    self.model = model
    self.loss = loss
    self.optimizer = optimizer

    # metrics to monitor
    metric_names = []
    for metric in metrics:
        metric_names.append(metric)

    # class to help monitor metrics
    self.metrics = {}
    self.metrics['train'] = MonitorMetrics(metric_names, 'train')
    self.metrics['valid'] = MonitorMetrics(metric_names, 'valid')
    self.metrics['test'] = MonitorMetrics(metric_names, 'test')

  @tf.function
  def train_step(self, x, y, metrics):
    """training step for a mini-batch"""
    with tf.GradientTape() as tape:
      preds = self.model(x, training=True)
      loss = self.loss(y, preds)
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
    metrics.update_running_metrics(y, preds)
    return loss

  @tf.function
  def test_step(self, x, y, metrics, training=False):
    """test step for a mini-batch"""
    preds = self.model(x, training=training)
    loss = self.loss(y, preds)
    metrics.update_running_metrics(y, preds)
    return loss
    

  def train_epoch(self, trainset, batch_size=128, shuffle=True, verbose=False, store=True):
    """train over all mini-batches and keep track of metrics"""

    # prepare data
    if shuffle:
      trainset.shuffle(buffer_size=batch_size)
    batch_dataset = trainset.batch(batch_size)
    num_batches = len(list(batch_dataset))

    # train loop over all mini-batches 
    start_time = time.time()
    running_loss = 0
    for i, (x, y) in enumerate(batch_dataset):      
      loss_batch = self.train_step(x, y, self.metrics['train'])
      self.metrics['train'].running_loss.append(loss_batch)
      running_loss += loss_batch
      progress_bar(i+1, num_batches, start_time, bar_length=30, loss=running_loss/(i+1))

    # store training metrics
    if store:
      if verbose:
        self.metrics['train'].update_print()
      else:
        self.metrics['train'].update()


  def evaluate(self, name, dataset, batch_size=128, verbose=True, training=False):
    """Evaluate model in mini-batches"""
    batch_dataset = dataset.batch(batch_size)
    num_batches = len(list(batch_dataset))
    for i, (x, y) in enumerate(batch_dataset):   
      loss_batch = self.test_step(x, y, self.metrics[name], training)
      self.metrics[name].running_loss.append(loss_batch)

    # store evaluation metrics
    if verbose:
      self.metrics[name].update_print()
    else:
      self.metrics[name].update()   
    

  def predict(self, x, batch_size=128):
    """Get predictions of model"""
    pred = self.model.predict(x, batch_size=batch_size)  
    return pred


  def set_early_stopping(self, patience=10, metric='loss', criterion=None):
    """set up early stopping"""
    self.early_stopping = EarlyStopping(patience=patience, metric=metric, criterion=criterion)
    

  def check_early_stopping(self, name='valid'):
    """check status of early stopping"""
    return self.early_stopping.status(self.metrics[name].get(self.early_stopping.metric)[-1])


  def set_lr_decay(self, decay_rate, patience, metric='loss', criterion=None):
    """set up learning rate decay"""
    self.lr_decay = LRDecay(optimizer=self.optimizer, decay_rate=decay_rate, 
                            patience=patience, metric=metric, criterion=criterion)

  def check_lr_decay(self, name='valid'):
    """check status and update learning rate decay"""
    self.lr_decay.check(self.metrics[name].get(self.lr_decay.metric)[-1])


  def get_metrics(self, name, metrics=None):
    """return a dictionary of metrics stored throughout training"""
    if metrics is None:
      metrics = {}
    metrics[name+'_loss'] = self.metrics[name].loss
    for metric_name in self.metrics[name].metric_names:
      metrics[name+'_'+metric_name] = self.metrics[name].get(metric_name)
    return metrics


  def set_learning_rate(self, learning_rate):
    """short-cut to set the learning rate"""
    self.optimizer.learning_rate.assign(learning_rate)






class RobustTrainer(Trainer):
  """Custom robust training loop (inherits all functions/variables from Trainer)"""

  def __init__(self, attacker, model, loss, optimizer, metrics):
    self.attacker = attacker
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


  def robust_train_step(self, x, y, mix=False, verbose=False):
    """performs a training epoch with attack to inputs"""

    x_attack = self.attacker.generate(x, y)  # object from attacks.py
    
    # mix real and perturbed data together
    if mix:
      x_attack = tf.concat([x_attack, x], axis=0)
      y = tf.concat([y, y], axis=0)
    return self.train_step(x_attack, y, self.metrics['train'])


  def robust_train_epoch(self, trainset, batch_size=128, shuffle=True, mix=False, verbose=False, store=True):
    """performs a training epoch with attack to inputs"""

    # prepare dataset
    if shuffle:
      trainset.shuffle(buffer_size=batch_size)
    batch_dataset = trainset.batch(batch_size)
    num_batches = len(list(batch_dataset))

    # loop through mini-batches and perform robust training steps
    start_time = time.time()
    running_loss = 0
    for i, (x, y) in enumerate(batch_dataset):    
      loss_batch = self.robust_train_step(x, y, mix, verbose)
      self.metrics['train'].running_loss.append(loss_batch)
      running_loss += loss_batch
      progress_bar(i+1, num_batches, start_time, bar_length=30, loss=running_loss/(i+1))

    # store training metrics
    if store:
      if verbose:
        self.metrics['train'].update_print()
      else:
        self.metrics['train'].update()
    

  def generate_attack(self, dataset, batch_size=128):
    """generates attack using mini-batches """

    # batch data
    batch_dataset = dataset.batch(batch_size)
    num_batches = len(list(batch_dataset))
    if np.mod(len(list(dataset)), batch_size) != 0:
      num_batches -= 1

    # generate attacks and return as tensorflow dataset
    x_attack = []
    y_attack = []
    for i, (x, y) in enumerate(batch_dataset):    
      if len(x) == batch_size:
        x_attack.append(self.attacker.generate(x, y))
        y_attack.append(y)
    return tf.data.Dataset.from_tensor_slices((np.concatenate(x_attack, axis=0), 
                                               np.concatenate(y_attack, axis=0)))


  def evaluate_attack(self, name, dataset, batch_size=128, verbose=True):
    """evaluates model based on compromised data that has been attacked"""
    dataset_attack = self.generate_attack(dataset, batch_size)    
    self.evaluate(name, dataset_attack, batch_size, verbose)





#------------------------------------------------------------------------------------------
# Helper classes
#------------------------------------------------------------------------------------------


class LRDecay():
  def __init__(self, optimizer, decay_rate=0.3, patience=10, metric='loss', criterion=None):

    self.optimizer = optimizer
    self.lr = optimizer.lr
    self.decay_rate = tf.constant(decay_rate)
    self.patience = patience
    self.metric = metric

    if criterion is None:
      if metric == 'loss':
        criterion = 'min'
      else: 
        criterion = 'max'
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


  def check(self, val):
    """ check status of learning rate decay"""
    if self.status(val):
      self.decay_learning_rate()
      print('  Decaying learning rate to %.6f'%(self.lr))


  def decay_learning_rate(self):
    """ sets a new learning rate based on decay rate"""
    self.lr = self.lr * self.decay_rate
    self.optimizer.learning_rate.assign(self.lr)



class EarlyStopping():
  def __init__(self, patience=10, metric='loss', criterion=None):

    self.patience = patience
    self.metric = metric

    if criterion is None:
      if metric == 'loss':
        criterion = 'min'
      else: 
        criterion = 'max'
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
   
   


