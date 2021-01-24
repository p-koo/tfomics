import sys, time
import numpy as np
import tensorflow as tf
from . import metrics
from .fit import LRDecay, EarlyStopping, progress_bar 

#------------------------------------------------------------------------------------------
# Custom fits
#------------------------------------------------------------------------------------------

def fit_lr_decay(model, loss, optimizer, x_train, y_train, validation_data, verbose=True,  
                 metrics=['auroc', 'aupr'], num_epochs=100, batch_size=128, shuffle=True, 
                 es_patience=10, es_metric='auroc', es_criterion='max',
                 lr_decay=0.3, lr_patience=3, lr_metric='auroc', lr_criterion='max'):

  # create tensorflow dataset
  trainset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  validset = tf.data.Dataset.from_tensor_slices(validation_data)

  # set up trainer
  trainer = Trainer(model, loss, optimizer, metrics)
  trainer.set_lr_decay(decay_rate=lr_decay, patience=lr_patience, metric=lr_metric)


  for epoch in range(num_epochs):  
    sys.stdout.write("\rEpoch %d \n"%(epoch+1))

    # train step
    trainer.train_epoch(trainset, shuffle=shuffle, batch_size=batch_size, verbose=verbose)

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



#------------------------------------------------------------------------------------------
# Trainer class
#------------------------------------------------------------------------------------------


class Trainer():
  def __init__(self, model, loss, optimizer, metrics):
    self.model = model
    self.metric_names = metrics
    self.loss = loss
    self.optimizer = optimizer

    self.metrics = {}
    self.metrics['train'] = MonitorMetrics('train', metrics)
    self.metrics['valid'] = MonitorMetrics('valid', metrics)
    self.metrics['test'] = MonitorMetrics('test', metrics)


  @tf.function
  def train_step(self, x, y):
    with tf.GradientTape() as tape:
      predictions = self.model(x, training=True)
      loss = self.loss(y, predictions)
      gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
    return loss, predictions


  @tf.function
  def test_step(self, x, y, training=False):
    preds = self.model(x, training=training)
    loss = self.loss(y, preds)
    return loss, preds
    

  def train_epoch(self, trainset, shuffle=True, batch_size=128, verbose=True):
    if shuffle:
      batch_dataset = trainset.shuffle(buffer_size=batch_size).batch(batch_size)
    num_batches = len(list(batch_dataset))
    
    running_loss = 0.
    pred_batch = []
    y_batch = []
    start_time = time.time()
    for i, (x, y) in enumerate(batch_dataset):      
      loss, pred = self.train_step(x, y)
      running_loss += loss
      pred_batch.append(pred)
      y_batch.append(y)
      if verbose:
        progress_bar(i+1, num_batches, start_time, bar_length=30, loss=running_loss/(i+1))
    pred = np.concatenate(pred_batch, axis=0)
    y = np.concatenate(y_batch, axis=0)

    # update metrics
    self.metrics['train'].update_loss(running_loss/num_batches)
    self.metrics['train'].update(y, pred)


  def evaluate(self, name, dataset, batch_size=128, verbose=True, training=False):
    batch_dataset = dataset.batch(batch_size)
    num_batches = len(list(batch_dataset))
    pred_batch = []
    y_batch = []
    loss = []
    for i, (x, y) in enumerate(batch_dataset):   
      loss_batch, preds = self.test_step(x, y, training)
      pred_batch.append(preds)
      y_batch.append(y)
      loss.append(loss_batch)
    pred = np.concatenate(pred_batch, axis=0)
    y = np.concatenate(y_batch, axis=0)
    loss = np.concatenate(loss, axis=0)

    # update metrics
    self.metrics[name].update_loss(np.mean(loss))
    self.metrics[name].update(y, pred)
    if verbose:
        self.metrics[name].print()
    

  def predict(self, x, batch_size=128):
    pred = self.model.predict(x, batch_size=batch_size)  
    return pred


  def set_early_stopping(self, patience=10, metric='loss', criterion='min'):
    self.early_stopping = EarlyStopping(patience=patience, metric=metric, criterion=criterion)

    
  def check_early_stopping(self, name='valid'):
    self.early_stopping.status(self.metrics[name].value[self.early_stopping.metric][-1])


  def set_lr_decay(self, decay_rate, patience, metric='loss', criterion='min'):
    self.lr_decay = LRDecay(optimizer=self.optimizer, decay_rate=decay_rate, 
                            patience=patience, metric=metric, criterion=criterion)


  def check_lr_decay(self, name='valid'):
    self.lr_decay.check(self.metrics[name].value[self.lr_decay.metric][-1])


  def get_metrics(self, name, metrics=None):
    if metrics is None:
      metrics = {}
    for metric_name in self.metrics[name].metric_names:
      metrics[name+'_'+metric_name] = self.metrics[name].value[metric_name]
    return metrics


#----------------------------------------------------------------------

class MonitorMetrics():
  """class to monitor metrics during training"""
  
  def __init__(self, name, metric_names):
    self.value = {}
    self.name = name
    self.metric_names = metric_names
    self.initialize_metrics(metric_names)
    

  def initialize_metrics(self, metric_names):
    """metric names can be list or dict"""
    self.value['loss'] = []
    if 'acc' in metric_names:
      self.value['acc'] = []
      self.value['acc_std'] = []
    if 'auroc' in metric_names:
      self.value['auroc'] = []
      self.value['auroc_std'] = []
    if 'aupr' in metric_names:
      self.value['aupr'] = []
      self.value['aupr_std'] = []
    if 'corr' in metric_names:
      self.value['corr'] = []
      self.value['corr_std'] = []
    if 'mcc' in metric_names:
      self.value['mcc'] = []
      self.value['mcc_std'] = []
    if 'mse' in metric_names:
      self.value['mse'] = []
      self.value['mse_std'] = []


  def calculate_metrics(self, label, pred):      
    metric_vals = {}
    if 'acc' in self.metric_names:
      metric_vals['acc'] = metrics.accuracy(label, pred)
    if 'auroc' in self.metric_names:
      metric_vals['auroc'] = metrics.auroc(label, pred)
    if 'aupr' in self.metric_names:
      metric_vals['aupr'] = metrics.aupr(label, pred)
    if 'rsquare' in self.metric_names:
      metric_vals['rsquare'] = metrics.rsquare(label, pred)
    if 'corr' in self.metric_names:
      metric_vals['corr'] = metrics.pearsonr(label, pred)
    return metric_vals


  def update_loss(self, loss):
    self.value['loss'].append(loss)
      

  def update(self, label, pred): 

    # calculate metrics   
    metric_vals = self.calculate_metrics(label, pred)

    #  update metric dictionary
    for metric_name in metric_vals.keys():
      self.value[metric_name].append(np.nanmean(metric_vals[metric_name]))
      self.value[metric_name+'_std'].append(np.nanstd(metric_vals[metric_name]))


  def print(self):
    for metric_name in self.metric_names:
      if metric_name == 'loss':
        print("  " + self.name + " "+ metric_name+":\t{:.5f}".format(self.value[metric_name][-1]))
      else:
        print("  " + self.name + " "+ metric_name+":\t{:.5f}+/-{:.5f}"
                                    .format(self.value[metric_name][-1], 
                                            self.value[metric_name+'_std'][-1]))



