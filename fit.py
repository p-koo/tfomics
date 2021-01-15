import sys
import tensorflow as tf
from trainer import Trainer


def fit_lr_decay(model, x_train, y_train, validation_data, metrics=['loss', 'auroc', 'aupr'], 
                 num_epochs=100, batch_size=128, shuffle=True, verbose=True, 
                 es_patience=10, es_metric='auroc', 
                 lr_decay=0.3, lr_patience=3, lr_metric='auroc'):

  # get validation data
  x_valid, y_valid = validation_data

  # create tensorflow dataset
  trainset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

  # set up trainer
  trainer = Trainer(model, metrics)
  trainer.set_lr_decay(decay_rate=lr_decay, patience=lr_patience, metric=lr_metric)

  for epoch in range(num_epochs):  
    sys.stdout.write("\rEpoch %d \n"%(epoch+1))

    # train step
    train_loss, pred, y = trainer.train_epoch(trainset, shuffle=shuffle, 
                                              batch_size=batch_size, verbose=verbose)
    trainer.update_metrics('train', train_loss, y, pred, verbose=0)

    # validation performance
    valid_loss, pred = trainer.loss_predict(x_valid, y_valid, batch_size)
    trainer.update_metrics('valid', valid_loss, y_valid, pred, verbose=verbose)

    # check learning rate decay      
    trainer.check_lr_decay(trainer.train_metrics.valid_metric.value[lr_metric][-1])

    # check early stopping
    if trainer.early_stopping(es_metric, patience=es_patience):
      print("Patience ran out... Early stopping.")
      break
