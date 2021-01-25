import tensorflow as tf

@tf.function
def input_grad_batch(input_var, y, model, loss):
  with tf.GradientTape() as tape:
    predictions = model(input_var)
    loss_value = loss(y, predictions)
    return tape.gradient(loss_value, input_var) 
        
        
class PGDAttack():
  def __init__(self, shape, model, loss, learning_rate=0.1, epsilon=0.1, num_steps=10):
    
    self.shape = [None].extend(list(shape[1:]))
    self.model = model
    self.loss = loss
    self.input_var = tf.Variable(tf.zeros(shape), shape=self.shape, dtype=tf.float32, trainable=True)   
    self.learning_rate = tf.Variable(learning_rate, trainable=False)
    self.epsilon = epsilon
    self.num_steps = num_steps

  def generate(self, x, y):
    x_pgd = tf.identity(x)
    for i in range(num_steps):
      self.input_var.assign(x_pgd)
      delta = input_grad_batch(self.input_var, y, self.model, self.loss)
      x_pgd += self.learning_rate*tf.math.sign(delta)      
      x_pgd = tf.clip_by_value(x_pgd, x-self.epsilon, x+self.epsilon) 
      #x = tf.clip_by_value(x, 0, 1) # ensure valid pixel range
    return x_pgd 



class FGSMAttack():
  def __init__(self, shape, model, loss, epsilon=0.1):
    
    self.shape = [None].extend(list(shape[1:]))
    self.model = model
    self.loss = loss
    self.input_var = tf.Variable(tf.zeros(shape), shape=self.shape, dtype=tf.float32, trainable=True)   
    self.epsilon = epsilon

  def generate(self, x, y):
    self.input_var.assign(x)
    delta = input_grad_batch(self.input_var, y, self.model, self.loss)
    return x + self.epsilon*tf.math.sign(delta)



class NoiseAttack():
  def __init__(self, shape, mean=0.0, stddev=0.1):
    self.shape = shape
    self.mean = mean
    self.stddev = stddev

  def generate(self, x):
    return x + tf.random.normal(x.shape, mean=self.mean, stddev=self.stddev)
    








    

