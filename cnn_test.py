import numpy as np
import tensorflow as tf
from mnist import MNIST

from model import *
import trainer

##############################################################################
# Model

# This model achieves about 98.6% accuracy after 8 epochs. About 98.0% after only 4 epochs.
model = FeedForwardModel([28, 28], 10, [Conv2DComponent(5, num_kernels=20, stride=2),
                                        ActivationComponent(tf.nn.relu),
                                        Conv2DComponent(4, num_kernels=10),
                                        ActivationComponent(tf.nn.relu),
                                        Conv2DComponent(3, num_kernels=5),
                                        ActivationComponent(tf.nn.relu),
                                        DropoutComponent(0.98),
                                        FullyConnectedComponent()])
x, out = model.build()

print
print model # Displays configured model components
print

learning_rate = 0.01

### Other variables used in training/evaluation
loss_fn = lambda out, y: tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(out, y))
optimizer_fn = lambda loss: tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
accuracy_fn = lambda out, y: tf.reduce_mean(tf.cast(tf.nn.in_top_k(out, y, 1), tf.float32))

##############################################################################
# Data loader and setup
print 'Loading images..'
mndata = MNIST('./mnist')
training_data, training_labels = mndata.load_training()
validation_data, validation_labels = mndata.load_testing()
print 'Training images: {}'.format(len(training_data))
print 'Validation images: {}'.format(len(validation_data))

print 'Reshaping images..'
training_data = np.reshape(training_data, [-1, 28, 28])
validation_data = np.reshape(validation_data, [-1, 28, 28])

##############################################################################
# Training

trainer = trainer.SupervisedTrainer(x, out,
                                    loss_display_interval=100,
                                    accuracy_display_interval=100,
                                    show_accuracy=False,
                                    loss_display_starting_iteration=5) # Loss super high at beginning

trainer.train(training_data,
              training_labels,
              validation_data,
              validation_labels,
              loss_fn=loss_fn,
              optimizer_fn=optimizer_fn,
              accuracy_fn=accuracy_fn,
              batch_size=100, # trains in batches of 100 training data points
              validation_set_size=10000, # Evaluate on all 10000 validation data points
              validation_interval=200) # Evaluates validation data every 300 batches