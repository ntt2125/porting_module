import tensorflow as tf
import numpy as np

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define batch size and number of classes
batch_size = 64
num_classes = 5  # Although we're using binary cross-entropy, we'll keep 5 classes for consistency

# Create random logits (model output before sigmoid)
# Shape: (batch_size, num_classes) = (64, 5)
logits = tf.random.normal((batch_size, num_classes))

# Create random targets (ground truth)
# Shape: (batch_size, num_classes) = (64, 5)
targets = tf.random.uniform((batch_size, num_classes), minval=0, maxval=2, dtype=tf.int32)
targets = tf.cast(targets, tf.float32)

print("Logits shape:", logits.shape)
print("Targets shape:", targets.shape)

# 1. Using BinaryCrossentropy with logits
bce_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True)
loss_from_logits = bce_logits(targets, logits)

# 2. Using BinaryCrossentropy with probabilities
probabilities = tf.nn.sigmoid(logits)
bce_probs = tf.keras.losses.BinaryCrossentropy(from_logits=False)
loss_from_probs = bce_probs(targets, probabilities)

print("Loss from logits:", loss_from_logits.numpy())
print("Loss from probabilities:", loss_from_probs.numpy())