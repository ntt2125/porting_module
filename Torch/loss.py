import torch
import torch.nn as nn

# Define batch size, number of classes, and create example data
batch_size = 64
num_classes = 5

# Create random logits (model output before softmax)
# Shape: (batch_size, num_classes) = (64, 5)
logits = torch.nn.functional.sigmoid(torch.randn(batch_size, num_classes))

# Create random targets (ground truth)
# Shape: (batch_size,) = (64,)
targets = torch.randint(0, num_classes, (batch_size,))

# Define the loss function
criterion = nn.BCEWithLogitsLoss()

# Compute the loss
loss = criterion(logits, targets)

print(f"Logits shape: {logits.shape}")
print(f"Targets shape: {targets.shape}")
print(f"Loss: {loss.item()}")