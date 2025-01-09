# We use huggingface accelerate to lauch mulitiple processes to train the model. 
# We use DataLoader with num_workers=2 to load the dataset.
# The dataset is a IterableDataset, which means it is not a normal dataset, and we need to use the `__iter__` method to load the dataset.

# We would like to understand how accelerate works with IterableDataset.

# %pip install accelerate torch

# %CUDA_VISIBLE_DEVICES="1,2" accelerate launch --mixed_precision=no --dynamo_backend=no --num_machines=1 --multi_gpu --num_processes=2 understand_accelerate_dataloader_iterabledataset.py
# %cat process_log_*.txt | grep "Generated input" | awk '{print $3, $4}' | sort | uniq
    # belo is an example output, we see it has 8 lines, because: 2 training processes, each has a dataloader with number_workers=2, each training process has 2 epochs.
    # [Process 3753429]
    # [Process 3753508]
    # [Process 3753509]
    # [Process 3753637]
    # [Process 3753762]
    # [Process 3753825]
    # [Process 3753892]
    # [Process 3753957]


import torch
from torch.utils.data import IterableDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
import os
import logging

# Step 1: Set up logging for each process
def setup_logger():
    process_id = os.getpid()
    log_filename = f'process_log_{process_id}.txt'  # Log file specific to each process
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s [Process %(process)d] %(message)s')
    file_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(file_handler)

    return logger

# Step 2: Define an IterableDataset
class RandomDataset(IterableDataset):
    def __init__(self, num_samples, input_size, logger):
        self.num_samples = num_samples
        self.input_size = input_size
        self.logger = logger
        self.logger.info(f"Initialized RandomDataset with {num_samples} samples and input size {input_size}")

    def __iter__(self):
        for _ in range(self.num_samples):
            # Generate random input and target data
            x = torch.randn(self.input_size)
            y = torch.tensor([1.0 if x.sum() > 0 else 0.0])
            self.logger.info(f"Generated input {x} and target {y}")
            yield x, y

# Step 3: Define a simple model
class SimpleModel(nn.Module):
    def __init__(self, input_size):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))

# Step 4: Define training loop
def train_loop(dataloader, model, loss_fn, optimizer, accelerator, logger):
    model.train()
    for batch in dataloader:
        x, y = batch
        x, y = x.to(accelerator.device), y.to(accelerator.device)

        # Forward pass
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backward pass
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        # Log loss with process info
        logger.info(f"Loss: {loss.item():.4f}")

# Step 5: Set up the main program
def main():
    # Hyperparameters
    num_samples = 5
    input_size = 3
    batch_size = 2
    num_epochs = 2

    # Initialize Accelerator
    accelerator = Accelerator()

    # Set up logger for each process
    logger = setup_logger()

    # Initialize dataset and dataloader
    dataset = RandomDataset(num_samples, input_size, logger)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2)

    # Initialize model, loss, and optimizer
    model = SimpleModel(input_size)
    loss_fn = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Prepare model and dataloader with Accelerator
    model, dataloader, optimizer = accelerator.prepare(model, dataloader, optimizer)

    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        train_loop(dataloader, model, loss_fn, optimizer, accelerator, logger)

if __name__ == "__main__":
    main()
