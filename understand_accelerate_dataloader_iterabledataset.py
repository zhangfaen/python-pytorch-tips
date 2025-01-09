# We use huggingface accelerate to lauch mulitiple processes to train the model. 
# We use DataLoader with num_workers=2 to load the dataset.
# The dataset is a IterableDataset, which means it is not a normal dataset, and we need to use the `__iter__` method to load the dataset.

# We would like to understand how accelerate works with IterableDataset.

# %pip install accelerate torch

# %CUDA_VISIBLE_DEVICES="1,2" accelerate launch --mixed_precision=no --dynamo_backend=no --num_machines=1 --multi_gpu --num_processes=2 understand_accelerate_dataloader_iterabledataset.py
# Example output.
# There are 2 log files, they maybe like:
# process_log_2713962.txt
# process_log_2713963.txt
# %cat process_log_2713962
    # 2025-01-09 22:43:35,311 [Process 2713962] Initialized RandomDataset with 7 samples and input size 3
    # 2025-01-09 22:43:37,246 [Process 2713962] Epoch 1/1
    # 2025-01-09 22:43:37,295 [Process 2714238] Generated input tensor([0.1305, 2.2071, 0.9696]) and target tensor([1.])
    # 2025-01-09 22:43:37,296 [Process 2714238] Generated input tensor([ 3.2712, -0.5973,  1.3487]) and target tensor([1.])
    # 2025-01-09 22:43:37,296 [Process 2714109] Generated input tensor([ 1.0111, -0.2302,  1.0533]) and target tensor([1.])
    # 2025-01-09 22:43:37,297 [Process 2714109] Generated input tensor([-0.0854, -1.2558, -1.1454]) and target tensor([0.])
    # 2025-01-09 22:43:37,299 [Process 2714238] Generated input tensor([-1.2963,  2.0268,  0.9641]) and target tensor([1.])
    # 2025-01-09 22:43:37,300 [Process 2714238] Generated input tensor([-1.6586, -0.4965, -0.1071]) and target tensor([0.])
    # 2025-01-09 22:43:37,300 [Process 2714109] Generated input tensor([1.8369, 0.9696, 0.2193]) and target tensor([1.])
    # 2025-01-09 22:43:37,301 [Process 2714109] Generated input tensor([-0.6410,  0.3606, -0.7900]) and target tensor([0.])
    # 2025-01-09 22:43:37,303 [Process 2714109] Generated input tensor([-0.1476, -0.2769,  1.4433]) and target tensor([1.])
    # 2025-01-09 22:43:37,304 [Process 2714238] Generated input tensor([-0.3036, -0.6777, -0.9838]) and target tensor([0.])
    # 2025-01-09 22:43:37,304 [Process 2714109] Generated input tensor([ 0.4820, -1.6041,  1.1559]) and target tensor([1.])
    # 2025-01-09 22:43:37,304 [Process 2714238] Generated input tensor([ 0.9129,  0.7994, -0.9876]) and target tensor([1.])
    # 2025-01-09 22:43:37,324 [Process 2714238] Generated input tensor([-2.9103, -2.0075, -1.3449]) and target tensor([0.])
    # 2025-01-09 22:43:37,324 [Process 2714109] Generated input tensor([-0.8961,  0.6508,  2.1183]) and target tensor([1.])
    # 2025-01-09 22:43:37,336 [Process 2713962] processing data generated from process tensor([2714109, 2714109], device='cuda:0')
    # 2025-01-09 22:43:37,513 [Process 2713962] processing data generated from process tensor([2714109, 2714109], device='cuda:0')
    # 2025-01-09 22:43:37,549 [Process 2713962] processing data generated from process tensor([2714109, 2714109], device='cuda:0')
    # 2025-01-09 22:43:37,610 [Process 2713962] processing data generated from process tensor([2714109], device='cuda:0')

# %cat process_log_2713963.txt
    # 2025-01-09 22:43:35,703 [Process 2713963] Initialized RandomDataset with 7 samples and input size 3
    # 2025-01-09 22:43:37,239 [Process 2713963] Epoch 1/1
    # 2025-01-09 22:43:37,269 [Process 2714045] Generated input tensor([ 1.1126, -1.0406, -0.0127]) and target tensor([1.])
    # 2025-01-09 22:43:37,270 [Process 2714045] Generated input tensor([ 1.1735,  0.6834, -1.1201]) and target tensor([1.])
    # 2025-01-09 22:43:37,272 [Process 2714045] Generated input tensor([-0.1060,  0.7487,  1.0431]) and target tensor([1.])
    # 2025-01-09 22:43:37,273 [Process 2714045] Generated input tensor([ 1.4460,  0.7440, -0.0254]) and target tensor([1.])
    # 2025-01-09 22:43:37,289 [Process 2714108] Generated input tensor([0.4646, 0.6436, 1.3351]) and target tensor([1.])
    # 2025-01-09 22:43:37,290 [Process 2714108] Generated input tensor([-0.8718, -0.1078,  1.0067]) and target tensor([1.])
    # 2025-01-09 22:43:37,293 [Process 2714108] Generated input tensor([ 1.1532, -0.5849, -0.3468]) and target tensor([1.])
    # 2025-01-09 22:43:37,294 [Process 2714108] Generated input tensor([ 0.6375, -1.2252, -1.5888]) and target tensor([0.])
    # 2025-01-09 22:43:37,363 [Process 2713963] processing data generated from process tensor([2714238, 2714238], device='cuda:1')
    # 2025-01-09 22:43:37,524 [Process 2713963] processing data generated from process tensor([2714238, 2714238], device='cuda:1')
    # 2025-01-09 22:43:37,559 [Process 2713963] processing data generated from process tensor([2714238, 2714238], device='cuda:1')
    # 2025-01-09 22:43:37,622 [Process 2713963] processing data generated from process tensor([2714238], device='cuda:1')
# What we can get from above 2 log files?
# 1. Each training process launched by accelerate has its own log file.
# 2. prefetch_factor of dataloader is 2. Number of batches loaded in advance by each worker.
#    2 means there will be a total of 2 * num_workers batches prefetched across all workers.
# 3. ONLY one dataloader uses its embeded 2 num_workers to load dataset, 
#    the other dataloader in another training process get data from the first dataloader without load data from dataset. 

# 原因是accelerator会将dataloader封装成 *DataLoaderDispatcher*, 而这个封装只使用process 0来迭代和分发数据。
# 其它process会初始化dataset的迭代器却不使用他们产生的数据，会导致空占内存，特别是shuffle操作会占用较大的内存。
# 具体参考`accelerator::DataLoaderDispatcher`的实现：`https://github.com/huggingface/accelerate/blob/54370d450406c679f9585c6f28e1f217a10af093/src/accelerate/data_loader.py#L682`

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
            print(os.getpid())
            yield x, y, os.getpid()

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
        x, y, pid = batch
        x, y = x.to(accelerator.device), y.to(accelerator.device)

        logger.info(f"processing data generated from process {pid}")

        # Forward pass
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backward pass
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        # Log loss with process info
        # logger.info(f"Loss: {loss.item():.4f}")

# Step 5: Set up the main program
def main():
    # Hyperparameters
    num_samples = 7
    input_size = 3
    batch_size = 2
    num_epochs = 1

    # Initialize Accelerator
    accelerator = Accelerator()

    # Set up logger for each process
    logger = setup_logger()

    # Initialize dataset and dataloader
    dataset = RandomDataset(num_samples, input_size, logger)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2, prefetch_factor=2)

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
