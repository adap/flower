Use with PyTorch
================
There is a really quick way to integrate flwr-dataset datasets to Pytorch DataLoaders. And the great news is that you can keep all the PyTorch Transform that you used with your dataset downloaded from PyTorch.

Standard setup - download the dataset, choose the partitioning::

  from flwr_datasets import FederatedDataset
  mnist_fds = FederatedDataset(dataset="mnist", partitioners={"train": 100})
  partition_idx_10 = mnist_fds.load_partition(10, "train")
  centralized_dataset = mnist_fds.load_full("test")

Apply Transforms, Create DataLoader::

  from torch.utils.data import DataLoader
  from torchvision.transforms import ToTensor

  transforms = ToTensor()
  partition_idx_10_torch = partition_idx_10.map(
        lambda img: {"img": transforms(img)}, input_columns="img"
    ).with_format("torch")
  dataloader_idx_10 = DataLoader(partition_idx_10_torch, batch_size=16)


You might want to keep the ToTensor() transform (especially if you already used it) because if typically if you use PyTorch model with convolution the number of channels of an image is expected to be on the ? dimention. And the ToTensor() transform besides transforming to Tensor changed switches the dimensions.


If you want to divide the dataset you can use (at any point before passing the dataset to the DataLoader)::

  partition_train_test = partition_idx_10.train_test_split(test_size=0.2)
  partition_train = partition_train_test["train"]
  partition_test = partition_train_test["test"]

Or you can simply calculate the indices yourself::

  partition_len = len(partition_idx_10)
  partition_train = partition_idx_10[:int(0.8 * partition_len)]
  partition_test = partition_idx_10[int(0.8 * partition_len):]

And during the training loop it'll behave slightly different. With typical dataloader you will get a list returned for each iteration::

  for batch in all_from_pytorch_dataloader:
    images, labels = batch
    # Equivalently
    images, labels = batch[0], batch[1]

With this dataset you will get a dictionary, therefore access sample a little bit different::

  for batch in dataloader:
    images, labels = batch["img"], batch["label"]


Do you want just copy-paste the example and change the name of the dataset, click here to go straight to the github.
Or, play around with the dataset yourself in the Google Colab.
If you still have any questions, feel free to join the Slack and ask it.
