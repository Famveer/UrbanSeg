import torch
from torch.utils.data.dataloader import Sampler

def ShuffleSampler_(trainset, random_state=2147483647):
  g_cpu = torch.Generator()
  g_cpu.manual_seed(random_state)
  indices = torch.randperm(len(trainset), generator=g_cpu).tolist()
  shuffled_dataset = torch.utils.data.Subset(trainset, indices)
  
  return shuffled_dataset

class SequentialSampler(Sampler):
  def __init__(self, data, index_start=0, batch_size=64):
    self.seq = list(range(len(data)))[index_start * batch_size:]

  def __iter__(self):
    return iter(self.seq)

  def __len__(self):
    return len(self.seq)
 
class ShuffleSampler(Sampler):
  def __init__(self, data, index_start=0, batch_size=64, random_state=2147483647):
    g_cpu = torch.Generator()
    g_cpu.manual_seed(random_state)
    indices = torch.randperm(len(data), generator=g_cpu).tolist()
    self.seq = list(indices)[index_start * batch_size:]

  def __iter__(self):
    return iter(self.seq)

  def __len__(self):
    return len(self.seq)
