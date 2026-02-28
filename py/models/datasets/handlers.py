import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from .samplers import ShuffleSampler, SequentialSampler

class DataHandler:
        
    def DataLoader(self, batch_size=64, shuffle_train=False, num_workers=0, worker_func=None):
        
        sampler_train = ShuffleSampler(self.datasets["train"], random_state=self.random_state)
        sampler_test = ShuffleSampler(self.datasets["val"], random_state=self.random_state)
        
        if shuffle_train:
          trainloader = DataLoader(self.datasets['train'], batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers, worker_init_fn=worker_func)
        else:
          trainloader = DataLoader(self.datasets['train'], batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_func, sampler=sampler_train)
        
        testloader = DataLoader(self.datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=sampler_test)
        
        self.dataloaders = {
            'train': trainloader,
            'val': testloader
        }
    
    def trainable_samples(self):
        for k,v in self.samples_ids.items():
            for v_ in v.keys():
                print(k, "data", ", class:", v_, ", number of samples:", len(v[v_]))
        
    def DataIdsLoader(self):
        self.samples_ids = {}
        for phase in self.dataloaders.keys():
            print("Getting", phase)
            try:
                list_classes = np.unique(self.datasets[phase].targets).tolist()
            except Exception as e:
                #print("Error:", e)
                list_classes = np.unique(self.datasets[phase].labels).tolist()
            
            # Create a dictionary to hold indices of each class
            label_to_indices = {}
            
            # Populate the dictionary with indices of images belonging to each class
            for batch_idx, (data_phase) in enumerate(self.dataloaders[phase]):
                labels = data_phase[-1]
                for i, label in enumerate(labels.tolist()):
                    if label not in label_to_indices:
                        label_to_indices[label] = []
                    label_to_indices[label].append(batch_idx * len(labels) + i)  # Compute global index
            
            self.samples_ids[phase] = label_to_indices

    def get_unlabeled_samples(self, phase="train", num_samples_per_class=10):
        
        indices_classes = self.samples_ids[phase]
        num_classes = len(indices_classes.keys())
    
        unlabeled_indices = []
        for class_label in range(num_classes):
            selected_indices = np.random.choice(indices_classes[class_label], num_samples_per_class, replace=False)
            unlabeled_indices.extend(selected_indices)
        return unlabeled_indices
        
    def plot(self, dataset=None):
        if dataset in ["train", "val"]: # train or val
            inputs, labels = next(iter(self.dataloaders[dataset]))
        elif dataset is not None:
            inputs = dataset
        else:
            inputs, labels = next(iter(self.dataloaders["val"]))
        # Set up the grid for displaying the images
        if len(inputs)==64:
            fig, axes = plt.subplots(8, 8, figsize=(15, 15))
        elif len(inputs)==32:
            fig, axes = plt.subplots(8, 4, figsize=(15, 15))
        elif len(inputs)==16:
            fig, axes = plt.subplots(4, 4, figsize=(15, 15))
        else:
            fig, axes = plt.subplots(8, 8, figsize=(15, 15))

        # Loop through each image and plot
        for i, ax in enumerate(axes.flat):
            # Get the ith image, permute it to [384, 384, 3] for matplotlib
            img = inputs[i].permute(1, 2, 0).numpy()
            
            # Normalize the image to [0, 1] for visualization (if needed)
            img = (img - img.min()) / (img.max() - img.min())
            
            # Plot the image in the grid
            ax.imshow(img)
            ax.axis('off')  # Hide the axis

        # Adjust layout and show the grid of images
        plt.tight_layout()
        plt.show()
