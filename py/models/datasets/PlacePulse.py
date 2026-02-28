import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from collections import Counter
from numpy import moveaxis
from sklearn.model_selection import train_test_split

from .handlers import DataHandler

class PlacePulse(DataHandler):
    def __init__(self, dataframe, 
                       random_state=42, 
                 ):
        super().__init__()
        '''
            Initialize:
            pp = PlacePulse()
            pp.DataPreparation(PROCESSED_PATH, IMAGES_PATH, delta=0.2)
            pp.process()
            pp.DataLoader()
            pp.plot()
        '''
        self.dataframe = dataframe
        self.random_state = random_state
        
    def plot(self, number_images=64, dataset=None, img_to_show="image_path"):
        if dataset == "train":
            data_loaded = self.train_df
        if dataset == "val":
            data_loaded = self.test_df
        else:
            data_loaded = self.test_df
        
        img_id = data_loaded["image_id"][:number_images].tolist()
        
        inputs = data_loaded[img_to_show][:number_images].tolist()
        inputs = [ Image.open(img_path_).resize((384, 384)).convert("RGB") for img_path_ in inputs ]
        labels = data_loaded["target"][:number_images].tolist()
        # Set up the grid for displaying the images
        grid_size = 8  # 8x8 grid for 64 images
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(25, 30))

        # Loop through each image and plot
        for i, ax in enumerate(axes.flat):
            # Get the ith image, permute it to [384, 384, 3] for matplotlib
            #img = np.array(inputs[i])#.permute(1, 2, 0).numpy()
            
            # Normalize the image to [0, 1] for visualization (if needed)
           # img = (img - img.min()) / (img.max() - img.min())
            
            # Plot the image in the grid
            if self.task_type == "classification":
                ax.set_title(f"{img_id[i]}\n{self.label_map[int(labels[i])]}", fontsize=15)
            else:
                ax.set_title(f"{img_id[i]}\n{labels[i]}", fontsize=15)
            ax.imshow(inputs[i])
            ax.axis('off')  # Hide the axis

        # Adjust layout and show the grid of images
        plt.tight_layout()
        plt.show()
    
    def load_img(self, image_path):
        img = Image.open(image_path)

        # Resize the image to target size if needed
        #img = img.resize((224, 224))

        # Convert to RGB if necessary (in case the image has an alpha channel or is grayscale)
        #img = img.convert('RGB')
    
        img = np.array(img)# / 255.0
        img = moveaxis(img, 2, 0)
        return img
    
    # still in process
    def find_n_equal_subsets(self, data, N=2):
        # Split the sorted data into N equal-sized subsets
        subset_size = len(data) // N  # Calculate size of each subset
        subsets = [data[i * subset_size:(i + 1) * subset_size] for i in range(N)]
        return subsets
        
    def find_n_quantiles(self, data, N=2):
        quantiles = np.linspace(0, 1, N + 1)[1:-1]  # Exclude 0 and 1
        return np.quantile(data, quantiles)
        
    def assign_n_labels(self, data, quantile_values):
        labels = np.digitize(data, quantile_values, right=True)
        return labels
        
    def filter_delta(self, df_, metric, delta):
        if delta<0.5:
            top_n = int(len(df_) * delta)
            
            self.top_df = df_.nlargest(top_n, metric)
            self.bot_df = df_.nsmallest(top_n, metric)
            
            print(f"Applying delta {delta}.")
            print(f"Top max: {self.top_df[metric].max()}, min:{self.top_df[metric].min()}, size: {len(self.top_df)}")
            print(f"Bot max: {self.bot_df[metric].max()}, min:{self.bot_df[metric].min()}, size: {len(self.bot_df)}")
            
            delta_df = pd.concat([self.top_df, self.bot_df], axis=0, ignore_index=True)
            delta_df.sort_values(by=[metric], inplace=True,ascending=False)
            return delta_df
        else:
            print(f"Value max: {df_[metric].max()}, min: {df_[metric].min()}, size: {len(df_)}")
            return df_
    
    def filter_mean_std(self, df_, metric, delta):
        u = df_[metric].mean()
        std = df_[metric].std()
        self.superior = u + delta*std
        self.inferior = u - delta*std

        self.top_df = df_[df_[metric] > self.superior].copy()
        self.bot_df = df_[df_[metric] < self.inferior].copy()
            
        print(f"Applying u ± delta*std: {u} ± {delta}*{std}.")
        print(f"Superior: {self.superior}, Inferior: {self.inferior}")
        print(f"Top max: {self.top_df[metric].max()}, min:{self.top_df[metric].min()}, size: {len(self.top_df)}")
        print(f"Bot max: {self.bot_df[metric].max()}, min:{self.bot_df[metric].min()}, size: {len(self.bot_df)}")
        
        delta_df = pd.concat([self.top_df, self.bot_df], axis=0, ignore_index=True)
        delta_df.sort_values(by=[metric], inplace=True,ascending=False)
        return delta_df
        
    def filter_mean_std_(self, x, u, std, metric, delta):
        print("u:", u, "std:", std)
        if x > u + delta*std:
            return 1
        elif u - delta*std > x:
            return 0
        else:
            return -np.inf
    
    def DataPreparation(self, delta=0.5, emotion="safety", city=None, country=None, divide_by="delta"):
        # copy df
        data = self.dataframe.copy()
        # filter country
        if country is not None and len(country)>0 and country!="all":
            data = data[data["country"]==country].copy()
        
        # filter city    
        if city is not None and len(city)>0 and city!="all":
            data = data[data["city"]==city].copy()
        
        # remove 0.0
        data = data[data[emotion]>0].copy()
        
        # sorting by scores
        data.sort_values(emotion, ascending=False, inplace=True)
        
        if divide_by=="std":
            self.divide_by="std"
            data = self.filter_mean_std(data, emotion, delta)
        
        elif divide_by=="delta":
            self.divide_by="delta"
            data = self.filter_delta(data, emotion, delta)
        
        else:
            self.divide_by="delta"
            data = self.filter_delta(data, emotion, delta)
        
        self.emotion = emotion
        self.data_df = data
        
    def TaskPreparation(self, task_type="regression", num_classes=None, categorize_by=None):
    
        # Regression
        if "reg" in task_type.lower() or "regression" in task_type.lower():
            self.task_type = "regression"
            self.data_df["target"] = self.data_df[self.emotion].apply(lambda x: x)
            self.data_df["label"] = self.data_df["target"].apply(lambda x: x)
        else:  # Classification 
            self.task_type = "classification"
    
            if num_classes in [None, 0, 1, 2] and categorize_by is None: # Binary
                if self.divide_by=="std":
                    self.data_df["target"] = self.data_df[self.emotion].apply(lambda x: 1 if x > self.superior else 0 )
                  
                elif self.divide_by=="delta":
                    self.data_df["target"] = self.data_df[self.emotion].apply(lambda x: 1 if x>5.0 else 0)
                    
                else:
                    self.data_df["target"] = self.data_df[self.emotion].apply(lambda x: 1 if x>5.0 else 0)
                
                self.data_df["label"] = self.data_df["target"].apply(lambda x: self.emotion if x==1 else f"not {self.emotion}")
            
            else:  # more than 2 classes 
                if categorize_by=="quantil":
                    quantiles = self.find_n_quantiles(data[emotion].values, N=num_classes)
                    n_labels = self.assign_n_labels(data[emotion].values, quantiles)
                    self.data_df["label"] = n_labels
              
                elif categorize_by=="floor":
                    self.data_df["label"] = self.data_df[self.emotion].apply(lambda x: int(np.floor(x)) )
                    
                elif categorize_by=="ceil":
                    self.data_df["label"] = self.data_df[self.emotion].apply(lambda x: int(np.ceil(x)) )
                    
                else:
                    self.data_df["label"] = self.data_df[self.emotion].apply(lambda x: int(np.floor(x)) )
                
                self.data_df["label"] = self.data_df["label"].apply(lambda x: str(x))
                    
                remaining_labels = list(self.data_df["label"].unique())
                label_mapping = {str(old_label): new_label for new_label, old_label in enumerate(sorted(remaining_labels))}
                self.data_df["target"] = self.data_df["label"].map(label_mapping)
                
            label_map = dict(zip(self.data_df["target"], self.data_df["label"]))
            self.label_map = dict(sorted(label_map.items()))
            
            
    def DataSplit(self, test_size=0.25, randomize_class=False, return_values=False):
    
        if test_size is None or test_size==0:
            self.train_df, self.test_df = self.data_df, self.data_df
            return
    
        if self.task_type == "classification":
            self.train_df, self.test_df = train_test_split(self.data_df, test_size=test_size, stratify=self.data_df["target"], random_state=self.random_state)
        
        else:
            self.train_df, self.test_df = train_test_split(self.data_df, test_size=test_size, random_state=self.random_state)

        if randomize_class:
            self.train_df["target"] = np.random.permutation(self.train_df["target"].values).astype(int)
        
        if return_values:
            return train_df, test_df
        else:
            return
    
    def get_data(self):
        return self.data_df
    
    def DataFormatLIP(self, data_formater, processor, device, randomize_class=False, return_values=False):
    
        self.datasets = {
            'train': data_formater(dataset=self.train_df, processor=processor, device=device),
            'val': data_formater(dataset=self.test_df, processor=processor, device=device)
        }
        
        if return_values:
            return self.datasets
        else:
            return
    
    def DataFormat(self, data_formater, transforms_list=None, **kwargs):
        
        #dataset_train = TensorDataset( Tensor(train_x), Tensor(train_y) )
        #dataset_test = TensorDataset( Tensor(test_x), Tensor(test_y) )
        
        train_transforms = transforms_list.get("train", {}) if transforms_list is not None else {}
        val_transforms   = transforms_list.get("val", {})   if transforms_list is not None else {}

        self.datasets = {
            'train': data_formater(dataset=self.train_df, **train_transforms, **kwargs),
            'val':   data_formater(dataset=self.test_df,  **val_transforms,   **kwargs)
        }
        
        
