# UrbanSeg

Exploring the PlacePulse 2.0 dataset and analyzing the urban safety perception through visual features. [Paper](https://fmorenovr.github.io/documents/papers/book_chapters/2021_MICAI.pdf).

# Requirements

- **Python**>=3.9

# Installation

Download the old libgfortran3 .deb from Ubuntu 18.04 and install Fortran:

```
# Download fortran3
wget http://archive.ubuntu.com/ubuntu/pool/universe/g/gcc-6/libgfortran3_6.4.0-17ubuntu1_amd64.deb

# Extract content
dpkg -x libgfortran3_6.4.0-17ubuntu1_amd64.deb /tmp/libgfortran3_extracted

# Copy to conda environment
cp /tmp/libgfortran3_extracted/usr/lib/x86_64-linux-gnu/libgfortran.so.3* ~/anaconda3/envs/ppseg/lib/
```

Then, install the requirements:

```
  pip install -r requirements.txt
```

Then, run (to avoid pytorch error):

```
sed -i 's/raise RuntimeError(msg)/import warnings; warnings.warn(msg)/' ~/anaconda3/envs/ppseg/lib/python3.9/site-packages/gluoncv/check.py
```

# Data

Obtain the Place Pulse 2.0 dataset [here](https://drive.google.com/drive/folders/1V1EjMaz-qqSLzMS4f8N-XkUhAUIqLKa7?usp=sharing).

### Data Preparation

* Download images and `pp2_images.zip` and `scores.csv`.  
* Create a `.env` file, and add the path of the data downloaded and models.  
  ```
    DATA_PATH=/path_to/datasets/
    MODEL_PATH=/path_to/models/
  ```
* First, run the notebook `notebooks/Data/CityScapes_construction.ipynb`.  

* Second, extract features running `notebooks/Models/Features_Extraction.ipynb`.  
  Next, perform classifier with `notebooks/Models/Ensemble_Models.ipynb`.  
  Finally, run explanations at `notebooks/Explanations/XAI.ipynb`.  
  
# Citation

```
@inproceedings{moreno2021urban,
  title={Urban Perception: Can We Understand Why a Street Is Safe?},
  author={Moreno-Vera, Felipe and Lavi, Bahram and Poco, Jorge},
  booktitle={Mexican International Conference on Artificial Intelligence},
  pages={277--288},
  year={2021},
  organization={Springer}
}
```

# Contact us  
For any issue please kindly email to `felipe [dot] moreno [at] fgv [dot] br`

