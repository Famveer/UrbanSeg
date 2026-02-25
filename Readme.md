# UrbanSeg

Exploring the PlacePulse 2.0 dataset and analyzing the urban safety perception through visual features. [Paper](https://fmorenovr.github.io/documents/papers/book_chapters/2021_MICAI.pdf).

# Requirements

- **Python**>=3.12

# Installation
```
  pip install -r requirements.txt
```

# Data

Obtain the Place Pulse 2.0 dataset [here](https://drive.google.com/drive/folders/1V1EjMaz-qqSLzMS4f8N-XkUhAUIqLKa7?usp=sharing).

### Data Preparation

* Download images and `pp2_raw_images.zip` and `scores.csv`.  
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

