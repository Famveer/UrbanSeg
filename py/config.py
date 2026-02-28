import os
from pydantic_settings import BaseSettings
import ast
from dotenv import load_dotenv
load_dotenv()

class Config(BaseSettings):

    DATA_PATH: str = os.getenv('DATA_PATH')
    MODEL_PATH: str = os.getenv('MODEL_PATH')

    ML_TASK: str = os.getenv('ML_TASK', "classifications")
    ML_TYPE: str = os.getenv('ML_TYPE', "ensemble")
    BATCH_SIZE: int = int(os.getenv('BATH_SIZE', "256"))
    NUM_EPOCHS: int = int(os.getenv('NUM_EPOCHS', "50"))
    
    SCORING_METHOD: str = os.getenv('SCORING_METHOD', "Qscores")
    PERCEPTION_METRIC: str = os.getenv('PERCEPTION_METRIC', "safety")
    PLACE_LEVEL: str = os.getenv('PLACE_LEVEL', "")
    
    DATASET_SEG_NAME: str = os.getenv("DATASET_SEG_NAME", "ADE20k")
    MODEL_SEG_NAME: str = os.getenv('MODEL_SEG_NAME', "DeepLabV3_ResNet101")
    
    CITY_STUDIED: str = os.getenv('CITY_STUDIED', "Boston")
    DELTA: float  = float(os.getenv('DELTA', "0.42"))

    TOP_K_FEATURES: int = int(os.getenv('TOP_K_FEATURES', "15"))
    RANDOM_STATE: int = int(os.getenv('RANDOM_STATE', "42"))
