from pydantic import BaseModel
from typing import List
import os

class Settings(BaseModel):
    cors_origins: List[str] = ["*"]
    saving_path: str = "results"
    img_height: int = 256
    img_width: int = 256
    num_classes: int = 11
    model_path: str = "model.h5"
    colormap_path: str = "ultimate.mat"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"