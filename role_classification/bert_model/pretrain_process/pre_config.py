import sys 
import os 
import pickle 

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
from role_classification.bert_model.model_config import ModelConfig

class PreModelConfig(ModelConfig):
    """
    设置模型的超参数

    Args:
        GlobalData ([class]): [全局文件路径]
    """

    premodel_save_path = os.path.join(BASE_PATH, "save_premodel")
    train_test_dir = BASE_PATH
    bath_size=64
    save_steps=2000
    n_epochs = 100