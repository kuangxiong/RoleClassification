import sys 
import os 
import pickle 

from config import GlobalData 

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

class ModelConfig(GlobalData):
    """
    设置模型的超参数

    Args:
        GlobalData ([class]): [全局文件路径]
    """
    bert_path = os.path.join(BASE_PATH, "bert-base-chinese")
    save_model = os.path.join(BASE_PATH, f"save_model")
    
    vocab_path = os.path.join(bert_path, 'vocab.txt')
    tokenizer_path = os.path.join(bert_path, 'tokenizer.json')
    

