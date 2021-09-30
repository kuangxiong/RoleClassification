from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
from role_classification.bert_model.model_config import ModelConfig

tokenizer = BertTokenizer.from_pretrained(ModelConfig.bert_path)
model = TFBertModel.from_pretrained(ModelConfig.bert_path)

inputs = tokenizer("我爱中国", padding='max_length', 
                truncation=True, max_length=10, return_tensors="tf")
print(inputs)
attention_mask = inputs['attention_mask']
print(attention_mask)
outputs = model(inputs, return_dict=True)
print(tf.outputs.last_hidden_state)
# last_hidden_states = outputs.last_hidden_state