#!/usr/bin/env python
# coding: utf-8

# In[1]:

import json
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

# In[13]:


from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np


# In[2]:
class Data(BaseModel):
    query: str

app = FastAPI()


# In[4]:


model = BertForSequenceClassification.from_pretrained('model_bert/')
tokenizer = BertTokenizer.from_pretrained('model_bert/')


# In[6]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# In[14]:


def get_result(sent):
    encoded_sent = tokenizer.encode_plus(sent,
                                        add_special_tokens=True,
                                        max_length=64,
                                        pad_to_max_length=True,
                                        return_attention_mask=True,
                                        return_tensors='pt')
    input_id = torch.LongTensor(encoded_sent['input_ids']).to(device)
    attention_mask = torch.LongTensor(encoded_sent['attention_mask']).to(device)
    
    with torch.no_grad():
        outputs = model(input_id, token_type_ids=None, attention_mask=attention_mask)
    
    logits = outputs[0]
    probability = np.squeeze(F.softmax(logits)).tolist()[1]
    return probability
    


# In[16]:

@app.post("/items/")
async def main(query: Data):
    print(query, type(query))
    query = str(query)
    prob = get_result(query)
    return {"prob": prob}     


# In[ ]:




