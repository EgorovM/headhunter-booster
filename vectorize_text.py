import torch
from transformers import BertTokenizer, BertModel

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
model = BertModel.from_pretrained(
    'DeepPavlov/rubert-base-cased',
    output_hidden_states = True, # Whether the model returns all hidden-states.
)

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

def tokenize_sentence(sentence):
    sentence = " ".join(sentence.split()[:510])
    
    # Add the special tokens.
    marked_text = "[CLS] " + sentence + " [SEP]"

    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    
    return indexed_tokens, segments_ids


def _get_embeddings(indexed_tokens, segments_ids):
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    
        # from all 12 layers. 
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)

        # Evaluating the model will return a different number of objects based on 
        # how it's  configured in the `from_pretrained` call earlier. In this case, 
        # becase we set `output_hidden_states = True`, the third item will be the 
        # hidden states from all layers. See the documentation for more details:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        hidden_states = outputs[2]
        
        # `token_vecs` is a tensor with shape [22 x 768]
    token_vecs = hidden_states[-2][0]

    # Calculate the average of all 22 token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)
    
    return sentence_embedding


def get_embeddings(sentence):
    indexed_tokens, segments_ids = tokenize_sentence(sentence)
    embeddings = _get_embeddings(indexed_tokens, segments_ids)
    
    return embeddings
