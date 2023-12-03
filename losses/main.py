
import torch
import torch.nn as nn
from losses.TextToBert import textToBert_dim96
from pytorch_transformers import  BertModel, BertConfig,BertTokenizer

def textToBert_dim(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased/vocab.txt')
    # s = 'Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all'
    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
    model = BertModel.from_pretrained('bert-base-cased')
    # pooled
    all_layers_all_words, pooled = model(ids)
    # print( pooled)
    net = nn.Sequential( nn.Linear(768, 768),nn.ReLU(),nn.Linear(768, 96))
    output = net(pooled)
    return output
def transpos(text):

    cond = textToBert_dim(text)
    print(cond)

if __name__ == "__main__":
    text = "a dog"
    cond = transpos(text)




