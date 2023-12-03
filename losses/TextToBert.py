
from pytorch_transformers import  BertModel, BertConfig,BertTokenizer
import torch
import torch.nn as nn

def textToBert_dim96(text):
    # tokenizer = BertTokenizer.from_pretrained('bert-base-cased/vocab.txt')
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='bert-base-chinese')
    # s = 'Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all'
    tokens = tokenizer.tokenize(text)
    print(tokens)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    # print(tokens)

    ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
    # print(ids.shape)

    model = BertModel.from_pretrained('bert-base-cased')
    # pooled 可以暂时理解为最后一层每一个句子第一个单词[cls]的结果
    all_layers_all_words, pooled = model(ids)
    # print( pooled)
    # print(pooled.shape)
    net = nn.Sequential( nn.Linear(768, 768),nn.ReLU(),nn.Linear(768, 96),nn.Tanh())
    output = (net(pooled))
    return output
