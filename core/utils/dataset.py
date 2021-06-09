import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class WordDataset(Dataset):
    """
    读入文本数据的Dataset类
    Dataset类需要实现__init__, __getitem__, __len__三个方法
    """
    def __init__(self, file_list, label_list, sentence_max_size, embedding, word2id, stopwords):
        self.x = file_list
        self.y = label_list
        self.sentence_max_size = sentence_max_size
        self.embedding = embedding
        self.word2id = word2id
        self.stopwords = stopwords

    def __getitem__(self, index):
        words = []
        with open(self.x[index], "r", encoding="utf8") as file:
            for line in file.readlines():
                words.extend(segment(line.strip(), stopwords))
        # 生成文章的词向量矩阵
        vec = generate_tensor(words, self.sentence_max_size, self.embedding, self.word2id)
        return vec, self.y[index]

    def __len__(self):
        return len(self.x)