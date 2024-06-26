import os
import torch
import random
import numpy as np
from transformers import BertTokenizer
from tqdm import tqdm
import pickle

class DataLoader(object):
    def __init__(self, params, token_pad_idx=0, tag_pad_idx=-1):
        self.batch_size = params.batch_size
        self.max_len = params.max_len
        self.device = params.device
        self.token_pad_idx = token_pad_idx
        self.tag_pad_idx = tag_pad_idx

        tags = self.load_tags()
        self.tag2idx = {tag: idx for idx, tag in enumerate(tags)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(tags)}
        params.tag2idx = self.tag2idx
        params.idx2tag = self.idx2tag

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", do_lower_case=False)


    def load_tags(self):
        tags = []
        file_path = "data/tags.txt"
        with open(file_path, 'r') as file:
            for tag in file:
                tags.append(tag.strip())
        return tags

    def load_sentences_tags(self, data_type, sentences_file, tags_file, d):
        if os.path.exists(f"data/{data_type}.pkl"):
            with open(f"data/{data_type}.pkl", 'rb') as file:
                return pickle.load(file)
        """
        将句子和标签分别存储在两个列表中并整理为字典，此步骤耗时较长，故将结果存储在pkl文件中
        """
        sentences = []
        tags = []
                    
        with open(sentences_file, 'r') as file:
            for line in tqdm(file):
                # replace each token by its index
                tokens = line.strip().split(' ')
                words = list(map(self.tokenizer.tokenize, tokens))
                word_lengths = list(map(len, words))
                words = ['[CLS]'] + [item for indices in words for item in indices]
                token_start_idxs = 1 + np.cumsum([0] + word_lengths[:-1])
                sentences.append((self.tokenizer.convert_tokens_to_ids(words),token_start_idxs))
        if tags_file != None:
            with open(tags_file, 'r') as file:
                for line in file:
                    # replace each tag by its index
                    tag_seq = [self.tag2idx.get(tag) for tag in line.strip().split(' ')]
                    tags.append(tag_seq)

            # checks to ensure there is a tag for each token
            assert len(sentences) == len(tags)
            for i in range(len(sentences)):
                assert len(tags[i]) == len(sentences[i][-1])

            d['tags'] = tags

        # 句子列表
        d['data'] = sentences
        # 句子数量
        d['size'] = len(sentences)
        pickle.dump(d, open(f"data/{data_type}.pkl", 'wb'))
        return d

    def load_data(self, data_type):
        """加载指定类型的数据。

        参数：
            data_type: (str) 可以是 'train'、'dev' 或 'test'。
        返回：
            data: (dict) 包含每种类型的数据及其对应的标签。
        """
        data = {}
        
        print('加载 ' + data_type)
        sentences_file = f"data/{data_type}.txt"
        tags_path = f"data/{data_type}_TAG.txt" if not data_type.endswith("test") else None
        assert tags_path == None
        data = self.load_sentences_tags(data_type, sentences_file, tags_path, data)
        return data

    def data_iterator(self, data, shuffle=False):
        """返回一个生成器，该生成器会产生批量数据和标签。

        参数：
            data：（字典）包含键 'data'、'tags' 和 'size' 的数据
            shuffle：（布尔值）是否应该对数据进行洗牌
            
        生成器：
            batch_data：（张量）形状：（batch_size，max_len）
            batch_tags：（张量）形状：（batch_size，max_len）
        """

        # 创建一个列表，决定遍历数据的顺序，避免了显式地对数据进行shuffle
        order = list(range(data['size']))
        if shuffle:
            random.seed(2021213356)
            random.shuffle(order)
        
        interMode = False if 'tags' in data else True
        
        # 计算batch数量
        if data['size'] % self.batch_size == 0:
            BATCH_NUM = data['size']//self.batch_size
        else:
            BATCH_NUM = data['size']//self.batch_size + 1


        for i in range(BATCH_NUM):
            # 剩余句子数量如果不足一个batch，则取到最后一个句子
            if i * self.batch_size < data['size'] < (i+1) * self.batch_size:
                sentences = [data['data'][idx] for idx in order[i*self.batch_size:]]
                if not interMode:
                    tags = [data['tags'][idx] for idx in order[i*self.batch_size:]]
            else:
                sentences = [data['data'][idx] for idx in order[i*self.batch_size:(i+1)*self.batch_size]]
                if not interMode:
                    tags = [data['tags'][idx] for idx in order[i*self.batch_size:(i+1)*self.batch_size]]

            # 大小为batch_size或最后一个batch剩余句子数量
            batch_len = len(sentences)

            # 计算每个batch的长度max_words_len
            batch_max_words_len = max([len(s[0]) for s in sentences])
            # print(batch_max_words_len)
            max_words_len = min(batch_max_words_len, self.max_len)
            max_token_len = 0


            # 对于不足该长度的，填充为该长度
            batch_data = self.token_pad_idx * np.ones((batch_len, max_words_len))
            batch_token_starts = []
            
            # 对于每个句子，填充到batch_data中
            for j in range(batch_len):
                cur_words_len = len(sentences[j][0])
                if cur_words_len <= max_words_len:
                    batch_data[j][:cur_words_len] = sentences[j][0]
                else:
                    batch_data[j] = sentences[j][0][:max_words_len]
                token_start_idx = sentences[j][-1]
                token_starts = np.zeros(max_words_len)
                token_starts[[idx for idx in token_start_idx if idx < max_words_len]] = 1
                batch_token_starts.append(token_starts)
                max_token_len = max(int(sum(token_starts)), max_token_len)
            
            if not interMode:
                batch_tags = self.tag_pad_idx * np.ones((batch_len, max_token_len))
                for j in range(batch_len):
                    cur_tags_len = len(tags[j])  
                    if cur_tags_len <= max_token_len:
                        batch_tags[j][:cur_tags_len] = tags[j]
                    else:
                        batch_tags[j] = tags[j][:max_token_len]
            
            # 将batch_data转换为张量
            batch_data = torch.tensor(batch_data, dtype=torch.long)
            batch_token_starts = torch.tensor(batch_token_starts, dtype=torch.long)
            if not interMode:
                batch_tags = torch.tensor(batch_tags, dtype=torch.long)

            # 将batch_data和batch_tags转换为cuda张量
            batch_data, batch_token_starts = batch_data.to(self.device), batch_token_starts.to(self.device)
            if not interMode:
                batch_tags = batch_tags.to(self.device)
                yield batch_data, batch_token_starts, batch_tags
            else:
                yield batch_data, batch_token_starts
