from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import json
import os
from PIL import Image, ImageFile
import torchvision.transforms as T
import transformers
import pickle
import math
from utils.utils import load_json
import re


ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

depressed_json = '../data/depressed.json'
normal_json = '../data/normal.json'
index = '../data/myIndex.pkl'

# 一次次load json太耗时了，先全部都存到内存中
depressed = np.array(load_json(depressed_json))
normal = np.array(load_json(normal_json))
# myIndex = np.load(index, allow_pickle=True).item()
with open(index, 'rb') as f:
    myIndex = pickle.load(f)


train_transform = T.Compose([
    T.Resize((224, 224)),
    # T.RandomCrop((384,384)),
    # T.RandomHorizontalFlip(),
    # T.RandomVerticalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class SinaDataset(Dataset):
    def __init__(self, args, split, transform=None):
        super().__init__()
        depress_ = depressed[myIndex['depressed'][split][0]]
        normal_ = normal[myIndex['normal'][split][0]]
        d_count, n_count = myIndex['depressed'][split][1], myIndex['normal'][split][1]
        self.label_split = len(depress_)
        self.data = np.concatenate((depress_, normal_))
        self.label = [1] * self.label_split + [0] * self.label_split
        self.label = np.array(self.label)
        self.count = np.concatenate((d_count, n_count))
        shuffle_idx = np.arange(2 * self.label_split)
        np.random.shuffle(shuffle_idx)
        # 按照count排序
        ind = np.argsort(self.count[shuffle_idx])
        self.data = np.take_along_axis(self.data[shuffle_idx], ind, 0)  
        self.label = np.take_along_axis(self.label[shuffle_idx], ind, 0)
        self.count = np.take_along_axis(self.count[shuffle_idx], ind, 0)
        
        self.split_ratio = args.split_ratio
        cut_split = int(len(self.data) * self.split_ratio)
        l_t = len(np.where(self.label[:cut_split]==1)[0])
        l_n = len(np.where(self.label[:cut_split]==0)[0])
        print('true : {}, false : {}'.format(l_t, l_n))
        
        self.pic_path = '../data/pic'
        self.transform = transform
        self.bs = args.batch_size
        
    def __getitem__(self, index):
        participant = self.data[index]
        nickname = participant['nickname']
        tweets = participant['tweets']
        words = ''
        idx = math.ceil((index + 1) / self.bs) * self.bs - 1
        idx = idx if idx < len(self.data) else len(self.data) - 1  # 防止溢出
        counts = self.count[idx] if self.count[idx] <= 10 else 10
        count = 0
        for tweet in tweets:
            if tweet['tweet_is_original'] == 'True' and tweet['tweet_content'] != '无' and tweet['tweet_content'] != '' and len(tweet['tweet_content']) != 1 and count < counts:
                count += 1
                words = words + self.filter_emoji(tweet['tweet_content']) + '.'

        label = self.label[index]
        return (words, None, label, count)

    def __len__(self):
        return int(len(self.data) * self.split_ratio)

    def load_image(self, path):
        try:
            image = Image.open(path).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
        except FileNotFoundError:
            image = torch.zeros(3, 224, 224)
        return image
    
    def filter_emoji(self, desstr, restr=''):  
        #过滤表情   
        try:  
            co = re.compile(u'[\U00010000-\U0010ffff]')  
        except re.error:  
            co = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')  
        return co.sub(restr, desstr) 

def get_dataloader(args, split):
    dataset = SinaDataset(args, split, transform=train_transform)
    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     'hfl/chinese-roberta-wwm-ext')  # 或者bert
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'hfl/chinese-xlnet-base')  # 或者bert
    '''
    此loader将文本拉伸至一句话
    '''

    # batch_size = 1的简单版本
    def collate_fn(batch):
        text = [sample[0] for sample in batch]
        token = tokenizer(text, return_tensors="pt",
                          padding=True, truncation=True, max_length=512)
        att_mask = token.attention_mask
        token_ids = token['input_ids']

        # img = img[0]
        label = [sample[2] for sample in batch]
        count = [sample[3] for sample in batch]
        return token_ids, att_mask, None, torch.LongTensor(label)
    
    bs = args.batch_size
    dataloader = DataLoader(
        dataset, batch_size=bs, shuffle=False, collate_fn=collate_fn)
    print('prepare {} dataset with length {}'.format(split, len(dataset)))

    return dataloader


if __name__ == '__main__':
    import argparse

    paser = argparse.ArgumentParser()
    args = paser.parse_args()
    args.batch_size = 4
    args.split_ratio = 0.2
    loader = get_dataloader(args, 'train')
    nums = []
    i = 0
    for i in loader:
        break
    #     nums.append(num.numpy()[0])
    # nums = np.array(nums)
    # print('max: ', np.max(nums))
    # print('min: ', np.min(nums))
    # print('avg: ', np.mean(nums))
    # print('median: ', np.median(nums))
    # nums.sort()
    # print(nums[:50])