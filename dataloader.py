from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import json
import os
from utils.utils import load_json
from PIL import Image, ImageFile
import torchvision.transforms as T
import transformers
import pickle
import math
import re

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
# depressed: 10198
# normal: 22236
# np.choice(22236, 10198, replace=False)

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
        l_t = len(np.where(self.label[:cut_split] == 1)[0])
        l_n = len(np.where(self.label[:cut_split] == 0)[0])
        print('true : {}, false : {}'.format(l_t, l_n))

        self.pic_path = '../data/pic'
        self.transform = transform
        self.bs = args.batch_size

    def __getitem__(self, index):
        participant = self.data[index]
        nickname = participant['nickname']
        tweets = participant['tweets']
        words = []
        imgs = []
        img_path = os.path.join(self.pic_path, nickname)
        idx = math.ceil((index + 1) / self.bs) * self.bs - 1
        idx = idx if idx < len(self.data) else len(self.data) - 1  # 防止溢出
        counts = self.count[idx] if self.count[idx] <= 10 else 10
        count = 0
        for tweet in tweets:
            if tweet['tweet_is_original'] == 'True' and tweet['tweet_content'] != '无' and tweet['tweet_content'] != '' and len(tweet['tweet_content']) != 1 and count < counts:
                count += 1
                words.append(self.filter_emoji(tweet['tweet_content']))
                pic_url = tweet['posted_picture_url']
                if pic_url == '无' or pic_url == None:
                    imgs.append(torch.zeros(3, 224, 224))
                else:
                    # get_image用到了循环，可能有多个图片？有的话此处直接只取第一张图片
                    if isinstance(pic_url, str):
                        tmp = []
                        tmp.append(pic_url)
                        pic_url = tmp
                    img_name = pic_url[0].split('/')[-1]
                    imgs.append(self.load_image(
                        os.path.join(img_path, img_name)))
        if count < counts:
            for i in range(counts - count):
                words.append(' ')
                imgs.append(torch.zeros(3, 224, 224))
        imgs = torch.stack(imgs)
        label = self.label[index]
        return (words, imgs, label, count)

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
        # 过滤表情
        try:
            co = re.compile(u'[\U00010000-\U0010ffff]')
        except re.error:
            co = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
        return co.sub(restr, desstr)


def get_dataloader(args, split):
    dataset = SinaDataset(args, split, transform=train_transform)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'hfl/chinese-xlnet-base')  # 或者bert
    '''
    为了多batch size而补零
    我的思路是：
    文本---首先将所有的sentence放在一个列表，用tokenizer自动补零 + 得到attention mask
    根据以前的长度reshape
    图像---直接stack
    '''

    # batch_size = 1的简单版本
    def collate_fn(batch):
        text = [sample[0] for sample in batch]
        text_sum = np.concatenate(text).tolist()
        token = tokenizer(text_sum, return_tensors="pt",
                          padding=True, truncation=True, max_length=256)
        att_mask = token.attention_mask
        token_ids = token['input_ids']
        att_mask = att_mask.reshape(len(text), -1, att_mask.shape[-1])
        token_ids = token_ids.reshape(len(text), -1, token_ids.shape[-1])

        img = [sample[1] for sample in batch]
        img = torch.stack(img)
        label = [sample[2] for sample in batch]
        # img = img[0]
        count = [sample[3] for sample in batch]
        return token_ids, att_mask, img, torch.LongTensor(label)

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
