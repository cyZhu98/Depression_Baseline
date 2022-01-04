import json
import numpy as np
from utils import load_json
import pickle
from collections import defaultdict
# depressed: 10198
# normal: 22236
# np.choice(22236, 10198, replace=False)

depressed_json = '../data/depressed.json'
normal_json = '../data/normal.json'

np.random.seed(42)

def sort_by_count(d_file, d_count, n_file, n_count):
    d_dict = defaultdict(list)
    n_dict = defaultdict(list)
    for i in d_file:
        cnt = d_count[i]
        cnt = cnt if cnt <= 10 else 100 # 只截取10组pair
        d_dict[cnt].append(i)
    for i in n_file:
        cnt = n_count[i]
        cnt = cnt if cnt <= 10 else 100 # 只截取10组pair
        n_dict[cnt].append(i)
    

def remove_duplicates(json_file):
    names = []
    choose_idx = []
    counts = []
    for idx, i in enumerate(json_file):
        name = i['nickname']
        tweets = i['tweets']
        count = 0
        if name in names:
            continue
        else:
            names.append(name)  
            for tweet in tweets:
                if tweet['tweet_is_original'] == 'True' and tweet['tweet_content'] != '无' and tweet['tweet_content'] != '' and len(tweet['tweet_content']) != 1:
                    count += 1
            if count > 1:
                choose_idx.append(idx)
                count = count if count <= 10 else 20
                counts.append(count)
    return np.array(choose_idx), np.array(counts)


def split_train_val_test(depressed_json, normal_json):
    depress = load_json(depressed_json)
    normal = load_json(normal_json)
    # 有重复的
    depressed_id, d_count = remove_duplicates(depress)
    normal_id, n_count = remove_duplicates(normal)
    print('Depressed samples before clean : {} , after clean : {}'.format(len(depress), len(depressed_id)))
    print('Normal samples before clean : {} , after clean : {}'.format(len(normal), len(normal_id)))
    total_number = len(depressed_id)
    print('Depressed number is : ', total_number)
    balance_normal_idx = np.random.choice(len(normal_id), total_number, replace=False)
    balance_normal = normal_id[balance_normal_idx]
    n_count = n_count[balance_normal_idx]
    train_num, val_num = int(total_number * 0.7), int(total_number * 0.1)
    test_num = total_number - train_num - val_num
    # np.random.shuffle(depressed_id)
    # np.random.shuffle(balance_normal)
    
    
    index = {'depressed':
        {
            'train': (depressed_id[:train_num], d_count[:train_num]),
            'val': (depressed_id[train_num : train_num + val_num], d_count[train_num : train_num + val_num]),
            'test': (depressed_id[-test_num:], d_count[-test_num:])
        }, 'normal':
            {
                'train': (balance_normal[:train_num], n_count[:train_num]),
                'val': (balance_normal[train_num : train_num + val_num], n_count[train_num : train_num + val_num]),
                'test': (balance_normal[-test_num:], n_count[-test_num:])
            }}
    
    with open('../data/myIndex.pkl', 'wb') as f:
        pickle.dump(index, f)


if __name__ == '__main__':
    split_train_val_test(depressed_json, normal_json)
