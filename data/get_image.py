import queue
import json
from urllib import request
import os
import pandas as pd
import socket
import threading
socket.setdefaulttimeout(30)
from tqdm import tqdm

def callback(blocknum, blocksize, totalsize):
    '''
    :param blocknum: 已下载数据块
    :param blocksize: 数据块大小
    :param totalsize: 远程文件大小
    :return:
    '''
    percent = 100.0*blocknum*blocksize/totalsize
    if(percent > 100):
        percent = 100
    # print('%.2f%%' % percent)


names = []


def get_pic(url_list):
    for i in tqdm(range(len(url_list))):
        nickname = url_list[i]['nickname']
        if nickname in names:
            continue
        names.append(nickname)
        path_file = os.path.join(path, str(nickname))
        # continue
        if not os.path.exists(path_file):
            os.mkdir(path_file)

        for j in range(len(url_list[i]['tweets'])):
            if(url_list[i]['tweets'][j]['posted_picture_url'] == '无' or url_list[i]['tweets'][j]['tweet_is_original'] == 'False'):
                continue
            else:
                if isinstance(url_list[i]['tweets'][j]['posted_picture_url'], str):
                    url = url_list[i]['tweets'][j]['posted_picture_url']
                    url = url.replace('wap180', 'bmiddle')
                    url = url.replace('large', 'bmiddle')
                    print(url)
                    o = url.split('/')[-1]
                    name = path_file+r'/'+str(o)
                    try:
                        request.urlretrieve(url, name, callback)
                    except:
                        print('error')
                else:
                    for mount in range(len(url_list[i]['tweets'][j]['posted_picture_url'])):
                        url = url_list[i]['tweets'][j]['posted_picture_url'][mount]
                        url = url.replace('wap180', 'bmiddle')
                        url = url.replace('large', 'bmiddle')
                        print(url)
                        o = url.split('/')[-1]
                        name = path_file+r'/'+str(o)
                        try:
                            request.urlretrieve(url, name, callback)
                        except:
                            print('error')


current_dir = '/media/sdc/zhuchenyang/Depression/sina/data'

depressed = os.path.join(current_dir, 'normal.json') # TODO:edit here
with open(depressed, 'r', encoding='utf-8') as f:
    json_data = json.load(f)

names = []
choose_idx = []
for idx, i in enumerate(json_data):
    name = i['nickname']
    if name in names:
        choose_idx.append(idx)
    else:
        names.append(name)
        # choose_idx.append(idx)
print(choose_idx)
print(len(json_data))

path = os.path.join(current_dir, 'pic')

for i in range(len(json_data)):
    nickname = json_data[i]['nickname']
    if nickname in names:
        continue
    names.append(nickname)
    path_file = os.path.join(path, str(nickname))

        # continue
    if not os.path.exists(path_file):
        os.mkdir(path_file)

    for j in range(len(json_data[i]['tweets'])):
        if(json_data[i]['tweets'][j]['posted_picture_url'] == '无' or json_data[i]['tweets'][j]['tweet_is_original'] == 'False'):
            continue
        else:
            for mount in range(len(json_data[i]['tweets'][j]['posted_picture_url'])):
                url = json_data[i]['tweets'][j]['posted_picture_url'][mount]
                url = url.replace('wap180', 'bmiddle')
                url = url.replace('large', 'bmiddle')
                print(url)
                o = url.split('/')[-1]
                name = path_file+r'/'+str(o)
                try:
                    request.urlretrieve(url, name, callback)
                except:
                    print('error')
list1 = list2 = list3 = list4 = list5 = []
along = len(json_data)
for i in range(along):
    if 0 <= i < along/5:
        list1.append(json_data[i])
    elif along/5 <= i < (along*2)/5:
        list2.append(json_data[i])
    elif (along*2)/5 <= i < (along*3)/5:
        list3.append(json_data[i])
    elif (along*3)/5 <= i < (along*4)/5:
        list4.append(json_data[i])
    else:
        list5.append(json_data[i])
t1 = threading.Thread(target=get_pic, args=(list1, ))
t2 = threading.Thread(target=get_pic, args=(list2, ))
t3 = threading.Thread(target=get_pic, args=(list3, ))
t4 = threading.Thread(target=get_pic, args=(list4, ))
t5 = threading.Thread(target=get_pic, args=(list5, ))
t1.start()
t2.start()
t3.start()
t4.start()
t5.start()
t1.join()
t2.join()
t3.join()
t4.join()
t5.join()
print('Exit!')
