from .utils import *
from config import RAW_DATASET_ROOT_FOLDER

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from abc import *
from pathlib import Path
import os
import tempfile
import shutil
import pickle

from collections import Counter
import random

# user의 item에 들어있지 않은 item 랜덤으로 선정
def random_neg(seed, k, item_lst, s): # random.Random()씨드고정 # k개 # item 번호 # user 안에 들어가있는 item 목록 
    t = seed.sample(item_lst, k) # item_lst 중 k개 랜덤 선택
    return_k = set(t)-set(s) # 최종 반환될 거 # 먼저 뽑은 t와 user의 item 목록의 차집합
    
    ran_seed = 1234
    while len(return_k) != k: # 반환되어야하는 set의 갯수가 뽑으려는 개수 k와 동일할때까지 반복
        ran_seed+=1
        t = random.Random(ran_seed).sample(item_lst, k-len(return_k)) # 있던거 발견하면 다른 씨드로 변경 # 더 뽑아야하는 개수만큼
        t = set(t)-set(s)-return_k # 새로 뽑은거가 기존 뽑은거와 목록에 안들어있는 것만 되도록
        return_k.update(t) # 반환되야하는 거에 업데이트
        if len(return_k)>k: # 한번에 많이 업데이트 되어서 뽑아야하는 갯수 넘어가면
            return_k = list(return_k)[:k] # 뽑는 갯수까지만 잘라서
    return list(return_k) 

# Noise 추가하는 부분
def addNoise(args, df, N_Aug, p):
    seed = args.dataloader_random_seed
    rng = random.Random(seed)

    smap = {s: i for i, s in enumerate(set(df['sid']))}
    user_group = df.groupby('uid')
    user2items = user_group.progress_apply(lambda d: list(d.sort_values(by='timestamp')['sid']))

    popularity = Counter() # 인기있는 item 목록 뽑기
    for user in sorted(user2items.keys()):
        popularity.update(user2items[user])
    popular_items = sorted(popularity, key=popularity.get, reverse=True)[:args.type_noise_item_size]

    use_start = max(user2items.index) + 1
    noise_u2seq = {}
    for rept in range(0,N_Aug): # 데이터 증강
        for user in user2items.index: # user 키 목록
            val_test = user2items[user][-2:] # vaild와 test 값은 따로 빼고, train만 바꾸기 위해서 
            new_item_lst = user2items[user][:-2]
            noise_u2seq[use_start] = new_item_lst+val_test # 새로운 데이터셋 dict으로 만들기 
            use_start+=1

    # 한 user의 seq 받아서 비율 p만큼의 갯수 뽑기
    # k개의 item 제거하고, 새로운 랜덤 아이템 k개로 대체
    for user in sorted(noise_u2seq.keys()): # user 키 목록 
        if p == None:
            p = 0
        change_k = round(len(noise_u2seq[user])*p) # 반올림해서 뽑아야하는 item의 갯수
        if change_k<1: # 갯수가 0보다 작으면
            change_k = 1 # 1로 대체

        new_item_lst = noise_u2seq[user][change_k:] # user에 해당하는 item 목록 # k개의 아이템을 새로운 item num으로 대체
        if args.type_noise_item == 'all_item': # all item 중 선택하는 경우
            replace_item = random_neg(rng, change_k , list(smap.keys()) , list(set(noise_u2seq[user])) ) # 새로운 item 목록 뽑기
        elif args.type_noise_item == 'popular_item': # popular item 중 선택하는 경우
            replace_item = random_neg(rng, change_k , popular_items , list(set(noise_u2seq[user])) ) # 인기있는 item 중 목록 뽑기
        for it in replace_item:
            index = np.random.randint(0, len(new_item_lst) )
            new_item_lst.insert(index, it) # 뽑은 값들 삽입

        noise_u2seq[user] = new_item_lst # 새로운 데이터셋 dict으로 만들기
    
    noise_u2seq = pd.Series(noise_u2seq)
    user2items = pd.concat([user2items,noise_u2seq]) # 기존 데이터에 합치기

    aug = []
    for i in user2items.keys():
        aug += user2items[i]
    umap = {u: i for i, u in enumerate(user2items.index)}
    smap = {s: i for i, s in enumerate(list(set(aug)))}
    
    #densifying index
    user2items.index = user2items.index.map(umap)
    for i in range(len(umap)):
        user2items[i] = [smap[i] for i in user2items[i]]
    return dict(user2items), umap, smap
    ### Noise 추가하는 부분 끝