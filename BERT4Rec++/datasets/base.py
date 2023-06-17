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

import re
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, args):
        self.args = args
        self.min_rating = args.min_rating
        self.min_uc = args.min_uc
        self.min_sc = args.min_sc
        self.split = args.split
        self.max_len = args.bert_max_len
        self.mat = None
        self.id_2_idx = None
        self.idx_2_id = None
        self.data_type = args.data_type
        self.N_Aug = args.N_Aug
        self.P = args.P
        self.ratio = args.dataset_ratio

        assert self.min_uc >= 2, 'Need at least 2 ratings per user for validation and test'

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @classmethod
    def raw_code(cls):
        return cls.code()

    @classmethod
    @abstractmethod
    def url(cls):
        pass

    @classmethod
    def is_zipfile(cls):
        return True

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return []

    @abstractmethod
    def load_ratings_df(self):
        pass

    def load_dataset(self):
        self.preprocess()
        dataset_path = self._get_preprocessed_dataset_path()
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset

    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        self.maybe_download_raw_dataset()
        df = self.load_ratings_df()
        df = self.partial_data(df, 0.1)
        df = self.make_implicit(df)
        df = self.filter_triplets(df)
        df = self.get_sequence(df)
        if self.data_type == 'similarity':
            print('Simialrity augmentation is processing...')
            self.get_cbf()
            df = self.sim_augmentation(df)
        df, umap, smap = self.densify_index(df)
        train, val, test = self.split_df(df, len(umap))
        dataset = {'train': train,
                   'val': val,
                   'test': test,
                   'umap': umap,
                   'smap': smap}
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def maybe_download_raw_dataset(self):
        folder_path = self._get_rawdata_folder_path()
        if folder_path.is_dir() and\
           all(folder_path.joinpath(filename).is_file() for filename in self.all_raw_file_names()):
            print('Raw data already exists. Skip downloading')
            return
        print("Raw file doesn't exist. Downloading...")
        if self.is_zipfile():
            tmproot = Path(tempfile.mkdtemp())
            tmpzip = tmproot.joinpath('file.zip')
            tmpfolder = tmproot.joinpath('folder')
            download(self.url(), tmpzip)
            unzip(tmpzip, tmpfolder)
            if self.zip_file_content_is_folder():
                tmpfolder = tmpfolder.joinpath(os.listdir(tmpfolder)[0])
            shutil.move(tmpfolder, folder_path)
            shutil.rmtree(tmproot)
            print()
        else:
            tmproot = Path(tempfile.mkdtemp())
            tmpfile = tmproot.joinpath('file')
            download(self.url(), tmpfile)
            folder_path.mkdir(parents=True)
            shutil.move(tmpfile, folder_path.joinpath('ratings.csv'))
            shutil.rmtree(tmproot)
            print()

    def partial_data(self, df):
        if self.ratio == 1:
            return df
        print(f'Partial {self.ratio} data')
        idx = round(df.shape[0] * self.ratio)
        df = df.iloc[:idx]
        return df

    def make_implicit(self, df):
        print('Turning into implicit ratings')
        df = df[df['rating'] >= self.min_rating]
        # return df[['uid', 'sid', 'timestamp']]
        return df

    def filter_triplets(self, df):
        print('Filtering triplets')
        if self.min_sc > 0:
            item_sizes = df.groupby('sid').size()
            good_items = item_sizes.index[item_sizes >= self.min_sc]
            df = df[df['sid'].isin(good_items)]

        if self.min_uc > 0:
            user_sizes = df.groupby('uid').size()
            good_users = user_sizes.index[user_sizes >= self.min_uc]
            df = df[df['uid'].isin(good_users)]
        return df
    
    def get_sequence(self,df):
        user_group = df.groupby('uid')
        user2items = user_group.progress_apply(lambda d: list(d.sort_values(by='timestamp')['sid']))
        return user2items
    
    def densify_index(self, user2items):
        print('Densifying index')
        aug = []
        for i in user2items.index:
            aug += user2items[i]
        aug = list(set(aug))
        umap = {u: i for i, u in enumerate(user2items.index)} 
        smap = {s: i for i, s in enumerate(aug)}
        #densifying index
        user2items.index = umap.values() 
        for i in range(len(umap)):
            user2items[i] = [smap[i] for i in user2items[i]]
        return user2items, umap, smap

    def split_df(self, user2items, user_count):
        if self.args.split == 'leave_one_out':
            print('Splitting')
            train, val, test = {}, {}, {}
            for user in range(user_count):
                items = user2items[user]
                train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
            return train, val, test
        else:
            raise NotImplementedError
        
    def sim_augmentation(self, user2items):
        K = self.N_Aug
        usr_idx_max = max(user2items.index) + 1 #추가할 인덱스
        for usr_idx in list(user2items.index): 
            seq = user2items[usr_idx][-self.max_len:] #max_len만큼의 시퀀스
            replace_num = round(len(seq) * self.P) #시퀀스의 P만큼 증강
            for _ in range(K): #증강규모만큼 반복
                copied_seq = seq.copy() #원본 시퀀스 복사
                choice = random.sample(copied_seq, k=replace_num)
                for c in choice:
                    idx = copied_seq.index(c) #샘플링된 영화의 인덱스
                    sim_item = self.get_movieid(c) #샘플링된 영화의 유사 영화
                    copied_seq[idx] = sim_item
                user2items[usr_idx_max] = copied_seq #새로운 인덱스에 새로운 시퀀스 추가
                usr_idx_max += 1
        return user2items

    def get_cbf(self):
        self.id_2_idx, self.idx_2_id = self.get_vocab()
        self.mat = self.cbf_preprocess()

    def cbf_preprocess(self):
        movie = self.load_contents_df()
        movie['genres'] = movie['genres'].str.split('|')
        movie['year'] = movie['title'].apply(lambda x: x.split(' (')[1])
        movie['year'] = movie['year'].apply(lambda x: x.replace(')','').strip())
        movie['title'] = movie['title'].apply(lambda x: x.split(' (')[0])
        p = re.compile('[^a-zA-Z0-9]')
        movie['title'] = movie['title'].apply(lambda x: p.sub(' ', x))
        movie['title'] = movie['title'].apply(lambda x: word_tokenize(x.lower()))
        stop_words = set(stopwords.words('english'))
        movie['title'] = movie['title'].apply(lambda x: [w for w in x if not w in stop_words])
        movie['title'] = movie['title'].apply(lambda x: ' '.join(x))
        movie['genres'] = movie['genres'].apply(lambda x: ' '.join(x).lower())
        movie['metadata'] = movie['title'] + ' ' + movie['genres'] + ' ' + movie['year']
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(movie['metadata'])
        cosine_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        return cosine_matrix

    def get_movieid(self,movie_id):
        idx = self.id_2_idx[movie_id]
        sim_idx = self.mat[idx].argsort()[::-1][random.randint(0,3)] # Top 3 random
        sim_item = self.idx_2_id[sim_idx]
        return sim_item
    
    def get_vocab(self):
        movie_id = self.load_contents_df()['sid']
        id_2_idx = {}
        idx_2_id = {}
        for i, c in enumerate(movie_id):
            id_2_idx[c] = i
            idx_2_id[id_2_idx[c]] = c
        return id_2_idx, idx_2_id

    def _get_rawdata_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)

    def _get_rawdata_folder_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath(self.raw_code())

    def _get_preprocessed_root_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath('preprocessed')

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        folder_name = '{}_aug_type-{}_N-Aug-{}_p-{}_min_rating{}-min_uc{}-min_sc{}-split{}' \
            .format(self.code(),self.data_type, self.N_Aug, self.P, self.min_rating, self.min_uc, self.min_sc, self.split)
        return preprocessed_root.joinpath(folder_name)

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('dataset.pkl')

