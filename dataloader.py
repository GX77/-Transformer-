# coding:utf8
import torch
from torch.utils import data
import numpy as np
import h5py


def create_collate_fn(pad_index, max_seq_len):
    def collate_fn(dataset):
        ground_truth = {}
        tmp = []
        for fn, caps, att_feat, word_feat in dataset:
            ground_truth[fn] = [c[:max_seq_len] for c in caps]
            for cap in caps:
                tmp.append([fn, cap, att_feat, word_feat])
        dataset = tmp
        dataset.sort(key=lambda p: len(p[1]), reverse=True)
        fns, caps, att_feats, word_feats = zip(*dataset)
        att_feats = torch.FloatTensor(np.array(att_feats))
        small = 100
        temp = []
        for i in word_feats:
            if i.shape[0] < small:
                small = i.shape[0]
            temp.append(list(i))
        w = []
        for i in temp:
            w.append(i[:small])
        # print(small)
        word_feats = torch.FloatTensor(np.array(w))

        # word_feats = torch.FloatTensor(word_feats)
        lengths = [min(len(c), max_seq_len) for c in caps]
        caps_tensor = torch.LongTensor(len(caps), max(lengths)).fill_(pad_index)
        for i, c in enumerate(caps):
            end_cap = lengths[i]
            caps_tensor[i, :end_cap] = torch.LongTensor(c[:end_cap])
        lengths = [l-1 for l in lengths]
        return fns, att_feats, word_feats, (caps_tensor, lengths), ground_truth

    return collate_fn


class CaptionDataset(data.Dataset):
    def __init__(self, att_feats, word_feats, img_captions):
        self.att_feats = att_feats  # 现在还只是一个字符串
        self.word_feats = word_feats
        self.captions = list(img_captions.items())  # {:[]} ———> [(,)]

    def __getitem__(self, index):
        fn, caps = self.captions[index]
        f_att = h5py.File(self.att_feats, 'r')
        f_word = h5py.File(self.word_feats, 'r')
        att_feat = f_att[fn][:]
        word_feat = f_word[fn][:]
        return fn, caps, np.array(att_feat), np.array(word_feat)

    def __len__(self):
        return len(self.captions)


def get_dataloader(att_feats, word_feats, img_captions, pad_index, max_seq_len, batch_size, num_workers=0, shuffle=True):
    dataset = CaptionDataset(att_feats, word_feats, img_captions)
    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 collate_fn=create_collate_fn(pad_index, max_seq_len + 1))  # 处理输入的dataset的封装
    return dataloader