import random
import torch


class GraphData(object):

    def __init__(self, data, feat_dim):
        super(GraphData, self).__init__()
        self.data = data
        self.feat_dim = feat_dim
        self.idx = list(range(len(data)))
        self.pos = 0

    def __reset__(self):
        self.pos = 0
        if self.shuffle:
            random.shuffle(self.idx)

    def __len__(self):
        return len(self.data) // self.batch + 1

    def __getitem__(self, idx):
        g = self.data[idx]
        return g.A, g.feas.float(), g.label

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos >= len(self.data):
            self.__reset__()
            raise StopIteration

        cur_idx = self.idx[self.pos: self.pos+self.batch]
        data = [self.__getitem__(idx) for idx in cur_idx]
        self.pos += len(cur_idx)
        gs, hs, labels = map(list, zip(*data))
        return len(gs), gs, hs, torch.LongTensor(labels)

    def loader(self, batch, shuffle, *args):
        self.batch = batch
        self.shuffle = shuffle
        if shuffle:
            random.shuffle(self.idx)
        return self
