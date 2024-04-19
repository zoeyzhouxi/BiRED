import ujson as json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")
class MySet(Dataset):
    def __init__(self, filenames):
        super(MySet, self).__init__()
        self.content = open(filenames).readlines()
       
    def __len__(self):
        # print(len(self.content))
        return len(self.content)

    def __getitem__(self, idx):
        rec = json.loads(self.content[idx])
        return rec

def collate_fn(recs):
    def dict(direct):
        values = torch.FloatTensor(list(map(lambda x: x[direct]['values'], recs)))
        masks = torch.FloatTensor(list(map(lambda x: x[direct]['masks'], recs)))
        deltas = torch.FloatTensor(list(map(lambda r: r[direct]['deltas'], recs)))
        evals = torch.FloatTensor(list(map(lambda r: r[direct]['evals'], recs)))
        eval_masks = torch.FloatTensor(list(map(lambda r: r[direct]['eval_masks'], recs)))       
        forwards = torch.FloatTensor(list(map(lambda r: r[direct]['forwards'], recs)))
        rec = {'values': values, 'masks': masks, 'deltas': deltas, 'evals': evals, 'eval_masks': eval_masks, 'forwards': forwards}
        return rec

    ret_dict = {'forward': dict('forward'), 'backward': dict('backward')}
    ret_dict['labels'] = torch.FloatTensor(list(map(lambda x: x['label'], recs)))

    return ret_dict

def get_loader(batch_size, filenames, shuffle = True):
    train_set = MySet(filenames)
    train_iter = DataLoader(dataset = train_set,
                            batch_size = batch_size,
                            num_workers = 4,
                            shuffle = shuffle,
                            pin_memory = True,
                            collate_fn = collate_fn)
    return train_iter

def get_dataset(filenames):
    train_set = MySet(filenames)
    return train_set