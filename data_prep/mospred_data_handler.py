"""
Wrapper to load the data from the MOSPRED dataset class
"""
from . import mospred_data_loader
from torch.utils.data import DataLoader

def loaders(args):
    """
    Get the data loaders
    """
    loaders_ = []
    for mode in ['train', 'dev'] + list(args.data.test_dataset):
        dataset = mospred_data_loader.MOSPRED_DATASET(mode, args)
        collate_fn = mospred_data_loader.MOSPRED_Collate()
        shuffle=True if mode=='train' else False
        loaders_.append(DataLoader(dataset, num_workers=8, shuffle=shuffle,
                          batch_size=args.batch_size, pin_memory=False,
                          drop_last=False, collate_fn=collate_fn))

    return loaders_


