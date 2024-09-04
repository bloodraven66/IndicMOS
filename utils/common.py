""" 
Common functions used across the codebase
"""

import os
import yaml
import s3prl.hub as hub
from pathlib import Path
from attrdict import AttrDict
from trainer import ssl_trainer
from utils.logger import logger
from models import base_ssl_regressor
from data_prep import mospred_data_handler

def load_config(yamlFile):
    """
    Load a yaml file and return it as a dictionary
    """
    with open(yamlFile) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        cfg = AttrDict(config)
    return cfg

def print_setup(args):
    """
    Print the setup of the model
    """
    logger.info(f'[train - {args.data.train_dataset}]')
    logger.info(f'[test - {args.data.test_dataset}]')
    logger.info(f'[model - {args.model}]')
    logger.info(f'[mconfig - {args.model_config}]')
    logger.info("========")
    
def load_mode(args):
    """
    Load the model, trainer and data loaders
    """
    print_setup(args)
    loaders = mospred_data_handler.loaders(args)
    trainer = get_trainer(args)
    model, args = get_model(args)
    args = AttrDict(args)
    return loaders, trainer, model, args

def get_model(args, custom_config=False, custom_args=False):
    """
    Get the model based on the arguments
    """
    model_args = load_config(args.model_config)
    if args.model in ["wav2vec2_custom", "hf_wav2vec2_custom"]:
        ssl_model = getattr(hub, args.model)(ckpt=args.path_or_url, weighted_sum=args.weighted_sum)
    else:
        ssl_model = getattr(hub, args.model)(weighted_sum=args.weighted_sum)
    combined_args = AttrDict({**model_args, **args})
    feat_dim = get_feat_model(args)
    model = base_ssl_regressor.ssl_mospred_model(
        ssl_model=ssl_model, 
        args=combined_args, 
        feat_dim=feat_dim, 
        **model_args
    )
    return model, args

def get_feat_model(args):
    """
    Define the feature dimension based on the model
    """
    feat_dims = {
        "wav2vec2": 768,
        "xls_r_300m": 1024,
        "wav2vec2_custom": {
            "indicw2v_large_pretrained": 1024,
            "indicw2v_base_pretrained": 768,
            },
    }
    assert args.model in feat_dims.keys(), f"Model {args.model} not supported"
    return feat_dims[args.model] if args.model != "wav2vec2_custom" else feat_dims[args.model][args.path_or_url.split("/")[-1].split(".")[0]]

def get_trainer(args):
    """
    Get the trainer based on the arguments
    """
    trainer = ssl_trainer
    return trainer

def get_chk_name(args):
    """
    Generate the checkpoint name based on the arguments
    """
    chk_name = f'{args.model}_modelconfig-{args.model_config.split("/")[-1].split(".")[0]}_train-{"".join(args.data.train_dataset)}.pt'
    if args.model == "wav2vec2_custom":
        chk_name = f'{chk_name[:-3]}_path-{args.path_or_url.split("/")[-1].split(".")[0]}.pt'
    if args.weighted_sum:
        chk_name = chk_name.replace(".pt", "_ws.pt")
    if args.use_cer:
        chk_name = chk_name.replace(".pt", "_cer.pt")
    if args.use_lang:
        chk_name = chk_name.replace(".pt", "_lang.pt")
    if args.use_mc:
        chk_name = chk_name.replace(".pt", "_mc.pt")
    if args.use_task:
        chk_name = chk_name.replace(".pt", "_task.pt")
    if args.data.exclude_lang:
        chk_name = chk_name.replace(".pt", f"_minus-{args.data.exclude_lang_name}.pt")
    logger.info(f'Chk: {chk_name}')
    return os.path.join(args.chk_folder, chk_name)

def get_files(path: Path, extension='.wav'):
    """
    Get all the files in a directory with a specific extension
    """
    path = path.expanduser().resolve()
    return list(path.rglob(f'*{extension}'))
