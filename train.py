"""
Entry point for training / infering a model

Author: Sathvik Udupa
"""

import os
import argparse
from utils import common


parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument("--config", type=str, default="config/custom_ssl_chk23.yaml", help="Configuration file to use")
parser.add_argument("--infer", default=False, action="store_true")
parser.add_argument("--infer_chk", default=False)
parser.add_argument("--model", default=False)
parser.add_argument("--path_or_url", default=False)
parser.add_argument("--weighted_sum", default=None, action="store_true")
parser.add_argument("--use_cer", default=None, action="store_true")
parser.add_argument("--use_lang", default=None, action="store_true")
args = parser.parse_args()

def main():
    """
    Main function to train the model
    """
    cfg = common.load_config(args.config)
    if args.infer:
        cfg['infer'] = True
    if args.infer_chk:
        cfg['infer_chk'] = args.infer_chk
    if args.model:
        cfg['model'] = args.model
    if args.path_or_url:
        cfg['path_or_url'] = args.path_or_url
    if args.weighted_sum is not None:
        cfg['weighted_sum'] = args.weighted_sum
    if args.use_cer is not None:
        cfg['use_cer'] = args.use_cer
    if args.use_lang is not None:
        cfg['use_lang'] = args.use_lang
    loaders, trainer, model, cfg = common.load_mode(cfg)
    trainer.main(cfg, model, loaders)

if __name__ == "__main__":
    main()

