"""
Inference script for IndicMOS

Author: Sathvik Udupa (sathvikudupa66@gmail.com)
"""

import os
import torch
import argparse
import torchaudio
import torch.nn as nn
import s3prl.hub as hub
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser(description="IndicMOS Inference")
parser.add_argument("--audio_path", type=str, required=True, help="Path to the audio file")
parser.add_argument("--use_cer", type=bool, default=False, help="Enable to use CER as an input feature for MOS prediction")
parser.add_argument("--use_language_id", type=bool, default=False, help="Enable to use Language ID as an input feature for MOS prediction")

REPO_ID = "viks66/IndicMOS"
SSL_NAME = "indicw2v_base_pretrained.pt"
BASE_PREDICTOR = "joint_indicw2v_base.pt"
CER_PREDICTOR = "joint_indicw2v_base_cer.pt"
LANG_ID_PREDICTOR = "joint_indicw2v_base_lang.pt"
CER_LANG_ID_PREDICTOR = "joint_indicw2v_base_cer_lang.pt"


class ssl_mospred_model(nn.Module):
    def __init__(
        self, 
        ssl_model,
        dim=768,
        use_cer=False,
        use_lang=False,
        
    ):
        super(ssl_mospred_model, self).__init__()
        self.ssl_model = ssl_model        
        if use_cer:
            dim = 68
        if use_lang:
            dim += 32
        
        self.linear = nn.Linear(dim, 1)
        self.use_cer = use_cer
        if use_cer:
            self.cer_embed = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Linear(32, 4),
                nn.ReLU(),
            )
            self.feat_proj = nn.Sequential(
                nn.ReLU(),
                nn.Linear(dim, 64),
            )
        self.use_lang = use_lang
        if use_lang:
            self.lang_embed = nn.Embedding(7, 32)
    
    def handle_cer_embed(self, feats, cer):
        if not self.use_cer:
            return feats
        feats = self.feat_proj(feats)
        cer = self.cer_embed(cer[:, None])
        feats = torch.cat([feats, cer], -1)
        return feats

    def handle_lang_embed(self, feats, lang):
        if not self.use_lang:
            return feats
        lang = self.lang_embed(lang)
        feats = torch.cat([feats, lang], -1)
        return feats

    def forward(self, x, cer_data=None, lang_data=None):
        feats = self.ssl_model(x)["hidden_states"][-1]
        feats = feats.sum(1)/feats.shape[1]
        feats = self.handle_cer_embed(feats, cer_data)
        feats = self.handle_lang_embed(feats, lang_data)
        feats = self.linear(feats)
        return feats.float()

def download_model_from_hub(chk_name, download_path):
    """
    Download the model from the model repo
    """
    path = hf_hub_download(repo_id=REPO_ID, repo_type="model", filename=chk_name, cache_dir=download_path)
    return path

def load_custom_model_from_s3prl(path):
    """
    Load the custom model from the local s3prl file
    """
    ssl_model = getattr(hub, "wav2vec2_custom")(ckpt=path)
    return ssl_model
    
def load_model(use_cer, use_language_id, download_path, device):
    """
    Load the model from the hub
    """
    if use_cer and use_language_id:
        chk = CER_LANG_ID_PREDICTOR
    elif use_cer:
        chk = CER_PREDICTOR
    elif use_language_id:
        chk = LANG_ID_PREDICTOR
    else:
        chk = BASE_PREDICTOR
    predictor_path = download_model_from_hub(chk, download_path) 
    ssl_path = download_model_from_hub(SSL_NAME, download_path)
    ssl_model = load_custom_model_from_s3prl(ssl_path)
    predictor = torch.load(predictor_path, map_location=device)
    
    mos_model = ssl_mospred_model(ssl_model, use_cer=use_cer, use_lang=use_language_id)
    mos_model.linear.weight.data = predictor["linear.weight"]
    mos_model.linear.bias.data = predictor["linear.bias"]

    if use_cer:
        mos_model.cer_embed[0].weight.data = predictor["cer_embed.0.weight"]
        mos_model.cer_embed[0].bias.data = predictor["cer_embed.0.bias"]
        mos_model.cer_embed[2].weight.data = predictor["cer_embed.2.weight"]
        mos_model.cer_embed[2].bias.data = predictor["cer_embed.2.bias"]
        
        mos_model.feat_proj[1].weight.data = predictor["feat_proj.1.weight"]
        mos_model.feat_proj[1].bias.data = predictor["feat_proj.1.bias"]
        
    if use_language_id:
        mos_model.lang_embed.weight.data = predictor["lang_embed.weight"]
    
    mos_model.to(device)
    return mos_model

def score(audio_path, cer=None, langid=None, use_cer=False, use_language_id=False, download_path="hf_inference_models", device="cpu"):
    """
    Single audio mos prediction
    """
    audio, _ = torchaudio.load(audio_path)
    mos_model = load_model(use_cer, use_language_id, download_path, device)
    score = mos_model(audio, cer_data=cer, lang_data=langid)
    return score

if __name__ == "__main__":
    args = parser.parse_args()
    score = score(audio_path=args.audio_path, use_cer=args.use_cer, use_language_id=args.use_language_id)
    print(score)

    
    