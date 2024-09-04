"""
Run the torch audio MOS baseline
"""

#pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/rocm5.7/torchaudio-2.2.0.dev20240102%2Brocm5.7-cp38-cp38-linux_x86_64.whl

import os
import random
import argparse
import torchaudio
from tqdm import tqdm
from pathlib import Path
import torchaudio.functional as F
from torchaudio.pipelines import SQUIM_SUBJECTIVE

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True)
parser.add_argument('--data_path', type=str, default="/home1/Sathvik/MOS_DATASET/LIMMITS")
parser.add_argument('--filename', type=str, default="all_scored_data.txt")
parser.add_argument('--audio_col', type=int, default=5)
parser.add_argument('--reference_path', type=str, default="/data/Database/LJSpeech-1.1/wavs/")
parser.add_argument('--save_path', type=str, default="results/torchaudio_mos")

def load_audio(path, target_rate=16000):
    """
    Load the audio file, resample if necessary
    """
    WAVEFORM, SAMPLE_RATE = torchaudio.load(path)
    if SAMPLE_RATE != target_rate:
        WAVEFORM = F.resample(WAVEFORM, SAMPLE_RATE, target_rate)
    return WAVEFORM

def score_subjective(audios, num_reference=1):
    """
    Score the audios using the subjective model from torchaudio
    """
    assert num_reference == 1
    subjective_model = SQUIM_SUBJECTIVE.get_model()
    reference_files = [os.path.join(args.reference_path, x) for x in os.listdir(args.reference_path)]
    result = {}
    for x in tqdm(audios):
        try:
            audio = load_audio(x)
        except:
            continue
        ref_audio = load_audio(random.choice(reference_files))
        scores = subjective_model(audio, ref_audio)
        id = Path(x).stem
        result[id] = scores.squeeze().item()
    
    with open(os.path.join(args.save_path + "_" + args.version+".txt"), 'w') as f:
        for k, v in result.items():
            f.write("{}\t{}\n".format(k, v))
    
def load_data():
    """
    Load the data from the file
    """
    assert args.version in ['23', '24']
    folder = args.data_path+args.version
    with open(os.path.join(folder, args.filename), 'r') as f:
        data = f.read().split("\n")[:-1]
    audios = [os.path.join(folder, x.split("\t")[args.audio_col]) for x in data]
    return audios

if __name__ == "__main__":
    args = parser.parse_args()
    audios = load_data()
    score_subjective(audios)
        


