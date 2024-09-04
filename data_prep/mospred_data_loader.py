import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
from utils.logger import logger
from torch.utils.data import Dataset

LANG_ID_MAPPING = {
    "hi": 0,
    "te": 1,
    "mr": 2,
    "kn": 3,
    "bn": 4,
    "en": 5,
    "ch": 6,
    "Hindi_M_INDICTTS": 0,
    "Hindi_F_INDICTTS": 0,
    "Hindi_F_SPIRE": 0,
    "Kannada_M_INDICTTS": 3,
    "Kannada_F_INDICTTS": 3,
    "Kannada_F_SPIRE": 3,
    "English_F_VCTK248": 5,
    "English_M_VCTK326": 5,
    "English_F_VCTK294": 5,    
}

TASK_ID_MAPPING = {
    "LIMMITS23": 0,
    "LIMMITS24": 1
}

class MOSPRED_DATASET(Dataset):
    """
    Class to load the MOSPRED dataset
    """
    def __init__(self, mode, args):
        """
        Initialize the dataset with all the metadata, audio files and MOS
        """
        self.args = args
        all_audio_files, all_scores, all_cer, all_lang, all_mc, all_task = {}, {}, {}, {}, {}, {}
        if mode == "train": 
            for x in args.data.train_dataset:
                manifest_file = os.path.join(args.data.data_path, x, args.data.train_filename)
                audio_folder = os.path.join(args.data.data_path, x, args.data.wav_folder)
                audio_files = {Path(f).stem: os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith('.wav')}
                all_audio_files = {**all_audio_files, **audio_files}
                scores, cer_lines, langs, mc_info = self.process(manifest_file, audio_files)
                task = {k:TASK_ID_MAPPING[x] for k in scores}
                all_scores = {**all_scores, **scores}
                all_cer = {**all_cer, **cer_lines}
                all_lang = {**all_lang, **langs}
                all_mc = {**all_mc, **mc_info}
                all_task = {**all_task, **task}
                
        elif mode == "dev":
            for x in args.data.train_dataset:
                manifest_file = os.path.join(args.data.data_path, x, args.data.dev_filename)
                audio_folder = os.path.join(args.data.data_path, x, args.data.wav_folder)
                audio_files = {Path(f).stem: os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith('.wav')}
                all_audio_files = {**all_audio_files, **audio_files}
                scores, cer_lines, langs, mc_info = self.process(manifest_file, audio_files)
                task = {k:TASK_ID_MAPPING[x] for k in scores}
                all_scores = {**all_scores, **scores}
                all_cer = {**all_cer, **cer_lines}
                all_lang = {**all_lang, **langs}
                all_mc = {**all_mc, **mc_info}
                all_task = {**all_task, **task}
        else:
            assert mode in ["LIMMITS23", "LIMMITS24"]
            manifest_file = os.path.join(args.data.data_path, mode, args.data.test_filename)
            audio_folder = os.path.join(args.data.data_path, mode, args.data.wav_folder)
            audio_files = {Path(f).stem: os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith('.wav')}
            scores, cer_lines, langs, mc_info = self.process(manifest_file, audio_files, mode="test")
            task = {k:TASK_ID_MAPPING[mode] for k in scores}
            all_scores = {**all_scores, **scores}
            all_cer = {**all_cer, **cer_lines}
            all_lang = {**all_lang, **langs}
            all_mc = {**all_mc, **mc_info}
            all_task = {**all_task, **task}
            all_audio_files = {**all_audio_files, **audio_files}
            
        self.scores = all_scores
        self.cers = all_cer
        self.langs = all_lang
        self.mc = all_mc
        self.tasks = all_task
        self.audio_files = all_audio_files
        logger.info(f'{mode} - {len(self.scores)} files, {len(self.audio_files)} audio files, {len(self.cers)} cer files')
        self.all_keys = list(self.scores.keys())
        
        self.all_keys = [k for k in self.all_keys if k in self.cers]
        
        
    def process(self, manifest, audios, mode="train"):
        """
        Read the manifest file and get the scores
        """
        with open(manifest, 'r') as f:
            manifest_lines = f.read().split('\n')
        if self.args.data.exclude_lang and mode == "train":    
            lang = {l.split("\t")[0]:LANG_ID_MAPPING[l.split("\t")[1]] for l in manifest_lines if len(l)>0 if l.split("\t")[0] in audios if l.split("\t")[1] not in self.args.data.exclude_lang_name}
        else:
            lang = {l.split("\t")[0]:LANG_ID_MAPPING[l.split("\t")[1]] for l in manifest_lines if len(l)>0 if l.split("\t")[0] in audios}
        mc = {l.split("\t")[0]:LANG_ID_MAPPING[l.split("\t")[2]] for l in manifest_lines if len(l)>0 if l.split("\t")[0] in audios}
        scores = {l.split("\t")[0]:float(l.split("\t")[3]) for l in manifest_lines if len(l)>0 if l.split("\t")[0] in audios}
        cer_file = self.args.data.cer_path.replace("VERSION", manifest.split('/')[-2][-2:])
        with open(cer_file, 'r') as f:
            cer_lines = f.read().split("\n")
        cer_lines = {l.split("\t")[0]:float(l.split("\t")[1]) for l in cer_lines if len(l)>0 if l.split("\t")[0] in audios}
        
        if self.args.data.exclude_lang:
            mc = {k:mc[k] for k in mc if k in lang}
            scores = {k:scores[k] for k in scores if k in lang}
            cer_lines = {k:cer_lines[k] for k in cer_lines if k in lang}
        
        
        return scores, cer_lines, lang, mc
        
    def __len__(self):
        """
        Length of the dataset
        """
        return len(self.all_keys)

    def __getitem__(self, i):
        """
        Get the item from the dataset
        """
        fname = self.all_keys[i]
        score = self.scores[fname]
        audio = self.audio_files[fname]
        lang = self.langs[fname]
        mc = self.mc[fname]
        task = self.tasks[fname]
        cer = self.cers[fname]
        y, sr = torchaudio.load(audio)
        y = y.squeeze()
        return y, score, cer, lang, mc, task, fname

class MOSPRED_Collate():
    """
    Collate function for the MOSPRED dataset, for audio padding
    """
    def __call__(self, batch):
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        audio_padded = torch.FloatTensor(len(batch), max_input_len)
        audio_padded.zero_()
        scores, cers, filenames, lengths = [], [], [], []
        langs, mcs, tasks = [], [], []
        for i in range(len(ids_sorted_decreasing)):
            audio = batch[ids_sorted_decreasing[i]][0]
            audio_padded[i, :audio.size(0)] = audio
            scores.append(batch[ids_sorted_decreasing[i]][1])
            cers.append(batch[ids_sorted_decreasing[i]][2])
            filenames.append(batch[ids_sorted_decreasing[i]][6])            
            lengths.append(audio.size(0))
            langs.append(batch[ids_sorted_decreasing[i]][3])
            mcs.append(batch[ids_sorted_decreasing[i]][4])
            tasks.append(batch[ids_sorted_decreasing[i]][5])
        scores = torch.from_numpy(np.array(scores))
        cers = torch.from_numpy(np.array(cers))
        lengths = torch.LongTensor(lengths)
        langs = torch.LongTensor(langs)
        mcs = torch.LongTensor(mcs)
        tasks = torch.LongTensor(tasks)
        
        return \
            audio_padded, \
            scores, \
            cers, \
            lengths, \
            langs, \
            mcs, \
            tasks, \
            filenames