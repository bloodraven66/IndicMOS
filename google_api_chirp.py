"""
Google cloud ASR API (gemini) used to obtain CER
"""
import os
import jiwer
import warnings
import argparse
from tqdm import tqdm
from pathlib import Path
from google.api_core import client_options
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True)
parser.add_argument("--version", required=True)
parser.add_argument("--filename", default="all_scored_data.txt")
parser.add_argument("--data_path", default="/home1/Sathvik/MOS_DATASET/LIMMITS")

def quickstart_v2(audio_file):
    """
    Transcribe a short audio file using synchronous speech recognition
    """
    with open(audio_file, "rb") as f:
        content = f.read()
    request = cloud_speech.RecognizeRequest(
        recognizer=recog_name,
        config=config,
        content=content,
    )
    response = client.recognize(request=request)
    for result in response.results:
        res = result.alternatives[0].transcript
    with open(os.path.join(savefolder, Path(audio_file).stem+".txt"), 'w') as f:
        f.write(res)
    return response
    
def decode(data):
    """
    Decode the audio files using Google Cloud API
    """
    audio_files = [l[1] for l in data]#[:1]
    if not os.path.exists(savefolder): os.mkdir(savefolder)
                            
    files = set(os.listdir(savefolder))
    print(len(files), 'present')

    present = set([Path(l).stem for l in os.listdir(savefolder)])
    files = [l for l in audio_files if Path(l).stem not in present]
    print(len(files), 'start')
    for filename in tqdm(files):
        try:
            quickstart_v2(filename)
        except:
            print(f"{filename} failed")

def infer(text):
    """
    Compute the WER and CER using jiwer
    """
    with open(text, 'r') as f:
        lines = f.read().split('\n')[:-1]
    delimitter = "\t" if len(lines[0].split('\t')) == 2 else " "
    lines = {'_'.join(l.split(delimitter)[0].split('_')[2:]):' '.join(l.split(delimitter)[1:]) for l in lines}
    print(lines)
    pred_dict = {}
    hyps, refs = [], []
    for x in os.listdir(savefolder):
        with open(os.path.join(savefolder, x), 'r') as f:
            line = f.read().split('\n')[0].strip()
        id = Path(x).stem
        pred_dict[id] = line
        if line == '':
            line = 'A'
        refs.append(lines[id])
        hyps.append(line)
        print(refs[-1])
        print(hyps[-1])
        
            
            
            
        
    print(len(lines), len(pred_dict))
    wer = round(jiwer.wer(hyps, refs)*100, 2)
    cer = round(jiwer.cer(hyps, refs)*100, 2)
    print(wer, cer)

def load_data():
    """
    Load the data from the file
    """
    fname = os.path.join(args.data_path+args.version, args.filename)
    data = {}
    with open(fname, 'r') as f:
        lines = f.read().split('\n')[:-1]
    for line in lines:
        id = line.split('\t')[0]
        wavpath = os.path.join(args.data_path+args.version, line.split('\t')[5])
        text = line.split('\t')[6]
        lang = line.split('\t')[1]
        if lang not in data:
            data[lang] = []
        data[lang].append((id, wavpath, text))
    return data
        
if __name__ == "__main__":
    args = parser.parse_args()
    language_code = {
        "bh": "hi-IN",
        "hi": "hi-IN",
        "te": "te-IN",
        "mr": "mr-IN",
        "kn": "kn-IN",
        "en": "en-IN",
        "bn": "bn-IN",
    }
    
    savefolder = ##PLACEHOLDER##
    data = load_data()
    if args.mode == "decode":
        for lang in data:    
            print(lang)
            lang_ = lang
            if lang == "ch":
                lang_ = "hi"
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ##PLACEHOLDER##
            project_id = ##PLACEHOLDER##
            region = ##PLACEHOLDER##
            reg_id = f"{lang_}-gemini"
            recog_name = f"projects/{project_id}/locations/{region}/recognizers/{reg_id}"
            client_options_var = client_options.ClientOptions(
                api_endpoint=f"{region}-speech.googleapis.com"
                )
            client = SpeechClient(client_options=client_options_var)
            
            
            config = cloud_speech.RecognitionConfig(
                    auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
                    language_codes=[language_code[lang_]],
                    model="gemini_asr"
                )
            decode(data[lang])
    elif args.mode == "infer":
        for lang in data:    
            infer(text)
    else:
        raise NotADirectoryError()

