"""
Used to generate the correlation results
"""
import os
import argparse
import numpy as np
import scipy.stats
from pathlib import Path
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True)
parser.add_argument('--ref', type=str, default="")
parser.add_argument('--pred', type=str, required=True)
parser.add_argument('--filename', type=str, default="all_scored_data_test.txt")
parser.add_argument('--sentmapping', default="")
parser.add_argument('--mos_col', type=int, default=3)
args = parser.parse_args()

def read_file(path, col, return_monocross=False, return_lang=False, read_pred=False):
    """
    Read the raw data, and return the data in a dictionary
    """
    fname = Path(path).stem
    if fname == "utt2ppl" and read_pred:
        
        with open(path, 'r') as f:
            uttlines = f.read().split('\n')[:-1]
        
        uttlines = {l.split(" ")[0]:5-float(l.split(" ")[1]) for l in uttlines}
        mapping_name = args.sentmapping.replace("VERSION", args.version)
        with open(mapping_name, 'r') as f:
            lines = f.read().split('\n')[:-1]
        
        mapped_data = {}
        for line in lines:
            line = line.split("\t")
            utt = line[0]
            sentid = line[1]
            score = line[2]
            pred_score = uttlines[sentid]
            mapped_data[utt] = pred_score
        print("Converting - ",len(uttlines), len(lines))
        return mapped_data
        
        
    with open(path, 'r') as f:
        lines = f.read().split('\n')[:-1]
    lines_ = {l.split("\t")[0]:float(l.split("\t")[col]) for l in lines}
    if not return_monocross:
        return lines_
    mc = {}
    lang = {}
    for x in lines:
        input_lang = x.split("\t")[1]
        target_lang = x.split("\t")[2]
        
        if args.version == "24":
            if "Hindi" in target_lang:
                target_lang = "hi"
            if "Kannada" in target_lang:
                target_lang = "kn"
            if "English" in target_lang:
                target_lang = "en"
                
        if input_lang == target_lang:
            mc[x.split("\t")[0]] = "mono"
        else:
            mc[x.split("\t")[0]] = "cross"
            
        lang[x.split("\t")[0]] = input_lang
    if return_lang:
        return lines_, mc, lang
    return lines_, mc
    

def plot(ref, pred):
    """
    Plot the reference and predicted values
    """
    plt.figure(figsize=(4, 4))
    plt.scatter(ref, pred, s=1)
    plt.ylabel("Predicted MOS")
    plt.xlabel("Reference MOS")
    plt.title("{0}".format(args.version))
    plt.savefig("plots/{0}_{1}.png".format(args.version, Path(args.pred).stem))


def compute_scores(ref_dict=None, pred_dict=None, ref_mean=None, pred_mean=None, tag=None, tag_sep=",", end="\n", plot=False, plot_name=None):
    """
    Compute all metrics
    """
    if ref_dict is not None:
        system_ref = ref_dict
        system_pred = pred_dict
        system_ref_mean_all, system_pred_mean_all = [], []
        for system in sorted(system_ref.keys()):
            system_ref_mean = np.mean(system_ref[system])
            system_pred_mean = np.mean(system_pred[system]) 
            system_ref_mean_all.append(system_ref_mean)
            system_pred_mean_all.append(system_pred_mean)
        # print(system_ref_mean, system_pred_mean)
    if ref_mean is not None:
        system_ref_mean_all = ref_mean
    if pred_mean is not None:
        system_pred_mean_all = pred_mean
    
    if plot:
        plt.clf()
        plt.plot(system_ref_mean_all, system_pred_mean_all, 'o')
        plt.xlabel("Reference MOS")
        plt.ylabel("Predicted MOS")
        name = plot_name if plot_name is not None else "test.png"
        plt.savefig(name)
    mse = np.mean(np.square(np.array(system_ref_mean_all) - np.array(system_pred_mean_all))).round(4).astype(str)
    lcc = np.corrcoef(system_ref_mean_all, system_pred_mean_all)[0][1].round(4).astype(str)
    srcc = scipy.stats.spearmanr(system_ref_mean_all, system_pred_mean_all)[0].round(4).astype(str)
    ktau = scipy.stats.kendalltau(system_ref_mean_all, system_pred_mean_all)[0].round(4).astype(str)
    if tag is None:
        print(mse+tag_sep+lcc+","+srcc+","+ktau, end=end)
    else:
        print(tag+tag_sep+mse+","+lcc+","+srcc+","+ktau, end=end)
    
    

def get_all_scores():
    """
    Generate scores for all configurations
    """
    args.ref = os.path.join(args.ref + args.version, args.filename)
    pred_files = read_file(args.pred, col=1, read_pred=True)
    ref_files, mc, lang_dict = read_file(args.ref, col=args.mos_col, return_monocross=True, return_lang=True)
    print(len(pred_files), len(ref_files))
    ref_files = {k:ref_files[k] for k in pred_files.keys() if k in ref_files.keys()}
    pred_files = {k:pred_files[k] for k in ref_files.keys()}
    print(len(pred_files), len(ref_files))
    system_ref, system_pred = {}, {}
    utt_ref, utt_pred = {}, {}
    mc_utt_ref, mc_utt_pred = {}, {}
    mc_system_ref, mc_system_pred = {}, {}
    lang_utt_ref, lang_utt_pred = {}, {}
    pred_files_all = [pred_files[k] for k in pred_files.keys()]
    ref_files_all = [ref_files[k] for k in pred_files.keys()]
    plot(ref_files_all, pred_files_all)
    for x in pred_files.keys():
        if args.version == "23":
            uttid = '_'.join(x.split("_")[2:-3])
            system = '_'.join(x.split("_")[:2])
        elif args.version == "24":
            uttid = "-".join(x.split("-")[4:])
            system = '-'.join(x.split("-")[:2])
        
        if system not in system_ref:
            system_ref[system] = []
            system_pred[system] = []
        system_ref[system].append(ref_files[x])
        system_pred[system].append(pred_files[x])
        

        if uttid not in utt_ref:
            utt_ref[uttid] = []
            utt_pred[uttid] = []
        utt_ref[uttid].append(ref_files[x])
        utt_pred[uttid].append(pred_files[x])
        
        monocross = mc[x]
        if monocross not in mc_utt_ref:
            mc_utt_ref[monocross] = []
            mc_utt_pred[monocross] = []
            mc_system_ref[monocross] = {}
            mc_system_pred[monocross] = {}
        # if uttid not in mc_utt_ref[monocross]:
        #     mc_utt_ref[monocross][uttid] = []
        #     mc_utt_pred[monocross][uttid] = []
        if system not in mc_system_ref[monocross]:
            mc_system_ref[monocross][system] = []
            mc_system_pred[monocross][system] = []
            
        mc_system_ref[monocross][system].append(ref_files[x])
        mc_system_pred[monocross][system].append(pred_files[x])    
        
        mc_utt_ref[monocross].append(ref_files[x])
        mc_utt_pred[monocross].append(pred_files[x])
        lang = lang_dict[x]
        if lang not in lang_utt_ref:
            lang_utt_ref[lang] = []
            lang_utt_pred[lang] = []
        # lang_utt_ref[lang][uttid].append(ref_files[x])
        # lang_utt_pred[lang][uttid].append(pred_files[x])
        lang_utt_ref[lang].append(ref_files[x])
        lang_utt_pred[lang].append(pred_files[x])
    
    print("num systems", len(system_ref.keys()))
    print("num_utts", len(ref_files_all))
    
    print("metrics across,MSE,LCC,SRCC,KTAU")
    compute_scores(system_ref, system_pred, tag="system,utt", tag_sep="\t", end=",")
    # compute_scores(utt_ref, utt_pred, tag=None, plot=True) #utt
    compute_scores(ref_mean=ref_files_all, pred_mean=pred_files_all, plot=True)
    # print("overall")
    # compute_scores(ref_mean=ref_files_all, pred_mean=pred_files_all, plot=True)
    for x in mc_utt_ref:
        compute_scores(ref_mean=mc_utt_ref[x], pred_mean=mc_utt_pred[x], tag=f"{x}_utt")
    for x in mc_system_ref:
        compute_scores(ref_dict=mc_system_ref[x], pred_dict=mc_system_pred[x], tag=f"{x}_system")
    for x in lang_utt_ref:
        compute_scores(ref_mean=lang_utt_ref[x], pred_mean=lang_utt_pred[x], tag=f"{x}_utt")
    
    
    

if __name__ == "__main__":
    print(args.version)
    get_all_scores()
    