import gc
import gzip
import glob
import pickle
from tqdm import tqdm

method = "ptmcmc"
prob_name = "Calcium_Oscillate"
result_dir = f"results/{prob_name}/{method}/"

fnames = glob.glob(result_dir + "*.pkl")
for fname in tqdm(fnames):
    try:
        with open(fname, "rb") as f:
            results = pickle.load(f)
    except pickle.UnpicklingError:
        continue;

    if method == "ptmcmc":
        if not("downsample_step" in results.keys()):
            downsample_step = 100
            results["downsample_step"] = downsample_step
            results["all_samples"] = results["all_samples"][::downsample_step, :, :]
            results["all_weights"] = results["all_weights"][::downsample_step, :]
            results["all_llhs"] = results["all_llhs"][::downsample_step, :]
            results["all_priors"] = results["all_priors"][::downsample_step, :]

    #https://stackoverflow.com/questions/71858937/fastest-and-most-efficient-way-to-save-and-load-a-large-dict
    gc.disable()
    try:
        gc.collect()
        with gzip.open(fname, "w") as fp:
            pickle.dump(results, fp, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        gc.enable()  