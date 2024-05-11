from matplotlib.pyplot import figure, cm
import numpy as np
import logging
import argparse
from ridge_utils.stimulus_utils import load_textgrids, load_generic_trfiles
from ridge_utils.dsutils import make_word_ds, make_phoneme_ds
from ridge_utils.dsutils import make_semantic_model
from ridge_utils.SemanticModel import SemanticModel
from ridge_utils.interpdata import lanczosinterp2D
from ridge_utils.util import make_delayed
import os
from ridge_utils import utils1
from ridge_utils.ridge import bootstrap_ridge

from ridge_utils.npp import zscore
import pickle
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "CheXpert NN argparser")
    parser.add_argument("featurename", help="Choose feature", type = str)
    args = parser.parse_args()
    stimul_features = np.load(args.featurename, allow_pickle=True)
    print(stimul_features.item().keys())
    Rstories = ['alternateithicatom', 'avatar', 'howtodraw', 'legacy',
            'life', 'myfirstdaywiththeyankees', 'naked',
            'odetostepfather', 'souls', 'undertheinfluence']

    # Pstories are the test (or Prediction) stories (well, story), which we will use to test our models
    Pstories = ['wheretheressmoke']

    allstories = Rstories + Pstories

    grids = load_textgrids(allstories)

    # Load TRfiles
    trfiles = load_generic_trfiles(allstories)

    # Make word and phoneme datasequences
    wordseqs = make_word_ds(grids, trfiles) # dictionary of {storyname : word DataSequence}
    # Downsample stimuli
    interptype = "lanczos" # filter type
    window = 3 # number of lobes in Lanczos filter
    downsampled_stimuli = dict() # dictionary to hold downsampled stimuli
    for story in allstories:
        downsampled_stimuli[story] = []
        for eachlayer in [8]:
            data = stimul_features.item()[story][eachlayer]
            temp = lanczosinterp2D(data, wordseqs[story].data_times, wordseqs[story].tr_times, window=window )
            downsampled_stimuli[story].append(temp)
    
    trim = 5
    Rstim = {}
    Pstim = {}
    for eachlayer in [8]:
        Rstim[eachlayer] = []
        Rstim[eachlayer].append(np.vstack([zscore(downsampled_stimuli[story][0][5+trim:-trim]) for story in Rstories]))

    for eachlayer in [8]:
        Pstim[eachlayer] = []
        Pstim[eachlayer].append(np.vstack([zscore(downsampled_stimuli[story][0][5+trim:-trim]) for story in Pstories]))
    storylens = [len(downsampled_stimuli[story][0][5+trim:-trim]) for story in Rstories]
    print(storylens)

    ndelays = 8
    delays = range(1, ndelays+1)

    print ("FIR model delays: ", delays)
    print(np.array(Rstim[8]).shape)
    delRstim = []
    for eachlayer in [8]:
        delRstim.append(make_delayed(np.array(Rstim[eachlayer])[0], delays))
        
    delPstim = []
    for eachlayer in [8]:
        delPstim.append(make_delayed(np.array(Pstim[eachlayer])[0], delays))


    # Print the sizes of these matrices
    print ("delRstim shape: ", delRstim[0].shape)
    print ("delPstim shape: ", delPstim[0].shape)
    colab_dict = {"delRstim":delRstim, "delPstim":delPstim}
    with open('./Stimuli/delayed_stim.pickle', 'wb') as pickle_file:
            pickle.dump(colab_dict, pickle_file)
    



