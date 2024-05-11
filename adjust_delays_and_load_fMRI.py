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

def load_subject_fMRI(subject, modality):
    fdir = './data/'
    fname_tr5 = os.path.join(fdir, 'subject{}_{}_fmri_data_trn.hdf'.format(subject, modality))
    print(fname_tr5)
    trndata5 = utils1.load_data(fname_tr5)
    print(trndata5.keys())

    fname_te5 = os.path.join(fdir, 'subject{}_{}_fmri_data_val.hdf'.format(subject, modality))
    tstdata5 = utils1.load_data(fname_te5)
    print(tstdata5.keys())
    
    trim = 5
    zRresp = np.vstack([zscore(trndata5[story][5+trim:-trim-5]) for story in trndata5.keys()])
    zPresp = np.vstack([zscore(tstdata5[story][0][5+trim:-trim-5]) for story in tstdata5.keys()])
    
    return zRresp, zPresp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "CheXpert NN argparser")
    parser.add_argument("subjectNum", help="Choose subject", type = int)
    parser.add_argument("featurename", help="Choose feature", type = str)
    parser.add_argument("modality", help="Choose modality", type = str)
    parser.add_argument("dirname", help="Choose Directory", type = str)
    parser.add_argument("numlayers", help="Number of Layers", type = int)
    args = parser.parse_args()
    stimul_features = np.load(args.featurename, allow_pickle=True)
    num_layers = args.numlayers
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
        Rstim[eachlayer].append(np.vstack([zscore(downsampled_stimuli[story][eachlayer][5+trim:-trim]) for story in Rstories]))

    for eachlayer in [8]:
        Pstim[eachlayer] = []
        Pstim[eachlayer].append(np.vstack([zscore(downsampled_stimuli[story][eachlayer][5+trim:-trim]) for story in Pstories]))
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
    trim = 5
    subject = '0'+str(args.subjectNum)
    zRresp, zPresp = load_subject_fMRI(subject, args.modality)
    colab_dict = {"delRstim":delRstim, "delPstim":delPstim, "zRresp":zRresp, "zPresp": zPresp}
    final_data_file = os.path.join('stim_', subject, args.modality, 'colab_'+subject +'_' +args.modality+'.pickle') 
    with open(final_data_file, 'wb') as pickle_file:
            pickle.dump(colab_dict, pickle_file)
    




