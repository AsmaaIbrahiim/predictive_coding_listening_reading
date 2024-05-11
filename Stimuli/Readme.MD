# Text models feature extraction
----------
- Harry Potter Dataset 
```bash
python extract_features_words.py --input_file ./Harry_Potter/stimuli_words.npy --model bert-base --sequence_length 20 --output_file harrypotter
```

- Narratives Dataset
```bash
python extract_features_words.py --input_file ./Narratives/tunnel_align.csv --model bart-base --sequence_length 20 --output_file tunneling
```

- Subset-Moth-Radio-Hour Dataset
```bash
python extract_features_words.py --input_file ./stories/ --model bert-base --sequence_length 20 --output_file subset_moth_radio
```

- Moth-Radio-Hour Dataset
```bash
python extract_features_words.py --input_file ./stories/ --model bert-base --sequence_length 20 --output_file moth_radio
```

- Little Prince
```bash
python extract_features_words.py --input_file ./LittlePrince/lppEN_word_information.csv --model bert-base --sequence_length 20 --output_file little_prince
```

# Downsampling
```
class TRFile(object):
    def __init__(self, trfilename, expectedtr):
        """Loads data from [trfilename], should be output from stimulus presentation code.
        """
        self.trtimes = []
        self.soundstarttime = -1
        self.soundstoptime = -1
        self.otherlabels = []
        self.expectedtr = expectedtr
        
        if trfilename is not None:
            self.load_from_file(trfilename)
        

    def load_from_file(self, trfilename):
        """Loads TR data from report with given [trfilename].
        """
        ## Read the report file and populate the datastructure
        for ll in open(trfilename):
            timestr = ll.split()[0]
            label = " ".join(ll.split()[1:])
            time = float(timestr)

            if label in ("init-trigger", "trigger"):
                self.trtimes.append(time)

            elif label=="sound-start":
                self.soundstarttime = time

            elif label=="sound-stop":
                self.soundstoptime = time

            else:
                self.otherlabels.append((time, label))
        
        ## Fix weird TR times
        itrtimes = np.diff(self.trtimes)
        badtrtimes = np.nonzero(itrtimes>(itrtimes.mean()*self.expectedtr))[0]
        newtrs = []
        for btr in badtrtimes:
            ## Insert new TR where it was missing..
            newtrtime = self.trtimes[btr]+self.expectedtr
            newtrs.append((newtrtime,btr))

        for ntr,btr in newtrs:
            self.trtimes.insert(btr+1, ntr)

    def simulate(self, ntrs):
        """Simulates [ntrs] TRs that occur at the expected TR.
        """
        self.trtimes = list(np.arange(ntrs)*self.expectedtr)
    
    def get_reltriggertimes(self):
        """Returns the times of all trigger events relative to the sound.
        """
        return np.array(self.trtimes)-self.soundstarttime

    @property
    def avgtr(self):
        """Returns the average TR for this run.
        """
        return np.diff(self.trtimes).mean()

def load_generic_trfiles(stories, root, expectedtr):
    """Loads a dictionary of generic TRFiles (i.e. not specifically from the session
    in which the data was collected.. this should be fine) for the given stories.
    """
    trdict = dict()

    for story in stories:
        try:
            trf = TRFile(os.path.join(root, "%s.report"%story), expectedtr)
            trdict[story] = [trf]
        except Exception as e:
            print (e)
    
    return trdict

def lanczosinterp2D(data, oldtime, newtime, window=3, cutoff_mult=1.0, rectify=False):
    """Interpolates the columns of [data], assuming that the i'th row of data corresponds to
    oldtime(i). A new matrix with the same number of columns and a number of rows given
    by the length of [newtime] is returned.
    
    The time points in [newtime] are assumed to be evenly spaced, and their frequency will
    be used to calculate the low-pass cutoff of the interpolation filter.
    
    [window] lobes of the sinc function will be used. [window] should be an integer.
    """
    ## Find the cutoff frequency ##
    cutoff = 1/np.mean(np.diff(newtime)) * cutoff_mult
    print ("Doing lanczos interpolation with cutoff=%0.3f and %d lobes." % (cutoff, window))
    
    ## Build up sinc matrix ##
    sincmat = np.zeros((len(newtime), len(oldtime)))
    for ndi in range(len(newtime)):
        sincmat[ndi,:] = lanczosfun(cutoff, newtime[ndi]-oldtime, window)
    
    if rectify:
        newdata = np.hstack([np.dot(sincmat, np.clip(data, -np.inf, 0)), 
                            np.dot(sincmat, np.clip(data, 0, np.inf))])
    else:
        ## Construct new signal by multiplying the sinc matrix by the data ##
        newdata = np.dot(sincmat, data)

    return newdata
def lanczosfun(cutoff, t, window=3):
    """Compute the lanczos function with some cutoff frequency [B] at some time [t].
    [t] can be a scalar or any shaped numpy array.
    If given a [window], only the lowest-order [window] lobes of the sinc function
    will be non-zero.
    """
    t = t * cutoff
    pi = np.pi
    #val = window * np.sin(np.pi*t) * np.sin(np.pi*t/window) / (np.pi**2 * t**2)
    val = ne.evaluate("window * sin(pi*t) * sin(pi*t/window) / (pi**2 * t**2)")
    val[t==0] = 1.0
    val[np.abs(t)>window] = 0.0

    return val
```

```
#load tunnel alignment
word_alignment = pd.read_csv('./Narratives/tunnel_align.csv', header=None)
word_alignment.head()

word_alignment[[2, 3]] = word_alignment[[2, 3]].apply(pd.to_numeric)
word_alignment['word_times'] = (word_alignment[2]+word_alignment[3])/2

tunnel_tr = load_generic_trfiles(['tunnel'],'./Narratives/TextGrid/', 1.5)
downsampled_data = lanczosinterp2D(tunnel_words_vector, dataset['word_times'],tunnel_tr['tunnel'][0].trtimes, window=3)
```

```
#load little prince run-1 alignment
word_alignment = pd.read_csv('./LittlePrince/lppEN_word_information.csv')
word_alignment.head()

word_alignment[['onset', 'offset']] = word_alignment[['onset', 'offset']].apply(pd.to_numeric)
word_alignment['word_times'] = (word_alignment['onset']+word_alignment['offset'])/2

run1_tr = load_generic_trfiles(['section1'],'./LittlePrince/TextGrid/', 2)
downsampled_data = lanczosinterp2D(littleprince_words_vector, word_alignment['word_times'][0:1517],tunnel_tr['section1'][0].trtimes, window=3)
```
