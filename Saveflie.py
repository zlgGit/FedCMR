import numpy as np
def save_training_info(round, accuracy, filename):
    with open(filename, 'a') as f:
        f.write(f'{round}\t')
        f.write(f'{accuracy}\r')

def save_fature_rep(allfeaturs, alllabels,filename):
    np.save(filename+f"_")