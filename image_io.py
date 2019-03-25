# Most frequently used Image IO functions required
# for many CV/ML research tasks at Clobotics
# Author: Achyut Sarma Boggaram (achyut.boggaram@clobotics.com)
# Copyrights: Clobotics Corporation


import numpy as np


def prewhiten(x):
    # #normalize the image before passing it into the CNN #Image pre-processing
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y
