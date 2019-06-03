"""!
@brief Data Preprocessor for wsj0-mix dataset for more efficient
loading and also in order to be able to use the universal pytorch
loader.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import os
import sys
current_dir = os.path.dirname(os.path.abspath('__file__'))
root_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(root_dir)

from glob2 import glob
from scipy.io.wavfile import read as scipy_wavread
import time
import numpy as np
import matplotlib.pyplot as plt


def parse_info_from_name(preprocessed_dirname):
    """! Given a name of a preprocessed dataset root dirname infer all the
    encoded information

    Args:
        preprocessed_dirname: A dirname as the one follows:
        wsj0_2mix_8k_4s_min_preprocessed or even a dirpath

    Returns:
        min_or_max: String whether the mixtures are aligned with the maximum
        or the minimum number of samples of the constituting sources
        n_speakers: number of speakers in mixtures
        fs: sampling rate in Hz
        wav_timelength: The timelength in seconds of all the mixtures and
                        clean sources
    """
    try:
        dirname = os.path.basename(preprocessed_dirname)
        elements = dirname.split('_')
        min_or_max = elements[-2]
        assert (min_or_max == 'min' or min_or_max == 'max')
        wav_timelength = float(elements[-3][:-1])
        fs = float(elements[-4][:-1])
        n_speakers = int(elements[-5].strip("mix"))

        return min_or_max, n_speakers, fs, wav_timelength

    except:
        raise IOError("The structure of the wsj0-mix preprocessed "
                      "dataset name is not in the proper format. A proper "
                      "format would be: "
                      "wsj0_{number of speakers}mix_{fs}k_{timelength}s_{min "
                      "or max}_preprocessed")


def infer_output_name(input_dirpath, wav_timelength):
    """! Infer the name for the output folder as shown in the example: E.g.
    for input_dirpath: wsj0-mix/2speakers/wav8k/min and for 4s timelength it
    would be wsj0_2mix_8k_4s_min_preprocessed

    Args: input_dirpath: The path of a wsj0mix dataset e.g.
                         wsj0-mix/2speakers/wav8k/min (for mixes with minimum
                         length)
          wav_timelength: The timelength in seconds of all the mixtures and
                          clean sources

    Returns: outputname: as specified in string format
             fs: sampling rate in Hz
             n_speakers: number of speakers in mixtures
    """

    try:
        elements = input_dirpath.lower().split('/')
        min_or_max = elements[-1]
        assert (min_or_max == 'min' or min_or_max == 'max')
        fs = int(elements[-2].strip('wav').strip('k'))
        n_speakers = int(elements[-3].strip("speakers"))
        output_name = "wsj0_{}mix_{}k_{}s_min_preprocessed".format(
            n_speakers,
            fs,
            float(wav_timelength))

        # test that the inferred output name is parsable back
        (inf_min_or_max,
         inf_n_speakers,
         inf_fs,
         inf_wav_timelength) = parse_info_from_name(output_name)
        assert(inf_min_or_max == min_or_max and
               inf_n_speakers == n_speakers and
               inf_fs == fs and
               inf_wav_timelength == wav_timelength)

        return output_name, fs, n_speakers

    except:
        raise IOError("The structure of the wsj0-mix is not in the right "
                "format. A proper format would be: "
                "wsj0-mix/{2 or 3}speakers/wav{fs in Hz}k/{min or max}")


def convert_wsj0mix_to_universal_dataset(input_dirpath,
                                         output_dirpath,
                                         wav_timelength):
    """! This function converts the wsj0mix dataset in a universal
    dataset where each sample has each own folder and all values are
    stored as Tensors for efficiency and in order to be loaded
    irrespective of the structure of the dataset.

    Args:
        input_dirpath: The path of a wsj0mix dataset e.g.
                       wsj0-mix/2speakers/wav8k/min (for mixes with
                       minimum length)
        output_dirpath: The path for storing the new dataset
                        (the directories would be created recursively)
        wav_timelength: The timelength in seconds of all the mixtures
                        and clean sources

    Intermediate:
        output_name: Default name would be infered as follows:
                     E.g. for wsj0-mix/2speakers/wav8k/min and for 4s
                     timelength:
                     it would be wsj0_2mix_8k_4s_min_preprocessed
    """
    output_name, fs, n_speakers = infer_output_name(input_dirpath,
                                                    wav_timelength)

    print(output_name, fs, n_speakers)

    # subsets = os.listdir(input_dirpath)
    # for subset in subsets:
    #     subset_input_dirpath = os.path.join(input_dirpath, subset)
    #     subset_output_dirpath = os.path.join(output_dirpath, subset)


    # mixtures_dir = os.path.join(input_dirpath, subsets[0], 'mix')
    # files = glob(mixtures_dir + '/*.wav')
    # unique_ids = set([os.path.basename(f) for f in files])
    # print(unique_ids)


def example_of_usage():
    input_dirpath = '/mnt/data/wsj0-mix/2speakers/wav8k/min'
    output_dirpath = '/mnt/data/wsj0mix_preprocessed'
    wav_timelength = 4
    convert_wsj0mix_to_universal_dataset(input_dirpath,
                                         output_dirpath,
                                         wav_timelength)


if __name__ == "__main__":
    example_of_usage()
