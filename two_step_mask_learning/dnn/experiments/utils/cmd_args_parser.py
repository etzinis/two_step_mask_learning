"""!
@brief Experiment Argument Parser

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import argparse


def get_args():
    """! Command line parser """
    parser = argparse.ArgumentParser(
        description='CometML Experiment Argument Parser')
    parser.add_argument("--train", type=str,
                        help="Training dataset",
                        default='WSJ2MIX8K',
                        choices=['WSJ2MIX8K', 'WSJ2MIX8KPAD', 'TIMITMF8K'])
    parser.add_argument("--val", type=str,
                        help="Validation dataset",
                        default='WSJ2MIX8K',
                        choices=['WSJ2MIX8K', 'WSJ2MIX8KPAD', 'TIMITMF8K'])
    parser.add_argument("-elp", "--experiment_logs_path", type=str,
                        help="""Path for logging experiment's audio.""",
                        default=None)
    parser.add_argument("--n_train", type=int,
                        help="""Reduce the number of training 
                            samples to this number.""", default=None)
    parser.add_argument("--n_val", type=int,
                        help="""Reduce the number of evaluation 
                            samples to this number.""", default=None)
    parser.add_argument("-ri", "--return_items", type=str, nargs='+',
                        help="""A list of elements that this 
                        dataloader should return. See available 
                        choices which are based on the saved data 
                        names which are available. There is no type 
                        checking in this return argument.""",
                        default=['mixture_wav', 'clean_sources_wavs'],
                        choices=['mixture_wav',
                                 'clean_sources_wavs',
                                 'mixture_wav_norm',
                                 'clean_sources_wavs_norm'])
    parser.add_argument("-tags", "--cometml_tags", type=str,
                        nargs="+", help="""A list of tags for the cometml 
                        experiment.""",
                        default=[])
    parser.add_argument("--experiment_name", type=str,
                        help="""Name of current experiment""",
                        default=None)
    parser.add_argument("--project_name", type=str,
                        help="""Name of current experiment""",
                        default="first_wsj02mix")

    # device params
    parser.add_argument("-cad", "--cuda_available_devices", type=str,
                        nargs="+",
                        help="""A list of Cuda IDs that would be 
                        available for running this experiment""",
                        default=['0'],
                        choices=['0', '1', '2', '3'])
    # Adaptive front-end parameters
    parser.add_argument("-afe_reg", "--adaptive_fe_regularizer", type=str,
                        help="""regularization on the trained basis.""",
                        default=None,
                        choices=['compositionality',
                                 'softmax',
                                 'binarized'])

    # model params
    parser.add_argument("-N", "--n_basis", type=int,
                        help="Dim of encoded representation",
                        default=256)
    parser.add_argument("-B", "--tasnet_B", type=int,
                        help="Number of dimensions in bottleneck.",
                        default=256)
    parser.add_argument("-H", "--tasnet_H", type=int,
                        help="Number of channels in convd.",
                        default=512)
    parser.add_argument("-L", "--n_kernel", type=int,
                        help="Length of encoding convolutional filters",
                        default=20)
    parser.add_argument("-P", "--tasnet_P", type=int,
                        help="Length of filters in convolution blocks",
                        default=3)
    parser.add_argument("-X", "--tasnet_X", type=int,
                        help="Number of convolutional blocks in each repeat",
                        default=8)
    parser.add_argument("-R", "--tasnet_R", type=int,
                        help="Number of repeats of TN Blocks",
                        default=4)
    parser.add_argument("-norm", "--norm_type", type=str,
                        help="""The type of the applied normalization layer.""",
                        default="gln", choices=['bn', 'gln'])
    parser.add_argument("-version", "--tasnet_version", type=str,
                        help="""The type of Tasnet you want to run.""",
                        default="simplified", choices=['full', 'simplified'])
    # training params
    parser.add_argument("-bs", "--batch_size", type=int,
                        help="""The number of samples in each batch. 
                            Warning: Cannot be less than the number of 
                            the validation samples""", default=4)
    parser.add_argument("--n_jobs", type=int,
                        help="""The number of cpu workers for 
                            loading the data, etc.""", default=4)
    parser.add_argument("--n_epochs", type=int,
                        help="""The number of epochs that the 
                        experiment should run""", default=50)
    parser.add_argument("-lr", "--learning_rate", type=float,
                        help="""Initial Learning rate""", default=1e-2)

    return parser.parse_args()
