"""!
@brief Library for experiment loss functionality

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')


def log_one_heatmap(experiment,
                    cmap,
                    title,
                    np_array):
    if np_array.shape[1] > 3 * np_array.shape[0]:
        many = int(np_array.shape[1] / (1. * np_array.shape[0]))
        ar = np.repeat(np_array, [many for _ in range(np_array.shape[0])], axis=0)
    else:
        ar = np_array
    plt.imshow(ar, cmap=cmap,
               interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.grid(False)
    experiment.log_figure(figure=plt, figure_name=title, overwrite=False)
    plt.close()


def log_heatmaps(experiment,
                 np_images,
                 titles,
                 cmaps):
    assert len(cmaps) == len(titles)
    assert len(titles) == len(np_images)

    for np_image, title, cmap in zip(np_images, titles, cmaps):
        log_one_heatmap(experiment,
                        cmap,
                        title,
                        np_image)


def create_and_log_afe_internal(experiment,
                                enc_masks,
                                mix_encoded,
                                encoder_basis,
                                decoder_basis):

    np_images = [enc_masks[0], enc_masks[1], mix_encoded,
                 encoder_basis, decoder_basis]
    cmaps = ['Blues', 'Reds', 'jet', 'Greens', 'Oranges']
    titles = ['Mask 1', 'Mask 2', 'Encoded Mix', 'Encoder Basis',
              'Decoder Basis']

    log_heatmaps(experiment, np_images, titles, cmaps)
    # experiment.log_image(masks, name='Mask Yolo')

    # viz.heatmap(
    #     X=rec_sources_masks[0],
    #     opts=dict(
    #         title='Reconstructed Source Mask 1',
    #         colormap='Blues'
    #     ),
    #     win='Reconstructed Source Mask 1'
    # )
    # viz.heatmap(
    #     X=rec_sources_masks[1],
    #     opts=dict(
    #         title='Reconstructed Source Mask 2',
    #         colormap='Reds'
    #     ),
    #     win='Reconstructed Source Mask 2'
    # )
    #
    # diff00 = np.abs(target_masks[0] - rec_sources_masks[0])
    # diff01 = np.abs(target_masks[0] - rec_sources_masks[1])
    # diff10 = np.abs(target_masks[1] - rec_sources_masks[0])
    # diff11 = np.abs(target_masks[1] - rec_sources_masks[1])
    #
    # if diff00.sum() < diff01.sum():
    #     diff_s1 = diff00
    #     diff_s2 = diff11
    # else:
    #     diff_s1 = diff01
    #     diff_s2 = diff10
    #
    # viz.heatmap(
    #     X=diff_s1,
    #     opts=dict(
    #         title='Reconstructed Mask Difference 1',
    #         colormap='Greens'
    #     ),
    #     win='Reconstructed Mask Difference 1'
    # )
    # viz.heatmap(
    #     X=diff_s2,
    #     opts=dict(
    #         title='Reconstructed Mask Difference 2',
    #         colormap='Greens'
    #     ),
    #     win='Reconstructed Mask Difference 2'
    # )
    #
    # viz.heatmap(
    #     X=target_masks[0],
    #     opts=dict(
    #         title='Target Mask 1',
    #         colormap='Blues'
    #     ),
    #     win='Target Mask 1'
    # )
    # viz.heatmap(
    #     X=target_masks[1],
    #     opts=dict(
    #         title='Target Mask 2',
    #         colormap='Reds'
    #     ),
    #     win='Target Mask 2'
    # )
    #
    # viz.heatmap(
    #     X=mix_encoded,
    #     opts=dict(
    #         title='Encoded Mixture',
    #         colormap='Jet'
    #     ),
    #     win='Encoded Mixture'
    # )