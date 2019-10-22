# two_step_mask_learning

A general two-step training recipe for sound source separation. In the first step the ideal masks are learned under a front end transformation. The ideal masks or targets seve as an upper bound for source separation performance. Then we train the parameters of the separation module using SI-SDR loss on the trained latent targets. 
