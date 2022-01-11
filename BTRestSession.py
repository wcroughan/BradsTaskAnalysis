
class BTRestSession:
    """
    Contains all the data from one post-task rest session
    """

    LFP_SAMPLING_RATE = 1500.0

    def __init__(self):
        self.name = ""
        self.lfp_fnames = []
