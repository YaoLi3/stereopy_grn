# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')
import stereo
from stereo.io import read_gem, read_gef
import sys
from regulatory_network import InferenceRegulatoryNetwork, PlotRegulatoryNetwork

from stereo.core.stereo_exp_data import StereoExpData
from stereo.preprocess.qc import cal_qc


if __name__ == '__main__':
    fn = sys.argv[1]
    grn = InferenceRegulatoryNetwork()
    data = InferenceRegulatoryNetwork.read_file(fn)
    print(type(data))

