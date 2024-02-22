"""
SenCID (Senescent Cell Identification) API
"""


import sys
from absl import app
#from pathlib import Path
#base_path = Path(__file__).parent
import os
import numpy as np
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_path)

from DataPro import ScaleData
from Pred import Pred, Recommend



FLAGS = None

def SenCID(adata, sidnums=[1,2,3,4,5,6], denoising = True, binarize = False, threads = 1, savetmp = False):
    r"""
    Run SenCID Program
    Parameters
    ----------
    adata
        Single-cell dataset in scanpy.AnnData form. Note that SenCID inputs the raw counts of data, thus it is recommended to process SenCID before the downstream analysis (etc., scaling or clustering of single cells)
    sidnums
        Numbers (1 to 6) of SID predictor index that used to calculate SID score. All six SIDs are calculated by default.
    denoising
        Bool for whether to use DCA method to denoise count matrix. Recommended for single cell data. True by default. 
    threads
        Number of threads used for DCA denoising.
    savetmp
        Bool for whether to save the temp file. False by default. 
    binarize
        Bool for whether to binarize SID Scores into crude Senescent(1)-Or-Not(0) judgements by the threshold 0.5. 
    Returns
    -------
    pred_dict
        A dictionary with 6 (if sidnums=[1,2,3,4,5,6] by default) DataFrames for SenCID prediction. Column 0 for each DataFrame are boolen values refering to SID scores ranging from 0 to 1; Column 1 are Decision Function values similar to SID scores, but without the ranges and centered by 0; Column 2 only exists when binarize = True, which refers to crude Senescent(1)-Or-Not(0) judgements by the threshold 0.5. 
    recSID
        DataFrame for the recommendation scores of each SID upon each cell. Column 0 for the most recommended SID for each cell. 
    tmps
        Empty if savetmp = False. A dictionary with 3 DataFrames of temp files for reproduction. "raw_mat" for the raw expression of senescent-related-genes in adata. "denoised_mat" for the dca-denoised expression. "scaled_mat" for the z-scored expression. 
    """
    data_scaled, tmps = ScaleData(adata = adata,  
                          denoising = denoising, 
                          threads = threads, 
                          savetmp = savetmp)
    if savetmp:
        tmps.append(data_scaled)
        tmp_names = ['raw_mat', 'denoised_mat', 'scaled_mat']
        tmps = dict(zip(tmp_names, tmps))
    else:
        tmps = None

    pred_list = [Pred(data_scaled, sidnum, binarize) for sidnum in sidnums]
    sid_list = ['SID'+str(sidnum) for sidnum in sidnums]
    pred_dict = dict(zip(sid_list, pred_list))
    recSID = Recommend(data_scaled)
    print('Finished. Giving SID scores and SID Recommendation...')
    return pred_dict, recSID, tmps

