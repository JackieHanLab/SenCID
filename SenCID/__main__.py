# -*- coding: utf-8 -*-
"""
Created on Fri May  6 18:27:29 2022

@author: admin
"""


import sys
import argparse
from absl import app
import os
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_path)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False    
    else:
        raise argparse.ArgumentTypeEror('Boolean value (no, false, f, n, 0, False) expected')    

def parse_args():
    parser = argparse.ArgumentParser() #将命令行字符串解析为Python对象

    # Hyper-parameters for prefix, prop and random seed
    parser.add_argument('--filepath', type=str, 
                        help='Directory that contains sequencing count data')
    parser.add_argument("--add_prefix", type=str2bool, default=False, 
                        help='Whether to add file name as the prefix of cell barcodes')
    parser.add_argument("--denoising", type=str2bool, default=True, 
                        help='Whether to use DCA method to denoise count matrix. Recommended for single cell data')
    parser.add_argument("--binarize", type=str2bool, default=True, 
                        help='Whether to binarize SID Scores into crude Senescent(1)-Or-Not(0) judgements by the threshold 0.5. ')
    parser.add_argument('--threads', type=int, default=1, 
                        help='Number of threads used for DCA denoising')
    parser.add_argument('--output_dir', type=str, default='output', 
                        help='Directory to save output of prediction under --filepath')
    parser.add_argument('--save_tmps', type=str, default=None, 
                        help='Directory to save temp file under --filepath/output_dir. No temp file will be saved by default')
    parser.add_argument('--sidnums', type=int, default=[1,2,3,4,5,6], 
                        nargs = '+', 
                        help='Numbers (1 to 6) of SID predictor index that used to calculate SID score. All six SIDs are calculated by default')
    parser.add_argument('--fileclass', type=str, default='h5', 
                        help='The type of the expression count files, should be one of h5, txt(tab separated), csv, h5ad')
    parser.add_argument('--genenameCol', type=str, default=None, 
                        help='Use the specified column in adata.var, which contains hgnc symbol (like BIRC5, BRCA1, etc). Only available when --fileclass is one of h5 or h5ad. Default is None which will use the adata.var_names')

   
    return parser.parse_known_args()


def main():

    FLAGS = None
    FLAGS, unparsed = parse_args()
    
    from utils import Mkdir
    from DataPro import GetFiles, MakeAdata
    from api import SenCID

    filepath = FLAGS.filepath if FLAGS.filepath[-1]=='/' else FLAGS.filepath+'/'
    add_prefix = FLAGS.add_prefix
    denoising = FLAGS.denoising
    binarize = FLAGS.binarize
    threads = FLAGS.threads
    genenameCol = FLAGS.genenameCol
    output_dir = FLAGS.output_dir if '/' in FLAGS.output_dir else filepath+FLAGS.output_dir
    
    Mkdir(output_dir)
    output_dir = output_dir+'/'
    if FLAGS.save_tmps is not None:
        if '/' in FLAGS.save_tmps:
            savetmp_file = FLAGS.save_tmps
        else:
            savetmp_file = output_dir+FLAGS.save_tmps
        Mkdir(savetmp_file)
        savetmp = True
    else:
        savetmp = False
    sidnums = FLAGS.sidnums
    fileclass = FLAGS.fileclass
    files, filenames = GetFiles(filepath, fileclass)
    
    for i, filename in enumerate(filenames):
        adata = MakeAdata(file = files[i], 
                                filename = filename, 
                                fileclass = fileclass, 
                                add_prefix = add_prefix, 
                                genenameCol = genenameCol)
        try:
            pred_dict, recSID, tmpfiles = SenCID(adata = adata, 
                                sidnums = sidnums, 
                                denoising = denoising, 
                                binarize = binarize, 
                                threads = threads, 
                                savetmp = savetmp) 
        except: 
            e = sys.exc_info()
            print(e) # (Exception Type, Exception Value, TraceBack)
            continue
        for k, sidnum in enumerate(sidnums):
            pred_dict['SID'+str(sidnum)].to_csv(output_dir+'predictions_'+'SID'+str(sidnum)+'_'+str(filename)+'.txt', sep='\t')
        if savetmp:
            tmpfiles['raw_mat'].to_csv(savetmp_file+'/raw_mat_'+filename+'.txt', sep = '\t')
            tmpfiles['denoised_mat'].to_csv(savetmp_file+'/denoised_mat_'+filename+'.txt', sep = '\t')
            tmpfiles['scaled_mat'].to_csv(savetmp_file+'/scaled_mat_'+filename+'.txt', sep = '\t')
        recSID.to_csv(output_dir+'RecommendSID_Index_'+str(filename)+'.txt', sep='\t')
    return
        
               
