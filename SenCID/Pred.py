# -*- coding: utf-8 -*-
"""
Created on Thu May  5 15:11:55 2022

@author: admin
"""


import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import numpy as np
import warnings
import joblib
import os
base_path = os.path.dirname(os.path.abspath(__file__))

warnings.filterwarnings("ignore")

def SplitLabel(df, featurelist):
    x = df.loc[:,featurelist]
    x = x.fillna(0)
    #print(x)
    return x

def GetModels(sidnum):
    model = joblib.load(str(base_path)+"/model/SID"+str(sidnum)+".pkl")
    model_L = joblib.load(str(base_path)+"/model/SID"+str(sidnum)+"_L.pkl")
    return model, model_L

def GetFeatures(sidnum):
    feature_dict = {
            1:['CCND1', 'CDKN1A', 'PCNA', 'CDK6', 'ZMAT3', 'EGR1', 'BHLHE40', 'MYLK', 'EDN1'
               , 'F3', 'PCK2', 'BTG2', 'TRIM22', 'OSTM1', 'DEPDC1', 'NCEH1', 'TP53INP1'
               , 'PRICKLE2', 'PGM2L1'], 
            2:['CCND1', 'HMGB2', 'CKB', 'CDKN2C', 'EDN1', 'FANCE', 'PLAT', 'ENC1', 'SV2A'
               , 'PHGDH', 'ANGPTL4', 'MFSD1', 'CDCA3'], 
            3:['CDK1', 'LMNB1', 'SOD2', 'CCND1', 'FGF2', 'GDF15', 'IGFBP3', 'IGFBP5', 'MKI67'
               , 'SERPINE1', 'TOP2A', 'CCNE1', 'CBX4', 'PRKCD', 'CDKN2D', 'CDK6', 'SLC16A7'
               , 'RNASET2', 'BUB1B', 'TAGLN', 'S100A6', 'RXRB', 'RRM2', 'RING1', 'CDKN2AIP'
               , 'PLAUR', 'MYBL2', 'MAP3K5', 'SMAD1', 'JAK2', 'HNRNPA1', 'HELLS', 'NPTXR', 'ETS2'
               , 'EGR1', 'DUSP6', 'DUSP4', 'E2F7', 'CKB', 'UBE2C', 'EHMT2', 'NDRG1', 'CD55'
               , 'ICAM3', 'ITGA2', 'BHLHE40', 'HDAC4', 'ING2', 'MYLK', 'PRKCH', 'RUNX1', 'SGK1'
               , 'TGFB1I1', 'AMPD3', 'SLC25A4', 'AOX1', 'BARD1', 'CALD1', 'CTSC', 'CKS2', 'COL6A3'
               , 'COL15A1', 'SLC31A2', 'DNASE1L1', 'DPYSL3', 'EDN1', 'EMP2', 'EPAS1', 'F3', 'FAT1'
               , 'FKBP5', 'FLNC', 'FN1', 'STMN1', 'LOX', 'LOXL2', 'LRP1', 'MATN2', 'MDK', 'MEST'
                , 'MXI1', 'PCOLCE', 'SERPINF1', 'PFKP', 'PLAT', 'PRPS1', 'PTPRK', 'PTX3', 'SAT1'
                , 'SLC1A1', 'SOX4', 'SVIL', 'SYT1', 'TIMP3', 'TMPO', 'ZNF22', 'BTG2', 'CDC7', 'STC2'
                , 'TNFRSF10D', 'P4HA2', 'SYNGR2', 'SYNGR1', 'DHRS3', 'SLIT2', 'ARHGAP29', 'KIF23'
                , 'ATP6V1G1', 'SH3PXD2A', 'DLGAP5', 'HEPH', 'KIF14', 'TROAP', 'EDIL3', 'KIF20A'
                , 'CITED2', 'NDC80', 'PGRMC2', 'PLK4', 'RAB40B', 'CNTRL', 'DNAJB4', 'CIT', 'DUSP10'
                , 'OIP5', 'COBLL1', 'NCAPH', 'PLA2G15', 'KIF4A', 'LMOD1', 'MPC2', 'CYFIP2'
                , 'TSPAN13', 'TNFRSF21', 'EML4', 'SLC43A3', 'ARHGEF3', 'ANGPTL4', 'SCARA3', 'DTL'
                , 'RAB6B', 'CYB5R1', 'GNG2', 'ANLN', 'MANSC1', 'CEP55', 'MCM10', 'PLXNA3'
                , 'CDK5RAP2', 'SULF2', 'EMC7', 'OLFML3', 'PLSCR4', 'KIAA1191', 'NIPAL3', 'SPC25'
                , 'RHOU', 'FAM111A', 'CLSPN', 'PERP', 'OGFRL1', 'SCD5', 'ARMC9', 'FAM214B'
                , 'COLEC12', 'BRIP1', 'SGIP1', 'MED30', 'SLFN11', 'ZNF300', 'FANK1', 'TIFA'
                , 'AHNAK2', 'PIK3IP1', 'FBXO32', 'CCDC58', 'FAM43A', 'GPR155', 'ZNF610', 'ZNF367'
                , 'OTUD1', 'ASPM', 'GAS2L3', 'NCKAP5', 'FAM111B', 'ARHGEF37'], 
             4:['CDKN2A', 'CDKN2B', 'GDF15', 'SERPINE1', 'CCNE1', 'TBX2', 'CEBPB', 'ARHGAP18'
                , 'RING1', 'MAPK10', 'PLAUR', 'IGF1R', 'HSPB1', 'CABIN1', 'ETS2', 'ETS1', 'CLU'
                , 'BMP2', 'GEM', 'INHBA', 'MMP14', 'ARPC1B', 'BAG3', 'BHLHE40', 'DHCR24', 'SENP7'
                , 'SGK1', 'ARF4', 'STS', 'CD81', 'CTSC', 'COL6A3', 'COL8A1', 'DPYSL3', 'EMP2'
                , 'EPAS1', 'F3', 'FN1', 'IFI6', 'HAS2', 'ITGAV', 'ITGB5', 'LAMC1', 'MEST'
                , 'SERPINF1', 'PPP1R3C', 'SAT1', 'TAP1', 'TGM2', 'SLC7A5', 'ADAM19', 'TNFRSF10D'
                , 'SPHK1', 'SLC5A6', 'SYNGR2', 'DHRS3', 'SNPH', 'CPQ', 'BLCAP', 'PRSS23', 'CIT'
                , 'IRAK3', 'FILIP1L', 'COBLL1', 'TBC1D9', 'OSTM1', 'GNG2', 'SAMD9', 'ERCC6L'
                , 'CENPJ', 'FIGNL1', 'PERP', 'TMBIM1', 'PIP4K2C', 'BORA', 'PGBD1', 'ARRDC4'
                , 'ZNF300', 'HTRA3', 'CYP2U1', 'FBXO32', 'FRMD6', 'SLC43A2', 'FAM43A', 'LACC1'
                , 'CCDC138', 'DCBLD1'], 
             5:['LMNB1', 'CCL2', 'GDF15', 'IGFBP3', 'MKI67', 'SERPINE1', 'TWIST1', 'CCNE1'
                , 'PLAUR', 'PIM1', 'IRS1', 'G6PD', 'EEF1A1', 'DUSP6', 'DUSP4', 'INHBA', 'ITGA2'
                , 'ITPKB', 'SIX1', 'STK32C', 'ABCA3', 'ASAH1', 'BPGM', 'CDC25B', 'COL15A1'
                , 'DPYSL3', 'F3', 'FUCA1', 'IFI6', 'GLS', 'HAGH', 'HAS2', 'INSIG1', 'STMN1', 'MANBA'
                , 'MDK', 'PCOLCE', 'SERPINF1', 'SERPINI1', 'PLAT', 'PLOD2', 'PPP1R3C', 'PTMA'
                , 'SCD', 'SFRP1', 'SLC1A1', 'TAP1', 'KHSRP', 'CREG1', 'GGH', 'SYNGR2', 'SLIT2'
                , 'AKAP12', 'ISG15', 'ATP6AP2', 'MLLT11', 'CBX5', 'ORC6', 'TUBG2', 'OSTM1', 'RAB6B'
                , 'DHRS7', 'TM7SF3', 'GNG2', 'USP53', 'HERC6', 'SULF2', 'EMC7', 'OLFML3', 'NCEH1'
                , 'NABP1', 'COLEC12', 'SLC9A7', 'LMBRD2', 'FANK1', 'ATG4A', 'WDR63', 'FAM43A'
                , 'DOCK11', 'PRICKLE2', 'PGM2L1', 'SLC46A3', 'IGIP'], 
             6:['MYC', 'AURKB', 'CCND1', 'CDKN1A', 'CDKN2B', 'ID1', 'IGFBP3', 'IGFBP5', 'LMNA'
                , 'MKI67', 'MMP2', 'STAT1', 'TWIST1', 'PCNA', 'CCNE1', 'TBX2', 'PRKCD', 'ARHGAP18'
                , 'WNT5A', 'UBA52', 'TAGLN', 'SLC13A3', 'RRM2', 'RRM1', 'RPS27A', 'RPL5', 'PSMB5'
                , 'MAPK10', 'ERRFI1', 'PGD', 'MYBL2', 'SMAD6', 'SMAD3', 'SMAD1', 'JUN', 'CDC26'
                , 'ETS2', 'ENG', 'EEF1B2', 'DUSP4', 'E2F7', 'JDP2', 'CLU', 'CKB', 'UBE2C', 'CENPA'
                , 'ANAPC10', 'CD9', 'CSF1', 'CXCL12', 'FAS', 'IGFBP6', 'VEGFA', 'BHLHE40'
                , 'CDK2AP1', 'DDB2', 'DHCR24', 'LGALS3', 'NUAK1', 'PMVK', 'SIX1', 'YAP1', 'ZFP36'
                , 'ABCA3', 'ANGPT1', 'SLC25A5', 'ANXA4', 'CDC20', 'CENPE', 'CENPF', 'CKS2'
                , 'COL6A3', 'COL8A1', 'DAB2', 'DGKA', 'GADD45A', 'DNASE1L1', 'DPYSL3', 'EMP2'
                , 'EPAS1', 'EPHA4', 'CLN8', 'ERH', 'FBL', 'ETV4', 'FANCA', 'FHL1', 'IFI6', 'GAS6'
                , 'HAS2', 'HLA-B', 'HMGB3', 'HMMR', 'HNRNPH1', 'SP110', 'IFIT1', 'KIF11', 'LAMC1'
                , 'STMN1', 'LIPA', 'LOXL2', 'LSAMP', 'MDK', 'MEST', 'ODC1', 'PAM', 'ENPP2', 'PLCB4'
                , 'PRIM1', 'PRRG1', 'HTRA1', 'RPL4', 'RPS9', 'RPS16', 'SORT1', 'SHMT2', 'SLC1A1'
                , 'SOX4', 'THBS1', 'TPM2', 'TTK', 'BTG2', 'SLC7A5', 'ENC1', 'PEA15', 'ADAM19', 'GGH'
                , 'SPHK1', 'P4HA2', 'PAPSS2', 'SYNGR2', 'SYNGR1', 'TRIP13', 'AKAP12', 'ISG15'
                , 'HDAC9', 'MELK', 'GINS1', 'GFPT2', 'AASS', 'TRIM22', 'NDC80', 'IFITM3', 'SEMA3C'
                , 'RAD51AP1', 'FILIP1L', 'OIP5', 'COBLL1', 'MAPRE3', 'KIAA0930', 'NCAPH', 'DDAH1'
                , 'DDX58', 'APOL2', 'RAD54B', 'POC1A', 'CNRIP1', 'CLIP3', 'TMEM251', 'SLC17A5'
                , 'CYFIP2', 'DKK3', 'TNFRSF21', 'UBE2T', 'LRP12', 'RRM2B', 'ANGPTL4', 'NUSAP1'
                , 'SCARA3', 'DPH5', 'COQ3', 'MIS18A', 'GNG2', 'RIN2', 'SAMD9', 'HAUS4', 'HERC6'
                , 'ZWILCH', 'CDCA8', 'CEP55', 'MNS1', 'MCM10', 'CDCA7L', 'PLXNA3', 'HHAT', 'NXT2'
                , 'SULF2', 'OLFML3', 'PRTFDC1', 'RCN3', 'C3orf14', 'NCEH1', 'LSM2', 'CLSPN'
                , 'CENPK', 'NABP1', 'NT5DC2', 'CENPH', 'NUP85', 'KIF18A', 'NUF2', 'ANTXR1'
                , 'SLC9A7', 'NEXN', 'ZNF300', 'HTRA3', 'FBXO32', 'RMI2', 'WDR63', 'LACC1', 'CKAP2L'
                , 'ARL6IP6', 'CDCA2', 'ITPRIPL2', 'ZNF610', 'PRICKLE2', 'SAMD9L', 'HYLS1', 'ASPM'
                , 'GAS2L3', 'FAM111B', 'CENPW', 'ARHGEF37', 'SHC4']
            }
    rfeatures = feature_dict[sidnum]
    return rfeatures

#cpm_zcol = pd.read_csv(filepath+'cpm_dca/cpm_zcol_seneinter_'+filename+'.txt', sep = '\t', index_col = 0)
def Pred(cpm_zcol, sidnum, binarize = False): 
    df_test = cpm_zcol.T
    rfeatures = GetFeatures(sidnum)
    x_test = SplitLabel(df_test, rfeatures)
    print('Loading models of SID%d...'%sidnum)
    model, model_L = GetModels(sidnum)
    print('Making predictions of SID%d...'%sidnum)
    model_decf = model.decision_function(x_test)
    Ltest_prob = model_L.predict_proba(x_test)
    model_pretest = model_L.predict(x_test)
    if binarize: 
        labels = pd.concat([pd.DataFrame(Ltest_prob[:,1]), pd.DataFrame(model_decf), pd.DataFrame(model_pretest)], axis = 1)
        labels.columns = ['SID_Score', 'Decision', 'Binarization']
    else: 
        labels = pd.concat([pd.DataFrame(Ltest_prob[:,1]), pd.DataFrame(model_decf)], axis = 1)
        labels.columns = ['SID_Score', 'Decision']
    labels.index = x_test.index
    return labels

def Recommend(cpm_zcol):
    df_test = cpm_zcol.T
    print('Loading Recommend model...')
    model_rec = joblib.load(str(base_path)+"/model/recommend_model.pkl")
    recommended = model_rec.predict(df_test)
    rec_decf = model_rec.decision_function(df_test)
    rec_labels = pd.concat([pd.DataFrame(recommended), pd.DataFrame(rec_decf)], axis = 1)
    rec_labels.columns = ['RecSID', 'rec_SID1', 'rec_SID2', 'rec_SID3', 'rec_SID4', 'rec_SID5', 'rec_SID6']
    rec_labels.index = df_test.index
    return rec_labels
