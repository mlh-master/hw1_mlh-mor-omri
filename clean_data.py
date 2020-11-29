# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rm_ext_and_nan(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    c_ctg = CTG_features.drop(labels=[extra_feature], axis='columns').apply(
       (lambda x: pd.to_numeric(x, errors='coerce'))).dropna()

    c_ctg1 = CTG_features.drop(labels=[extra_feature], axis='columns')
    c_ctg1.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    c_ctg1.dropna()
    # --------------------------------------------------------------------------
    return c_ctg


def nan2num_samp(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe of the dictionary c_cdf containing the "clean" features
    """
    c_cdf = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    temp =CTG_features.drop(axis=1, labels=[extra_feature]).apply((lambda x: pd.to_numeric(x, errors = 'coerce')))
    for feature in temp:
        summ = temp.loc[:, feature].value_counts().sort_index().cumsum()
        cdf = summ/(summ.max())
        idx = np.where(temp.loc[:, feature].isna())[0]
        s = np.random.uniform(0, 1, idx.size)
        i = 0
        while i < idx.size:
            nanval = (s[i]-cdf).abs().idxmin()
            temp.loc[idx[i]+1, feature] = nanval
            i += 1
    c_cdf = temp
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_cdf)


def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_cdf
    :return: Summary statistics as a dicionary of dictionaries (called d_summary) as explained in the notebook
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    d_summary = {}
    for feature in c_feat:
        d_summary[feature] = c_feat.describe()[feature]['min':].to_dict()
    # -------------------------------------------------------------------------
    return d_summary


def rm_outlier(c_feat, d_summary):
    """

    :param c_feat: Output of nan2num_cdf
    :param d_summary: Output of sum_stat
    :return: Dataframe of the dictionary c_no_outlier containing the feature with the outliers removed
    """
    c_no_outlier = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    '''
    temp = {}
    for feature in c_feat:
        temp[feature] = {}
        iqr = d_summary[feature]['75%']-d_summary[feature]['25%']
        LP = d_summary[feature]['25%']-1.5*iqr
        UP = d_summary[feature]['75%']+1.5*iqr
        temp2 = c_feat.copy()[feature]
        temp2[(temp2 <= LP)] = np.nan
        temp2[(temp2 >= UP)] = np.nan
        temp[feature] = temp2
        c_no_outlier = temp
    '''
    temp = c_feat.copy()
    for feat in c_feat:
        IQR = d_summary[feat]['75%'] - d_summary[feat]['25%']
        lower_thresh = d_summary[feat]['25%'] - 1.5 * IQR
        upper_thresh = d_summary[feat]['75%'] + 1.5 * IQR
        temp[feat][(c_feat[feat] < lower_thresh)] = np.nan
        temp[feat][(c_feat[feat] > upper_thresh)] = np.nan

    c_no_outlier = temp
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_no_outlier)


def phys_prior(c_cdf, feature, thresh):
    """

    :param c_cdf: Output of nan2num_cdf
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------
    temp = {}
    temp = c_cdf[feature]
    filt_feature = temp[(temp<= thresh) & (temp>= 0)] #AC cannot be negative
    # -------------------------------------------------------------------------
    return filt_feature


def norm_standard(CTG_features, selected_feat=('LB', 'ASTV'), mode='none', flag=False):
    """

    :param CTG_features: Pandas series of CTG features
    :param selected_feat: A two elements tuple of strings of the features for comparison
    :param mode: A string determining the mode according to the notebook
    :param flag: A boolean determining whether or not plot a histogram
    :return: Dataframe of the normalized/standardazied features called nsd_res
    """
    x, y = selected_feat
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    '''
    nsd_res = CTG_features.copy()
    n_bins = 100
    if mode == 'MinMax':
        for feature in CTG_features:
            min = CTG_features.loc[:, feature].min()
            max = CTG_features.loc[:, feature].max()
            nsd_res[feature] = (CTG_features.loc[:, feature]-min)/(max-min)
    elif mode == 'mean':
        for feature in CTG_features:
            min = CTG_features.loc[:, feature].min()
            max = CTG_features.loc[:, feature].max()
            mean = CTG_features.loc[:, feature].mean()
            nsd_res[feature] = (CTG_features.loc[:, feature] - mean) / (max - min)
    elif mode == 'standard':
        for feature in CTG_features:
            mean = CTG_features.loc[:, feature].mean()
            std = CTG_features.loc[:, feature].std()
            nsd_res[feature] = (CTG_features.loc[:, feature] - mean) / std
    if flag == True:
        plt.figure()
        plt.hist(nsd_res[selected_feat[0]], bins=n_bins)
        plt.hist(nsd_res[selected_feat[1]], bins=n_bins)
        plt.legend([selected_feat[0], selected_feat[1]])
        plt.title([mode])
    '''
    nsd_res = CTG_features.copy()
    if mode != 'none':
        for feat in CTG_features:
            feature = nsd_res[feat]
            mean_feat = feature.mean()
            std_feat = feature.std()
            min_feat = feature.min()
            max_feat = feature.max()
            func_st = lambda x: (x - mean_feat) / std_feat
            func_min_max = lambda x: (x - min_feat) / (max_feat - min_feat)
            func_mean = lambda x: (x - mean_feat) / (max_feat - min_feat)
            switch_mode = {'standard': func_st,
                           'MinMax': func_min_max,
                           'mean': func_mean}
            nsd_res[feat] = nsd_res[feat].map(switch_mode[mode])
    if flag:
        n_bins = 100
        plt.figure()
        plt.hist(nsd_res[selected_feat[0]], bins=n_bins)
        plt.hist(nsd_res[selected_feat[1]], bins=n_bins)
        plt.legend([selected_feat[0], selected_feat[1]])
        plt.title([mode])
    # -------------------------------------------------------------------------
    return pd.DataFrame(nsd_res)
