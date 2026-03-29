import numpy as np


def compute_outlier_imputation(arr, cut_off,left_thresh,impute):
    perc_up = np.percentile(arr, left_thresh)
    perc_down = np.percentile(arr, cut_off)
    #print(perc_up,perc_down)
    length_start=arr.shape[0]
    if impute:
        arr[arr < perc_up] = perc_up
        arr[arr > perc_down] = perc_down
    else:
        #print(arr[arr < perc_up].shape,arr[arr > perc_down].shape)
        arr[arr < perc_up] = np.nan
        arr[arr > perc_down] = np.nan
    length_end=arr.shape[0]
    # print(length_start)
    # print(length_end)
    # print(length_start-length_end)
    return arr


def outlier_imputation(data, id_attribute, value_attribute, cut_off,left_thresh,impute):
    grouped = data.groupby([id_attribute])[value_attribute]
    #print(cut_off)
    for id_number, values in grouped:
        #print("=========")
        #print(id_number)
        #print(values.max(),values.min(),values.mean())
        index = values.index
        values = compute_outlier_imputation(values, cut_off,left_thresh,impute)
        data.loc[index, value_attribute] = values
    data=data.dropna(subset=[value_attribute])
    #print(data.shape)
    return data

