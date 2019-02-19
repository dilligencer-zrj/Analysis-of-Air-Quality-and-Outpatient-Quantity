import numpy as np

def cal_mean_var(array):
    mean = array.mean( axis=0 )
    var = array.var( axis=0 )
    return  mean,var


