import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

old_dataset = True
if old_dataset:
# write (train_data, train_targets) and (test_data, test_targets) to .csv files
    df_in_concat  = pd.DataFrame()
    df_out_concat = pd.DataFrame()
    for subdir in [1010, 1111, 2020, 2121]:
        for num in [0, 625, 1250, 1875, 2500, 3125, 3750, 4375, 5000, 5625, 6250, 6875, 7500, 8125, 8750, 9375]:
            os.chdir("/home/yzhang/Results/flowfield_rawdat/conv_"+str(subdir)+"_p"+str(num))
            print(os.getcwd())
            df_in = pd.DataFrame(data = np.loadtxt("input_features.dat", delimiter=','), columns = ['position', 'area', 'diameter', 'angleup', 'angledown', 'shape', 'pdrop', 'reloc'])
            df_in.to_csv('test_data.csv')
            df_in_concat = df_in_concat.append(df_in, ignore_index=True)
            df_out = pd.DataFrame(data = np.loadtxt("output_targets.dat"), columns = ['Fr'])
            df_out.to_csv('test_targets.csv')
            df_out_concat = df_out_concat.append(df_out, ignore_index=True)
    df_in_concat.to_csv('train_data.csv')
    df_out_concat.to_csv('train_targets.csv')
else:
    for subdir in [1010, 1111, 2020, 2121]:
        for num in [1000, 2000, 3000, 4000, 6000, 7000, 8000, 9000]:
            os.chdir("/home/yzhang/Results/flowfield_rawdat/conv_"+str(subdir)+"_p"+str(num))
            print(os.getcwd())
            df_in = pd.DataFrame(data = np.loadtxt("input_features.dat", delimiter=','), columns = ['position', 'area', 'diameter', 'angleup', 'angledown', 'shape', 'pdrop', 'reloc'])
            df_in.to_csv('test_data.csv')
            df_out = pd.DataFrame(data = np.loadtxt("output_targets.dat"), columns = ['Fr'])
            df_out.to_csv('test_targets.csv')
