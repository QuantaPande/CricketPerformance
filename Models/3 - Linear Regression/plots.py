# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 22:16:47 2018

@author: chint
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas

df = pandas.read_csv('train_x_shuffle.csv')
data = df.values

df_test = pandas.read_csv('test_x_shuffle.csv')
data_test = df_test.values

# =============================================================================
# groups = [1,2,3,4,5,6,7,8,9,10]
# i = 1
# 
# plt.figure()
# for group in groups:
#     	plt.subplot(len(groups), 1, i)
#     	plt.plot(data[1294:1836, group])
#     	plt.title(df.columns[group], y=0.5, loc='right')
#     	i += 1
#         
# plt.title('India', y=12)
# plt.show()
# 
# i = 1
# plt.figure()
# for group in groups:
#     	plt.subplot(len(groups), 1, i)
#     	plt.plot(data_test[206:352:, group])
#     	plt.title(df_test.columns[group], y=0.5, loc='right')
#     	i += 1
#         
# plt.title('India', y=12)
# plt.show()
# =============================================================================

print(np.mean(data[:,1]))
print(np.mean(data_test[:,1]))