import numpy as np
import pandas as pd


# root = 'code/DLEPS/dleps/code/DLEPS/reference_drug/'

# Read fingerprints
df_fgrps_train = pd.read_csv('fgrps_train.csv')
df_fgrps_ref = pd.read_csv('fgrps_ref.csv')

# Calculate ssp
ssp_list = []
for i in range(df_fgrps_train.shape[0]):
    ssp = np.zeros((df_fgrps_ref.shape[0], df_fgrps_ref.shape[1]))
    for j in range(df_fgrps_ref.shape[0]):
        for m in range(df_fgrps_train.shape[1]):
            for n in range(df_fgrps_ref.shape[1]):
                if df_fgrps_train[i][m] == df_fgrps_ref[j][n]:
                    ssp[j][n] = 1
    ssp_list.append(ssp)

ssp_stack = np.stack(ssp_list)
ssp_stack.shape

# Split train, test
TEST_SIZE = 75
ssp_train = ssp_stack[TEST_SIZE:]
ssp_test = ssp_stack[:TEST_SIZE]

# Save dataset
import h5py

h5f = h5py.File('ssp_data_train.h5', 'w')
h5f.create_dataset('data', data=ssp_train)
h5f.close()

h5f = h5py.File('ssp_data_test.h5', 'w')
h5f.create_dataset('data', data=ssp_test)
h5f.close()

# Read datset
h5f = h5py.File('ssp_data_train.h5', 'r')
ssp_train = h5f['data'][:]
h5f = h5py.File('ssp_data_test.h5', 'r')
ssp_test = h5f['data'][:]

print(ssp_train.shape)
print(ssp_test.shape)



