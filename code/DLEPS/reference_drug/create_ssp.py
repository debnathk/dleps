import numpy as np
import pandas as pd

# Initialize Ray
ray.init()

root = 'code/DLEPS/reference_drug/'

# Read fingerprints
df_fgrps_train = pd.read_csv(root + 'fgrps_train.csv')
df_fgrps_ref = pd.read_csv(root + 'fgrps_ref.csv')
df_fgrps_train = np.array(df_fgrps_train)
df_fgrps_ref = np.array(df_fgrps_ref)
print(df_fgrps_ref.shape)
print(df_fgrps_train.shape)

# Calculate ssp
# ssp_list = []
# for i in range(df_fgrps_train.shape[0]):
#     ssp = np.zeros((df_fgrps_ref.shape[0], df_fgrps_ref.shape[1]))
#     for j in range(df_fgrps_ref.shape[0]):
#         for m in range(df_fgrps_train.shape[1]):
#             for n in range(df_fgrps_ref.shape[1]):
#                 if df_fgrps_train.iloc[i, m] == df_fgrps_ref.iloc[j, n]:
#                     ssp[j][n] = 1
#     ssp_list.append(ssp)

# for i in range(df_fgrps_train.shape[0]):
# for i in range(10):
#     # Create a boolean array where True indicates matching elements
#     match_array = np.isin(df_fgrps_train[i], df_fgrps_ref)

#     # Use broadcasting to create the final matrix
#     ssp = match_array[:, np.newaxis] * np.ones_like(df_fgrps_ref, dtype=int)
#     ssp_list.append(ssp)

# ssp_stack = np.stack(ssp_list)

# for i in range(10):
#     ssp = np.zeros((10, 10))
#     for j in range(10):
#         for m in range(10):
#             for n in range(10):
#                 if df_fgrps_train.iloc[i, m] == df_fgrps_ref.iloc[j, n]:
#                     ssp[j][n] = 1
#     ssp_list.append(ssp)

@ray.remote
def process_row(row_a, b):
    c = np.zeros((b.shape[0], b.shape[1]), dtype=int)
    
    for j in range(b.shape[0]):
        for m in range(row_a.shape[0]):
            for n in range(b.shape[1]):
                if row_a[m] == b[j][n]:
                    c[j][n] = 1
    
    return c

# Parallel processing using Ray
ssp_list = ray.get([process_row.remote(row_a, df_fgrps_ref) for row_a in df_fgrps_train])

# Shutdown Ray
ray.shutdown()

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


# import numpy as np

# a = np.array([[1, 2, 3],
#               [4, 5, 6],
#               [7, 8, 9]])

# b = np.array([[1, 2, 3],
#               [4, 5, 6]])

# clist = []

# print(result)
