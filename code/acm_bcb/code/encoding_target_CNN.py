# Encoding of target proteins is adopted form DeepPurpose - https://github.com/kexinhuang12345/DeepPurpose/tree/master
import pandas as pd

# '?' padding
amino_char = ['?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O',
       'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z']

MAX_SEQ_PROTEIN = 1000

def trans_protein(x):
	temp = list(x.upper())
	temp = [i if i in amino_char else '?' for i in temp]
	if len(temp) < MAX_SEQ_PROTEIN:
		temp = temp + ['?'] * (MAX_SEQ_PROTEIN-len(temp))
	else:
		temp = temp [:MAX_SEQ_PROTEIN]
	return temp

def encode_protein(df_data, target_encoding='CNN', column_name = 'BindingDB Target Chain Sequence', save_column_name = 'target_encoding'):
	print('encoding protein...')
	print('unique target sequence: ' + str(len(df_data[column_name].unique())))
	if target_encoding == 'CNN':
		AA = pd.Series(df_data[column_name].unique()).apply(trans_protein)
		AA_dict = dict(zip(df_data[column_name].unique(), AA))
		df_data[save_column_name] = [AA_dict[i] for i in df_data[column_name]]
		# the embedding is large and not scalable but quick, so we move to encode in dataloader batch. 
	else:
		raise AttributeError("Please use the correct protein encoding available!")
	return df_data

if __name__ == "__main__":
	df = pd.read_csv('../data/BindingDB_IC50_human.csv')
	df_sample = df.head(10)
	df_sample_encoded = encode_protein(df_sample)
	df_sample_encoded.to_csv('../data/sample_encoding.csv', index=False)