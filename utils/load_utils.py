def load_data(file):
    import pickle

    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
        
    return data_dict
