import pickle

def extract_configuration(path):
    
    with open(path, 'rb') as f:
        conf = pickle.load(f)
    
    return {
        key.split('__')[-1]: val
        for key, val in conf.items()
    }
