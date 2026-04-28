import pandas as pd
import numpy as np
import pickle
def load_numerai_data():
    print("# Loading data...")
    train = pd.read_csv('numerai_training_data.csv', header=0)

    trainEras = train['era'] == 'era1'#['era' + str(i) for i in range(1, 20)]
    testEras = train['era'] == 'era2'#['era' + str(i) for i in range(2,4)]

    testBoolArr = [i in testEras for i in train['era'].values]
    test = train[testEras]

    #trainBoolArr = [i in trainEras for i in train['era'].values]
    #era = train['era'] in ['era' + str(i) for i in range(1,60)]
    train = train[trainEras]

    train_bernie = train.drop([
        'id', 'era', 'data_type', 'target_charles', 'target_elizabeth',
        'target_jordan', 'target_ken', 'target_frank', 'target_hillary'
    ],axis=1)

    tournament = pd.read_csv('numerai_tournament_data.csv', header=0)

    validation = tournament[tournament['data_type'] == 'validation']
    #test = tournament[tournament['data_type'] == 'test']
    # Transform the loaded CSV data into numpy arrays
    features = [f for f in list(train_bernie) if "feature" in f]
    data = dict()
    data['x_train'] = train_bernie[features]
    data['y_train'] = train_bernie['target_bernie']
    data['x_validation'] = validation[features]
    data['y_validation'] = validation['target_bernie']
    data['x_test'] = test[features]
    data['y_test'] = test['target_bernie']
    return data

def load_optimized_models(file):
    return pickle.load(open(file,'rb'))
def pickleIt(obj,fname):
    """
    Saves any object to the local directory

    Args:
        obj: Object to be saved.
        fname: Name of file.

    Returns:
        None
    """
    pickle_out = open(fname + ".pickle", "wb")
    pickle.dump(obj, pickle_out)
    pickle_out.close()
