import numpy as np
import random
import argparse
import model_LSTM
import os.path


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=
        'Choose your options')
    
    parser.add_argument('--reset', type=bool, default=False, help=' train model from scratch ')
    parser.add_argument('--weights',type=str,default='test.hd5' ,help=' path to weights ')
    parser.add_argument('--path',type=str, help=' path to data ')
     
    args= parser.parse_args() #create an object having name and age as attributes)

    print(args)
    reset, weights, path = args.reset, 'data/saved_models/'+args.weights, args.path

    model = model_LSTM.NamesModel(path)
    
    if reset or not os.path.isfile(weights):
        model.train_model()
    else:
        model.load_model(weights)

    # model.predict()

    model.save_model(weights)