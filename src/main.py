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
    parser.add_argument('--epochs',type=int, default=60, help=' number of epochs in training ')
    parser.add_argument('--diversity', type=float, default=1., help=' degree of diversity in generation ')
     
    args= parser.parse_args() #create an object having name and age as attributes)

    print(args)
    reset = args.reset 
    weights = 'data/saved_models/'+args.weights
    path = args.path
    epochs = args.epochs
    diversity = args.diversity

    model = model_LSTM.NamesModel(path)
    
    if reset or not os.path.isfile(weights):
        model.train_model(epochs)
    else:
        model.load_model(weights)

    model.predict(diversity)

    model.save_model(weights)