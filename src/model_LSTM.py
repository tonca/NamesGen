from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random


class NamesModel:

    MAX_LENGTH = 20
    
    def __init__(self, path):
        
        # Insert the path to the data
        self.names = open(path).read().lower().split()
        random.shuffle(self.names)
        self.text = ''.join(e+"\n" for e in self.names)

        print('data length:', len(self.text))

        self.chars = sorted(list(set(self.text)))
        print('total chars:', len(self.chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))


        # Cut the text in semi-redundant sequences of MAX_LENGTH characters
        step = 3
        subsections = []
        next_chars = []
        for i in range(0, len(self.text) - self.MAX_LENGTH, step):
            subsections.append(self.text[i: i + self.MAX_LENGTH])
            next_chars.append(self.text[i + self.MAX_LENGTH])
        print('nb sequences:', len(subsections))


        print('Vectorization...')
        self.x = np.zeros((len(subsections), self.MAX_LENGTH, len(self.chars)), dtype=np.bool)
        self.y = np.zeros((len(subsections), len(self.chars)), dtype=np.bool)
        for i, section in enumerate(subsections):
            for t, char in enumerate(section):
                self.x[i, t, self.char_indices[char]] = 1
            self.y[i, self.char_indices[next_chars[i]]] = 1

        print('Build model...')
        self.model = Sequential()
        self.model.add(LSTM(128, input_shape=(self.MAX_LENGTH, len(self.chars))))
        self.model.add(Dense(len(self.chars)))
        self.model.add(Activation('softmax'))

        optimizer = RMSprop(lr=0.01)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        print(self.x.shape)
        print(self.y.shape)


    def load_model(self, path):
        print( "Loading weights form {}".format(path) )
        self.model.load_weights(path)

    def save_model(self, path):
        print("Weights stored at {}".format(path))
        self.model.save_weights(path)

    def train_model(self, epochs):
        
        self.model.fit(self.x,
            self.y,
            batch_size=128,
            epochs=epochs)


    @staticmethod
    def sample(preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)


    def predict(self, diversity):

        start_index = random.randint(1,len(self.text)-self.MAX_LENGTH)

        print()
        print('----- diversity:', diversity)

        section = self.text[start_index: start_index + self.MAX_LENGTH]
        print('----- Generating with seed: "' + section + '"\n-----------\n')

        out_text = ''
        for i in range(600):
            x_pred = np.zeros((1, self.MAX_LENGTH, len(self.chars)))
            for t, char in enumerate(section):
                x_pred[0, t, self.char_indices[char]] = 1.

            preds = self.model.predict(x_pred, verbose=0)[0]

            next_index = self.sample(preds, diversity)
            next_char = self.indices_char[next_index]

            section = section[1:] + next_char

            out_text = out_text + next_char

        results = set(out_text.split()[1:-1])

        print('Diversity: {}'.format(diversity))
        print(results)

        collisions = len(set(self.names).intersection(results))
        print('n collisions for diversity {}: {}'.format(diversity,collisions))