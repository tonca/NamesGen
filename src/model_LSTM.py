from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

# Insert the path to the data
path = "data/eng_M.csv"

names = open(path).read().lower().split()
random.shuffle(names)
text = ''.join(e+"\n" for e in names)

print('data length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


# cut the text in semi-redundant sequences of maxlen characters
maxlen = 20
step = 3
subsections = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    subsections.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(subsections))


print('Vectorization...')
x = np.zeros((len(subsections), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(subsections), len(chars)), dtype=np.bool)
for i, section in enumerate(subsections):
    for t, char in enumerate(section):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

print(y.shape)

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
model.fit(x, y,
          batch_size=128,
          epochs=80)


results = {}

start_index = random.randint(1,len(text)-maxlen)

for diversity in [0.2, 0.5, 1.0, 1.2]:
    print()
    print('----- diversity:', diversity)

    generated = ''
    section = text[start_index: start_index + maxlen]
    generated += section
    print('----- Generating with seed: "' + section + '"\n-----------\n')

    out_text = ''
    for i in range(600):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(section):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]

        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        section = section[1:] + next_char

        out_text = out_text + next_char

    results[diversity] = set(out_text.split()[1:-1])

    print('Diversity: {}'.format(diversity))
    print(results[diversity])

    collisions = len(set(names).intersection(results[diversity]))
    print('n collisions for diversity {}: {}'.format(diversity,collisions))
