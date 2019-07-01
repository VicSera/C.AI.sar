import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import os
import matplotlib.pyplot as plt

# ------- Checkpoint paths for all authors ---------- #

path_caesar = './training_data/training_data_caesar'
path_petronius = './training_data/training_data_petronius'
path_ovid = './training_data/training_data_ovid'
path_horace = './training_data/training_data_horace'
path_cicero = './training_data/training_data_cicero'

training_data = {
    'Caesar' : path_caesar,
    'Petronius' : path_petronius,
    'Ovid' : path_ovid,
    'Horace' : path_horace,
    'Cicero' : path_cicero
}

# ------- Input and formatting ------------- #

vocab = ['\n', ' ', '!', ',', '.', ':', ';', '?', 'A', 'B'
, 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R'
, 'S', 'T', 'U', 'V', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g'
, 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x'
, 'y', 'z', '-']

def format_text(text, vocab):
    new_text = ''
    for char in text:
        if char in vocab and not(char == ' ' and new_text.endswith(' ')):
            new_text += char
    return new_text

text = '' # text to classify
with open('./text_to_classify.txt', 'r') as file: # convenient way to open a file without having to actually close it
    text = format_text(file.read(), vocab) # read input

print(len(text), "characters given")

char2id = {u:i for i, u in enumerate(vocab)} # dictionary from char to id
id2char = np.array(vocab) # dictionary from id to char (array, coincidentally)

text_as_int = np.array([char2id[c] for c in text]) # convert all text to numbers


# -------- Preparing the input in batches for the model ------------- #

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequence = char_dataset.batch(len(text), drop_remainder=False)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequence.map(split_input_target)
BATCH_SIZE = len(text)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)

vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

# -------- Creating the model -------------- #

import functools
rnn = functools.partial(
    tf.keras.layers.GRU, recurrent_activation='sigmoid')

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, 
                              batch_input_shape=[batch_size, None]),
        rnn(rnn_units,
            return_sequences=True, 
            recurrent_initializer='glorot_uniform',
            stateful=True),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(
    vocab_size = len(vocab), 
    embedding_dim=embedding_dim, 
    rnn_units=rnn_units, 
    batch_size=1)

# ----- Guessing the author -------- #

def get_loss(model, dataset, path):
    saved_checkpoints_dir = path # get checkpoints from specified path
    tf.train.latest_checkpoint(saved_checkpoints_dir) # train and load the checkpoint
    model.load_weights(tf.train.latest_checkpoint(saved_checkpoints_dir))
    model.build(tf.TensorShape([1, None]))
    hidden = model.reset_states()
    for (batch_n, (inp, target)) in enumerate(dataset):
        predictions = model(inp)
        loss = float(tf.losses.sparse_softmax_cross_entropy(target, predictions))
    return loss # get loss for the provided dataset, based on the current model

best_prediction = ''
lowest_loss = 10

authors = ['Caesar', 'Petronius', 'Ovid', 'Horace', 'Cicero']
losses = []

print("Loss per author (how far off the prediction algorithm was; lower = better):")
for author in training_data:
    loss = get_loss(model, dataset, training_data[author])
    losses.append(loss)
    if (loss < lowest_loss):
        best_prediction = author
        lowest_loss = loss
    print(author, ':', '{:.4f}'.format(loss))

print("\n\nFinal results:")
def format_losses(losses):
    lowest = min(losses)
    for iterator in range(len(losses)):
        losses[iterator] = lowest / losses[iterator]
    s = sum(losses)
    for iterator in range(len(losses)):
        losses[iterator] = losses[iterator] * 100 / s
        print(authors[iterator] + ' ' + '{:.4f}'.format(losses[iterator]) + '%')

print("Best guess:", best_prediction, ':', '{:.4f}'.format(lowest_loss), ' loss\n')
format_losses(losses)

colors = ['grey'] * 5
index_of_best = 0
for i in range(len(authors)):
    if authors[i] == best_prediction:
        colors[i] = 'red'
        index_of_best = i;

plt.bar(authors, losses, color = colors)
plt.ylabel("Certainty in %")
plt.xlabel("Author")
plt.title("Percentages based on loss values")
plt.show()
input()
