import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import os
import time

# ------- Input and formatting ------------- #

input_caesar = './input_caesar.txt'
save_caesar = "./training_data_caesar\\"

input_petronius = './input_petronius.txt'
save_petronius = "./training_data_petronius\\"

input_ovid = './input_ovid.txt'
save_ovid = "./training_data_ovid\\"

input_horace = './input_horace.txt'
save_horace = "./training_data_horace\\"

input_cicero = './input_cicero.txt'
save_cicero = "./training_data_cicero\\"

authors = {
    input_caesar : save_caesar,
    input_petronius : save_petronius,
    input_ovid : save_ovid,
    input_horace : save_horace,
    input_cicero : save_cicero
}

vocab = ['\n', ' ', '!', ',', '.', ':', ';', '?', 'A', 'B'
    , 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R'
    , 'S', 'T', 'U', 'V', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g'
    , 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x'
    , 'y', 'z', '-']

def format_text(text, vocab):
    new_text = ''
    for char in text:
        if char in vocab and not(char == ' ' and new_text[-1:] == ' '):
            new_text += char
    return new_text

print(authors)

for input in authors:
    save = authors[input]
    print("Currently at " + input)
    text = ''
    with open(input, 'r') as file: # convenient way to open a file without having to actually close it
        text = format_text(file.read(), vocab) # read input

    print(len(text), " chars")

    char2id = {u:i for i, u in enumerate(vocab)} # dictionary from char to id
    id2char = np.array(vocab) # dictionary from id to char (array, coincidentally)

    text_as_int = np.array([char2id[c] for c in text]) # convert all text to numbers


    # -------- Preparing the input in batches for the model ------------- #

    seq_length = 100 # longest sentence to take (TODO: tweak this number)
    examples_per_epoch = len(text)//seq_length

    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = sequences.map(split_input_target)

    BATCH_SIZE = 64
    steps_per_epoch = examples_per_epoch//BATCH_SIZE
    BUFFER_SIZE = 10000

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

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
        batch_size=BATCH_SIZE)



    # ----- Checkpoints ------- #

    # Directory where the checkpoints will be saved
    checkpoint_dir = save
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    #to resume training
    #saved_checkpoints_dir = "D:\\Programare\\Programe\\Python\\ML\\cAIsar\\saved_checkpoints\\"
    #UNCOMMENT TO RESTORE SAVE
    tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    # ----- Training -------- #
    optimizer = tf.train.AdamOptimizer()
    EPOCHS=5

    for epoch in range(EPOCHS):
        start = time.time()
        hidden = model.reset_states()
    
        for (batch_n, (inp, target)) in enumerate(dataset):
              with tf.GradientTape() as tape:
                  predictions = model(inp)
                  loss = tf.losses.sparse_softmax_cross_entropy(target, predictions)
              grads = tape.gradient(loss, model.variables)
              optimizer.apply_gradients(zip(grads, model.variables))
              if batch_n % 5 == 0:
                  template = 'Epoch {} Batch {} Loss {:.4f}'
                  print(template.format(epoch+1, batch_n, loss))
        #if (epoch + 1) % 5 == 0:
        model.save_weights(checkpoint_prefix)
        print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
        print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

