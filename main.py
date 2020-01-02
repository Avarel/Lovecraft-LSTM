from __future__ import absolute_import, division, print_function, unicode_literals
from typing import List, Dict, Set, Tuple

import logging

import numpy as np
import os
import datetime

import tensorflow as tf

logging.basicConfig()
logger = logging.getLogger("lovecraft-lstm")
logger.setLevel(logging.DEBUG)

def txt_data(dir: str) -> Tuple[int, str]:
    """Obtain a string from appending all the data within a directory.

    :param dir (str): the directory to scan .txt files from

    :rtype (int, str):
    :returns int: Number of files found.
    :returns str: All of the text in the directory as one string
    """
    text = ''
    count = 0
    for file in os.listdir(dir):
        if file.endswith('.txt'):
            logger.debug("Found file %s", file)
            count += 1
            text += open(os.path.join(dir, file)).read()
    return count, text


def extract_vocab(text: str) -> Tuple[List[str], Set[str], np.ndarray]:
    """Extracts the vocabulary, the character-integer mapping, 
    and the integer-character mapping from the text.

    :param text (str): the text to extract information from

    :rtype (list, set, array):
    :returns vocab (list): The vocabulary
    :returns char2int (set): The character-integer mapping
    :returns int2char (array): The integer-character mapping
    """
    vocab = sorted(set(text))
    return vocab, {c: i for i, c in enumerate(vocab)}, np.array(vocab)


def parse_text(vocab: List[str], char2int: np.ndarray, text: str) -> np.ndarray:
    return np.array([char2int[ch] for ch in text], dtype=np.int32)


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def prep_dataset(data: np.ndarray, batch_size: int, buffer_size: int, seq_length: int) -> tf.data.Dataset:
    char_dataset = tf.data.Dataset.from_tensor_slices(data)
    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

    return sequences.map(split_input_target).shuffle(buffer_size).batch(batch_size, drop_remainder=True)


def build_model(vocab_size: int, embedding_dim: int, rnn_units: int, batch_size: int) -> tf.keras.Model:
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(
            vocab_size, embedding_dim,
            batch_input_shape=[batch_size, None]
        ),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(
            rnn_units,
            return_sequences=True,
            stateful=True,
            recurrent_initializer='glorot_uniform'
        ),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(
            rnn_units,
            return_sequences=True,
            stateful=True,
            recurrent_initializer='glorot_uniform'
        ),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(vocab_size)
    ])


# batch_size = 64
# buffer_size = 10000
# embedding_dim = 256
# epochs = 50
# seq_length = 200
# examples_per_epoch = len(text)//seq_length
# #lr = 0.001 #will use default for Adam optimizer
# rnn_units = 1024
# vocab_size = len(vocab)

def time_for_file():
    return datetime.datetime.now().strftime("_%m.%d.%y-%H.%M.%S")


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def run_model(
        model: tf.keras.Model,
        checkpoint_dir: str,
        tr_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        epochs: int = 50,
        patience: int = 10
):
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=loss)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=patience)

    logger.info("Begin training... (this will take a while)")
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
    history = model.fit(tr_dataset, epochs=epochs, callbacks=[
                        checkpoint_callback, early_stop], validation_data=val_dataset)
    logger.info("Training stopped as there was no improvement after {} epochs".format(patience))


def generation(
        batch_size: int = 64,
        buffer_size: int = 10000,
        embedding_dim: int = 256,
        seq_length: int = 200,
        rnn_units: int = 1024,
):
    logger.info("Looking through data...")

    logger.info("Looking through training data...")
    tr_count, tr_text = txt_data("./data/training")
    logger.info("Found %d training files.", tr_count)

    logger.info("Looking through validation data...")
    val_count, val_text = txt_data("./data/validation")
    logger.info("Found %d validation files.", val_count)

    logger.info("Preparing data...")
    vocab, char2int, int2char = extract_vocab(tr_text + val_text)
    vocab_len = len(vocab)

    tr_data = parse_text(vocab, char2int, tr_text)
    tr_dataset = prep_dataset(tr_data, batch_size, buffer_size, seq_length)
    val_data = parse_text(vocab, char2int, val_text)
    val_dataset = prep_dataset(val_data, batch_size, buffer_size, seq_length)

    logger.info("Training text size:       \t%d", len(tr_data))
    logger.info("Validation text size:     \t%d", len(val_text))
    logger.info("Training:validation ratio:\t%f", len(tr_data) / len(val_text))

    logger.info("Building model...")

    model = build_model(
        vocab_size=vocab_len,
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=batch_size
    )

    model.summary()

    # return

    checkpoint_dir = os.path.join(
        './checkpoints/', 'checkpoint' + time_for_file())

    run_model(model, checkpoint_dir, tr_dataset, val_dataset)

    model = build_model(vocab_len, embedding_dim, rnn_units, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))


    def generate_text(model, start_string):
        logger.info('Generating with seed: "%s"', start_string)

        num_generate = 1000
        input_eval = [char2int[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)
        text_generated = []
        temperature = 1.0
        model.reset_states()
        for i in range(num_generate):
            predictions = model(input_eval)
            predictions = tf.squeeze(predictions, 0)
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(
                predictions,      num_samples=1)[-1, 0].numpy()
            input_eval = tf.expand_dims([predicted_id], 0)
            text_generated.append(int2char[predicted_id])
        return (start_string + ''.join(text_generated))
    logger.info(generate_text(model, start_string="the deep dark"))

generation()
