from __future__ import absolute_import, division, print_function, unicode_literals
from typing import List, Dict, Set, Tuple

import logging

import numpy as np
import os
import datetime

import tensorflow as tf

import ml_data as data

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_datasets(batch_size: int,
                 data_root: str = './data/',
                 buffer_size: int = 10000,
                 seq_length: int = 200) -> Tuple[int, Set[str], np.ndarray, tf.data.Dataset, tf.data.Dataset]:
    logger.info("Looking through data...")

    logger.info("Looking through training data...")
    tr_count, tr_text = data.txt_from_dir(os.path.join(data_root, "training"))
    logger.info("Found %d training files.", tr_count)

    logger.info("Looking through validation data...")
    val_count, val_text = data.txt_from_dir(os.path.join(data_root, "validation"))
    logger.info("Found %d validation files.", val_count)

    logger.info("Preparing data...")
    vocab, char2int, int2char = data.extract_vocab(tr_text + val_text)
    vocab_len = len(vocab)

    tr_data = data.parse_text(vocab, char2int, tr_text)
    tr_dataset = data.data_into_dataset(
        tr_data, batch_size, buffer_size, seq_length)
    val_data = data.parse_text(vocab, char2int, val_text)
    val_dataset = data.data_into_dataset(
        val_data, batch_size, buffer_size, seq_length)

    logger.info("Training text size:       \t%d", len(tr_data))
    logger.info("Validation text size:     \t%d", len(val_text))
    logger.info("Training:validation ratio:\t%f", len(tr_data) / len(val_text))

    return vocab_len, char2int, int2char, tr_dataset, val_dataset


def build_model(vocab_size: int, embedding_dim: int, rnn_units: int, batch_size: int) -> tf.keras.Model:
    logger.info("Building model...")
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


def train_model(model: tf.keras.Model,
                tr_dataset: tf.data.Dataset,
                val_dataset: tf.data.Dataset,
                checkpoint_dir: str,
                epochs: int = 100,
                patience: int = 10):
    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=loss)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=patience)

    logger.info("Begin training... (this will take a while)")
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_best_only=True,
        save_weights_only=True)
    history = model.fit(tr_dataset, epochs=epochs, callbacks=[
                        checkpoint_callback, early_stop], validation_data=val_dataset)
    logger.info(
        "Training stopped, no improvement after {} epochs".format(patience))


def generate_text(model,
                  char2int: Set[str],
                  int2char: np.ndarray,
                  start_string: str):
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
            predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(int2char[predicted_id])
    return (start_string + ''.join(text_generated))