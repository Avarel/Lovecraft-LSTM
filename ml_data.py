from __future__ import absolute_import, division, print_function, unicode_literals
from typing import List, Dict, Set, Tuple

import logging

import numpy as np
import os
import datetime

import tensorflow as tf


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def time_for_file():
    return datetime.datetime.now().strftime("_%m.%d.%y-%H.%M.%S")


def txt_from_dir(dir: str) -> Tuple[int, str]:
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
            text += open(os.path.join(dir, file)).read() + '\n'
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


def data_into_dataset(data: np.ndarray, batch_size: int, buffer_size: int, seq_length: int) -> tf.data.Dataset:
    char_dataset = tf.data.Dataset.from_tensor_slices(data)
    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

    return sequences.map(split_input_target).shuffle(buffer_size).batch(batch_size, drop_remainder=True)