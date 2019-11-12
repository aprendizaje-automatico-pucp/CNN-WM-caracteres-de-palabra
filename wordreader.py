    # Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing language text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

import tensorflow as tf

def _read_words(filename):
  """Read as list of words.
  Split words on whitespace.
  Returns list of words each word a list of sequences of characters.
  """
  with tf.gfile.GFile(filename, "r") as f:
      # Read words as list.
      words = f.read().splitlines()
      # Split words into sequences on whitespace.
      return [word.split() for word in words]

def _build_vocab(filename):
  data = _read_words(filename)
  # data is a list of words, each a list of character sequences.
  # Counter expects a simple list.
  data = [seq for word in data for seq in word]
  counter = collections.Counter(data)
  # Frequency of 1 sequences we accumulate in <unk/> and delete from dict.
  unk = [c[0] for c in counter.items() if c[1] == 1]
  counter['<unk>'] = len(unk)
  for u in unk: del counter[u]
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  # Add nul sequence as initial.
  count_pairs.insert(0, ('<nul>', 0))
  # Add <s> sequence for word separation.
  count_pairs.insert(1,('<s>', 0))
  # Cool! Split sorted key, value pairs and index the keys.
  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))
  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  """Substitute ids for words encountered in training.
  Skips sequencias not previously defined, 
  but we hope to define all > 1 in frequency.
  data is list of words, each word a list of sequences.
  """
  data = _read_words(filename)
  return [[word_to_id[seq]for seq in word if seq in word_to_id] for word in data]

def lang_word_data(data_path=None, lang=None):
  """Load language word data from data directory "data_path".

  Reads language word text files, converts strings to integer ids,
  and packages in list of word_ids.

  The language datasets come from WOLD - world loanword study:

  Args:
    data_path: string path to the directory where language word tables 
      have been placed.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects is a list of word_id lists.
  """

  all_path = os.path.join(data_path, lang+".txt")
  train_path = os.path.join(data_path, lang+".train.txt")
  valid_path = os.path.join(data_path, lang+".valid.txt")
  test_path = os.path.join(data_path, lang+".test.txt")

  word_to_id = _build_vocab(all_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = word_to_id
  return train_data, valid_data, test_data, vocabulary

