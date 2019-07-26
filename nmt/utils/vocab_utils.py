# Copyright 2017 Google Inc. All Rights Reserved.
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

"""Utility to handle vocabularies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import os
import tensorflow as tf
import numpy
from tensorflow.python.ops import lookup_ops

from ..utils import misc_utils as utils


UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
UNK_ID = 0


def load_vocab(vocab_file):
  vocab = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
    vocab_size = 0
    for word in f:
      vocab_size += 1
      vocab.append(word.strip())
  return vocab, vocab_size


def check_vocab(vocab_file, out_dir,v_size, check_special_token=True, sos=None,
                eos=None, unk=None):
  """Check if vocab_file doesn't exist, create from corpus_file."""
  f=open(out_dir+"/point_vocab.txt",'w')
  for i in range(0,1000):
    f.write(str(i)+'\n')
  f.write('-1'+'\n')
  if tf.gfile.Exists(vocab_file):
    utils.print_out("# Vocab file %s exists" % vocab_file)
    vocab, vocab_size = load_vocab(vocab_file)
    if check_special_token:
      # Verify if the vocab starts with unk, sos, eos
      # If not, prepend those tokens & generate a new vocab file
      if not unk: unk = UNK
      if not sos: sos = SOS
      if not eos: eos = EOS
      assert len(vocab) >= 3
      if vocab[0] != unk or vocab[1] != sos or vocab[2] != eos:
        utils.print_out("The first 3 vocab words [%s, %s, %s]"
                        " are not [%s, %s, %s]" %
                        (vocab[0], vocab[1], vocab[2], unk, sos, eos))
        vocab = [unk, sos, eos] + vocab
        vocab_size +=3
        vocab_size=min(vocab_size,v_size)
        vocab=vocab[:vocab_size]
        new_vocab_file = os.path.join(out_dir, os.path.basename(vocab_file))
        with codecs.getwriter("utf-8")(
            tf.gfile.GFile(new_vocab_file, "wb")) as f:
          for word in vocab:
            f.write("%s\n" % word)
        vocab_file = new_vocab_file
  else:
    raise ValueError("vocab_file '%s' does not exist." % vocab_file)

  vocab_size = len(vocab)
  return vocab_size, vocab_file


def create_vocab_tables(src_vocab_file, tgt_vocab_file, share_vocab):
  """Creates vocab tables for src_vocab_file and tgt_vocab_file."""
  src_vocab_table = lookup_ops.index_table_from_file(
      src_vocab_file, default_value=UNK_ID)
  if share_vocab:
    tgt_vocab_table = src_vocab_table
  else:
    tgt_vocab_table = lookup_ops.index_table_from_file(
        tgt_vocab_file, default_value=UNK_ID)
  return src_vocab_table, tgt_vocab_table

from tqdm import tqdm
def count_lines(fname):
    with open(fname) as f:
        return sum(1 for line in f)
def load_embed_txt(embed_file,vocab_file):
  emb_dict = dict()
  emb_size = 300
  vocab=dict()
  with open(vocab_file,'r') as f:
        for line in f:
            tokens = line.strip()
            vocab[tokens]=1
  for word in vocab:
    emb_dict[word]=list(0.01*numpy.random.randn(emb_size))
  with open(embed_file, 'r', encoding='iso-8859-1') as f:
    for line in tqdm(f,total=count_lines(embed_file)):
      temp = line.split()
      word = temp[0]
      vector = temp[1:]
      if word in vocab:
            try:
                emb_dict[word]=list(map(float, vector))
                cont+=1
            except:
                continue
  return emb_dict, emb_size
