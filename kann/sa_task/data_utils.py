
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf

# Special tokens
JOINER = '##'
UNK = '<unk>'
CLS = '<cls>'
SEP = '<sep>'


def identity(x):
  """Identity function."""
  return x


def compose(*funcs):
  """Returns a function that is the composition of multiple functions."""

  def wrapper(x):
    for func in reversed(funcs):
      x = func(x)
    return x

  return wrapper


def load_tokenizer(vocab_file, default_value=-1):
  """Loads a tokenizer from a vocab file."""

  # Build lookup table that maps subwords to ids.
  table = tf.lookup.TextFileInitializer(vocab_file, tf.string,
                                        tf.lookup.TextFileIndex.WHOLE_LINE,
                                        tf.int64,
                                        tf.lookup.TextFileIndex.LINE_NUMBER)
  static_table = tf.lookup.StaticHashTable(table, default_value)

  # Build tokenizer.
  tokenizer = text.WordpieceTokenizer(static_table,
                                      suffix_indicator=JOINER,
                                      max_bytes_per_word=100,
                                      max_chars_per_token=None,
                                      token_out_type=tf.int64,
                                      unknown_token=UNK)

  tokenizer2 = text.WordpieceTokenizer(static_table,
                                      suffix_indicator=JOINER,
                                      max_bytes_per_word=100,
                                      max_chars_per_token=None,
                                      token_out_type=tf.string,
                                      unknown_token=UNK)

  return tokenizer, tokenizer2


def pipeline(dset, preprocess_fun=identity, filter_fn=None, bufsize=1024):
  """Common (standard) dataset pipeline.
  Preprocesses the data, filters it (if a filter function is specified), caches it, and shuffles it.

  Note: Does not batch"""

  # Apply custom preprocessing.
  dset = dset.map(preprocess_fun)

  # Apply custom filter.
  if filter_fn is not None:
    dset = dset.filter(filter_fn)

  # Cache and shuffle.
  dset = dset.cache().shuffle(buffer_size=bufsize, seed=tf.random.set_seed(1234))

  return dset


def tokenize_fun(tokenizer):
  """Standard text processing function."""
  wsp = text.WhitespaceTokenizer()
  return compose(tokenizer.tokenize, wsp.tokenize, text.case_fold_utf8)


def padded_batch(dset, batch_size, sequence_length, label_shape=()):
  """Pads examples to a fixed length, and collects them into batches."""

  # We assume the dataset contains inputs, labels, and an index.
  padded_shapes = {
      'text': (sequence_length,),
      'inputs': (sequence_length,),
      'labels': label_shape,
      'index': (),
  }

  # Filter out examples longer than sequence length.
  dset = dset.filter(lambda d: d['index'] <= sequence_length)

  # Pad remaining examples to the sequence length.
  dset = dset.padded_batch(batch_size, padded_shapes)

  return dset


def load_tfds(name, split, preprocess_fun, filter_fn=None, data_dir=None):
  """Helper that loads a text classification dataset
  from tensorflow_datasets"""

  # Load raw dataset.
  dset = tfds.load(name, split=split, data_dir=data_dir)

  # Apply common dataset pipeline.
  dset = pipeline(dset, preprocess_fun=preprocess_fun, filter_fn=filter_fn)

  return dset


def imdb(split,
         vocab_file,
         sequence_length=1000,
         batch_size=64,
         transform=identity,
         filter_fn=None,
         data_dir=None):
  """Loads the imdb reviews dataset."""
  tokenizer, tokenizer2 = load_tokenizer(vocab_file)
  tokenize, tokenize2 = tokenize_fun(tokenizer), tokenize_fun(tokenizer2)

  def _preprocess(d):
    """Applies tokenization."""
    text = tokenize2(d['text']).flat_values
    tokens = tokenize(d['text']).flat_values
    preprocessed = {
        'text': text,
        'inputs': tokens,
        'labels': d['label'],
        'index': tf.size(tokens),
    }
    return transform(preprocessed)

  # Load dataset.
  dset = load_tfds('imdb_reviews',
                   split,
                   _preprocess,
                   filter_fn,
                   data_dir=data_dir)

  # Pad remaining examples to the sequence length.
  dset = padded_batch(dset, batch_size, sequence_length)

  return dset


def get_as_list(x):
    return [batch for batch in x]
