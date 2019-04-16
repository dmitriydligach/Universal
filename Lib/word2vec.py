import numpy as np
import os, os.path

def write_vectors(alphabet, weights, path):
  """Write model to file given an alphabet and weights"""

  # number of words in alphabet must be the same
  # as the number of rows in weights matrix
  # alphabet maps words to integers
  # integers are indicies into weights matrix

  with open(path, 'w') as outfile:
    outfile.write('%s %s\n' % (len(alphabet), weights.shape[1]))
    for word, index in list(alphabet.items()):
      vector = [str(element) for element in weights[index]]
      vector_as_string = ' '.join(vector)
      outfile.write('%s %s\n' % (word, vector_as_string))

class Model:
  """Represents a word2vec model"""

  def __init__(self, path, verbose=False):
    """Initiaize from a word2vec model file"""

    self.count = None      # number of vectors
    self.dimensions = None # number of dimensions
    self.vectors = {}      # key: word, value: numpy vector
    self.verbose = verbose # verbosity

    with open(path) as file:
      for line in file:
        elements = line.strip().split()
        if len(elements) < 5: # parse header
          self.count = int(elements[0])
          self.dimensions = int(elements[1])
          continue
        word = elements[0]
        vector = [float(element) for element in elements[1:self.dimensions+1]]
        self.vectors[word] = np.array(vector)

  def select_vectors(self, alphabet):
    """Return vectors for items in alphabet"""

    average = self.average_words(list(alphabet.keys()))
    vecs = np.zeros((len(alphabet), self.dimensions))

    oov_count = 0
    for word, index in list(alphabet.items()):
      if word in self.vectors:
        vecs[index, :] = self.vectors[word]
      else:
        # also tried np.random.uniform(low=-0.25, high=0.25, size=self.dimensions)
        vecs[index, :] = average
        oov_count = oov_count + 1

    oov_rate = float(oov_count) / len(alphabet)
    if(self.verbose == True):
      print('embedding oov rate: %s%%' % (round(oov_rate * 100, 2)))
    return vecs

  def average_words(self, words):
    """Compute average vector for a list of words"""

    words_found = 0 # words in vocabulary
    sum = np.zeros(self.dimensions)
    for word in words:
      if word in self.vectors:
        sum = sum + self.vectors[word]
        words_found = words_found + 1

    if words_found == 0:
      return sum
    else:
      return sum / words_found

  def words_to_vectors(self, infile, outfile):
    """Convert texts from infile to vectors and save in outfile"""

    matrix = []
    with open(infile) as file:
      for line in file:
        average = self.average_words(line.split())
        matrix.append(list(average))

    np.savetxt(outfile, np.array(matrix))

if __name__ == "__main__":

  path = '/Users/Dima/Loyola/Data/Word2Vec/Models/GoogleNews-vectors-negative300.txt'
  model = Model(path)

  data = '/Users/Dima/Soft/CnnBritz/cnn-text-classification-tf/data/rt-polaritydata/'
  model.words_to_vectors(os.path.join(data, 'rt-polarity.neg'), 'neg.txt')
  model.words_to_vectors(os.path.join(data, 'rt-polarity.pos'), 'pos.txt')
