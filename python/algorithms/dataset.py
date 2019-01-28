import json
import tensorflow as tf

class JSONDataset(object):

    def __init__(self, filename):
        self.filename = filename
        with open(self.filename, 'rb') as json_in:
            file = json.load(json_in)
            self.iter = iter(list(file.items()))


    def input_func(self):
        k,v = self.iter.__next__()
        return (tf.constant(k, dtype=tf.string), tf.constant(v, dtype=tf.int64))
