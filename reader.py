import copy
import math
import pandas as pd
import tensorflow as tf

data_seq_train = pd.read_csv('data_seq_train.txt', header=None, sep='\t')
data_sec_train = pd.read_csv('data_sec_train.txt', header=None, sep='\t')
data_seq_train = data_seq_train.values.tolist()
data_sec_train = data_sec_train.values.tolist()
data_seq = []
data_seq.extend([d[0] for d in data_seq_train])
data_sec = []
data_sec.extend([d[0].replace(' ', 'J') for d in data_sec_train])
idx = [i for i in range(len(data_sec)) if len(data_sec[i]) != len(data_seq[i])]
### data_remedy
for ids in idx:
    data_sec[ids] += 'J' * (len(data_seq[ids]) - len(data_sec[ids]))

seq_char = list(set(''.join(data_seq)))
sec_char = list(set(''.join(data_sec)))
seq_dict = dict([(seq_char[i], i + 1) for i in range(len(seq_char))])
sec_dict = dict([(sec_char[i], i + 1) for i in range(len(sec_char))])


class Reader(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size, seq_dict, sec_dict):
        self.x = data[0]
        self.y = data[1]
        self.batch_size = batch_size
        dx = []
        dy = []
        for x in self.x:
            tx = list(x)
            tx = [seq_dict[tx[i]] for i in range(len(tx))]
            dx.append(tx)
        for y in self.y:
            ty = list(y)
            ty = [sec_dict[ty[i]] for i in range(len(ty))]
            dy.append(ty)
        self.x = dx
        self.y = dy

    def __len__(self):
        if len(self.x) % self.batch_size == 0:
            return len(self.x) // self.batch_size
        else:
            return math.ceil(len(self.x) / self.batch_size) - 1

    def make_data(self, x, y, size):
        max_seq_len = 0
        seq_lens = []
        for example in x:
            max_seq_len = max(max_seq_len, len(example))
            seq_lens.append(len(example))

        for i in range(len(x)):
            x[i] += [0] * (max_seq_len - seq_lens[i])
            y[i] += [0] * (max_seq_len - seq_lens[i])
        mask = tf.sequence_mask(seq_lens)
        x = tf.convert_to_tensor(x, dtype=tf.int32)
        y = tf.convert_to_tensor(y, dtype=tf.int32)
        return [x, mask], y

    def __getitem__(self, idx):
        x = copy.deepcopy(self.x[idx * self.batch_size:(idx + 1) * self.batch_size])
        y = copy.deepcopy(self.y[idx * self.batch_size:(idx + 1) * self.batch_size])
        return self.make_data(x, y, self.batch_size)


train_dataset = Reader([data_seq, data_sec], 32, seq_dict, sec_dict)

if __name__ == '__main__':
    reader = Reader([data_seq, data_sec], 32, seq_dict, sec_dict)
    for batch in reader:
        pass
    print("iter1 finished")
    for batch in reader:
        pass
    print("iter2 finished")
