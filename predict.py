import tensorflow as tf
from TENER import TENER
import tensorflow_addons as tfa
import json
from reader import inverse_sec_dict, inverse_seq_dict
from reader import test_dataset

output_dir = './model'
optimizer = tf.keras.optimizers.Adam(lr=0.01)
model = TENER(8, 23, 128, 2, 128, 2, 256, 0.5, True, 0.3, False, 0.3)
ckpt = tf.train.Checkpoint(optimizer = optimizer, model = model)
ckpt.restore(tf.train.latest_checkpoint(output_dir))

for _,(text_batch,labels_batch) in enumerate(test_dataset):
    logits, text_lens = model.predict(text_batch)
    paths = []
    for logit, text_len in zip(logits, text_lens):
        viterbi_path, _ = tfa.text.viterbi_decode(logit[:text_len],model.transition_params)
        paths.append(viterbi_path)
    print(paths[0])
    print([inverse_seq_dict[id] for id in paths[0]])


