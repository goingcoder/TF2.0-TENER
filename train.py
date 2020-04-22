import tensorflow as tf
from TENER import TENER
from tqdm import tqdm
import tensorflow_addons as tfa
def train_one_step(text_batch,labels_batch):
    with tf.GradientTape() as tape:
        text_batch = [text_batch[0],tf.cast(text_batch[1],tf.int32) ]
        logits, text_lens, log_likelihood = model(text_batch,labels_batch,training=True)
        loss = -tf.reduce_mean(log_likelihood)
    gradients = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, logits, text_lens

def get_acc_one_step(logits, text_lens, labels_batch):
    paths = []
    accuracy = 0
    for logit, text_len, labels in zip(logits, text_lens, labels_batch):
        viterbi_path, _ = tfa.text.viterbi_decode(logit[:text_len], model.transition_params)
        paths.append(viterbi_path)
        correct_prediction = tf.equal(
            tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([viterbi_path], padding='post'),
                                 dtype=tf.int32),
            tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([labels[:text_len]], padding='post'),
                                 dtype=tf.int32)
        )
        accuracy = accuracy + tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy = accuracy / len(paths)
    return accuracy

output_dir = './model'
model = TENER(8, 23, 128, 2, 128, 2, 256, 0.5, True, 0.3, False, 0.3)
optimizer = tf.keras.optimizers.Adam(0.01)
ckpt = tf.train.Checkpoint(optimizer=optimizer,model=model)
ckpt.restore(tf.train.latest_checkpoint(output_dir))
ckpt_manager = tf.train.CheckpointManager(ckpt,output_dir,checkpoint_name='model.ckpt',max_to_keep=3)

from reader import train_dataset
from my_log import logger
best_acc = 0
step = 0
for epoch in range(5):
    for _, (text_batch, labels_batch) in enumerate(train_dataset):
        step = step + 1
        loss, logits,text_lens = train_one_step(text_batch,labels_batch)
        if step % 20 == 0:
            accuracy = get_acc_one_step(logits,text_lens,labels_batch)
            logger.info('epoch %d, step %d, loss %.4f , accuracy %.4f' % (epoch, step, loss, accuracy))
            if accuracy > best_acc:
                best_acc = accuracy
                ckpt_manager.save()
                logger.info("model saved")
logger.info("finished")
