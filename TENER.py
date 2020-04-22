import math
import tensorflow as tf
import copy
import tensorflow_addons as tfa


class RelativeSinusoidalPositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, padding_idx, init_size=1568):
        super(RelativeSinusoidalPositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        assert init_size % 2 == 0
        weight = self.get_embedding(init_size + 1, embedding_dim, padding_idx)
        self.__setattr__('weight', tf.stop_gradient(weight))
        self.padding_idx = padding_idx

    def get_embedding(self, num_embeddings, embedding_dim, padding_idx=None):
        """ Build sinusodal embeddings
        :param num_embeddings:
        :param embedding_dim:
        :param padding_idx:
        :return:
        """
        half_dim = embedding_dim // 2
        emb = tf.math.log(10000.0) / (half_dim - 1)
        emb = tf.math.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
        emb = tf.matmul(tf.expand_dims(tf.range(-num_embeddings // 2, num_embeddings // 2, dtype=tf.float32), 1),
                        tf.expand_dims(emb, 0))
        emb = tf.reshape(tf.concat([tf.math.sin(emb), tf.math.cos(emb)], axis=1), shape=(num_embeddings, -1))
        if embedding_dim % 2 == 1:
            emb = tf.concat([emb, tf.zeros(shape=(num_embeddings, 1))], axis=1)
        if padding_idx is not None:
            up = emb[:padding_idx, :]
            updates = tf.zeros(shape=(1, emb.shape[1]))
            down = emb[padding_idx + 1:, :]
            emb = tf.concat([up, updates, down], axis=0)
        self.origin_shift = num_embeddings // 2 + 1
        return emb

    def call(self, inputs):
        shape = inputs.shape
        bsz, seq_len = shape[0], shape[1]
        max_pos = self.padding_idx + seq_len
        if max_pos > self.origin_shift:
            weight = self.get_embedding(
                max_pos * 2,
                self.embedding_dim,
                self.padding_idx
            )
            weight = tf.cast(weight, tf.float32)
            self.__delattr__('weight')
            self.origin_shift = weight.shape[0] // 2
            self.__setattr__('weight', tf.stop_gradient(weight))
        positions = tf.range(-seq_len, seq_len, dtype=tf.int64) + self.origin_shift
        embed = tf.stop_gradient(tf.gather(self.weight, positions, axis=0))
        return embed


class RelativeMultiHeadAttn(tf.keras.layers.Layer):
    def __init__(self, d_model, n_head, dropout, r_w_bias=None, r_r_bias=None, scale=False):
        super().__init__()
        self.qv_linear = tf.keras.layers.Dense(d_model * 2, use_bias=False)
        self.n_head = n_head
        assert d_model % n_head == 0
        self.head_dim = d_model // n_head
        self.dropout_layer = tf.keras.layers.Dropout(dropout)

        self.pos_embed = RelativeSinusoidalPositionalEmbedding(d_model // n_head, 0, 1200)
        if scale:
            self.scale = tf.cast(tf.math.sqrt(d_model // n_head), dtype=tf.float32)
        else:
            self.scale = tf.constant(1, dtype=tf.float32)
        if r_r_bias is None or r_w_bias is None:
            r_r_initializer = tf.keras.initializers.GlorotNormal()
            r_w_initializer = tf.keras.initializers.GlorotNormal()
            self.r_r_bias = tf.Variable(r_r_initializer(shape=(n_head, d_model // n_head)))
            self.r_w_bias = tf.Variable(r_w_initializer(shape=(n_head, d_model // n_head)))
        else:
            self.r_r_bias = r_r_bias  ## r_r_bias is v
            self.r_w_bias = r_w_bias  ## r_w_bias is u

    def call(self, inputs, mask):
        """

        :param inputs: batch_size * max_len * d_model
        :param mask: batch_size * max_len
        :return:
        """
        return self.__call__(inputs, mask)

    def __call__(self, x, mask):
        """
        :param x: batch_size * max_len * d_model
        :param mask: batch_size * max_len
        :return:
        """
        shape = x.shape
        batch_size, max_len, d_model = shape[0], shape[1], shape[2]
        pos_embed = self.pos_embed(mask)  ## l * head_dim

        qv = self.qv_linear(x)  ##batch_size , max_len , d_model * 2
        q, v = tf.split(qv, num_or_size_splits=2, axis=-1)
        q = tf.transpose(tf.reshape(q, shape=(batch_size, max_len, self.n_head, -1)), [0, 2, 1, 3])
        k = tf.transpose(tf.reshape(x, shape=(batch_size, max_len, self.n_head, -1)), [0, 2, 1, 3])
        v = tf.transpose(tf.reshape(v, shape=(batch_size, max_len, self.n_head, -1)), [0, 2, 1, 3])
        rw_head_q = q + self.r_r_bias[:, None]

        AC = tf.einsum('bnqd,bnkd->bnqk', rw_head_q, k)  ## b * n * l * d

        D_ = tf.einsum('nd,ld->nl', self.r_w_bias, pos_embed)[None, :, None]  ## 1 * n * 1 * l
        B_ = tf.einsum('bnqd,ld->bnql', q, pos_embed)

        BD = B_ + D_
        BD = self._shift(BD)
        attn = AC + BD
        ## b * n * l * d
        attn = attn / self.scale

        def mask_fill_inf(matrix, mask):
            mask = tf.cast(mask, tf.float32)
            matrix = tf.cast(matrix, tf.float32)
            negmask = 1 - mask

            num = tf.cast(tf.math.pow(10, 8), tf.float32)

            return tf.multiply(matrix, mask) + (-((negmask * num + num) - num))

        attn = mask_fill_inf(attn, mask[:, None, None, :])

        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.dropout_layer(attn)  ## b x n x l x l
        v = tf.reshape(tf.transpose(tf.matmul(attn, v), [0, 2, 1, 3]), (batch_size, max_len, d_model))
        return v

    def _shift(self, BD):
        """

        :param BD: batch_size * n_head * max_len * 2max_len
        :return: batch_size * n_head * max_len * max_len
        """
        shape = BD.shape
        bsz, n_head, max_len, _ = shape[0], shape[1], shape[2], shape[3]
        zero_pad = tf.zeros((bsz, n_head, max_len, 1), dtype=tf.float32)
        BD = tf.reshape(tf.concat([BD, zero_pad], axis=-1), shape=(bsz, n_head, -1, max_len))
        BD = tf.reshape(BD[:, :, :-1], shape=(bsz, n_head, max_len, -1))
        BD = BD[:, :, :, max_len:]
        return BD


def make_positions(tensor, padding_idx):
    mask = tf.cast(tf.not_equal(tensor, padding_idx), tf.int32)
    return tf.cumsum(mask, axis=1) * mask + padding_idx


class SinusoidalPositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, padding_idx, init_size=1568):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx
        )

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """

        :param num_embedding:
        :param embedding_dim:
        :param padding_idx:
        :return:
        """
        half_dim = embedding_dim // 2
        emb = tf.math.log(10000.0) / (half_dim - 1)
        emb = tf.math.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
        emb = tf.matmul(tf.expand_dims(tf.range(num_embeddings, dtype=tf.float32), 1),
                        tf.expand_dims(emb, 0))
        emb = tf.reshape(tf.concat([tf.math.sin(emb), tf.math.cos(emb)], axis=1), shape=(num_embeddings, -1))
        if embedding_dim % 2 == 1:
            emb = tf.concat([emb, tf.zeros(num_embeddings, 1)], axis=1)
        if padding_idx is not None:
            up = emb[:padding_idx, :]
            updates = tf.zeros(shape=(1, emb.shape[1]), dtype=tf.float32)
            down = emb[padding_idx + 1:, :]
            emb = tf.concat([up, updates, down], axis=0)

        return emb

    def call(self, inputs):
        """Input is expected to be size of [bsz * seqlen]"""
        shape = inputs.shape
        bsz, seq_len = shape[0], shape[1]
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos > self.weight.shape[0]:
            # recompute/expand embeddings if needed
            self.weight = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weight = tf.cast(self.weight, dtype=tf.float32)
        positions = make_positions(inputs, self.padding_idx)
        return tf.stop_gradient(tf.reshape(tf.nn.embedding_lookup(self.weight, positions), (bsz, seq_len, -1)))


class SequentialLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, feedforward_dim, dropout):
        super(SequentialLayer, self).__init__()
        self.dense1 = tf.keras.layers.Dense(feedforward_dim)
        self.relu = tf.keras.layers.ReLU()
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dense2 = tf.keras.layers.Dense(d_model)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs):
        output = self.dense1(inputs)
        output = self.relu(output)
        output = self.dropout1(output)
        output = self.dense2(output)
        output = self.dropout2(output)
        return output


class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, self_atten, feedfoward_dim, after_norm, dropout):
        super(TransformerLayer, self).__init__()
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.self_atten = self_atten
        self.after_norm = after_norm
        self.ffn = SequentialLayer(d_model, feedfoward_dim, dropout)

    def call(self, inputs, mask):
        return self.__call__(inputs, mask)

    def __call__(self, x, mask):
        residual = x
        if not self.after_norm:
            x = self.norm1(x)
        x = self.self_atten(x, mask)
        x = x + residual

        if self.after_norm:
            x = self.norm1(x)
        residual = x
        if not self.after_norm:
            x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        if self.after_norm:
            x = self.norm2(x)
        return x


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, n_head, feedfoward_dim, dropout, after_norm=True,
                 scale=False, dropout_attn=None, **kwargs):
        super().__init__(**kwargs)
        if dropout_attn is None:
            dropout_attn = dropout
        self.d_model = d_model
        self.pos_embed = SinusoidalPositionalEmbedding(d_model, 0, init_size=1024)
        self_atten = RelativeMultiHeadAttn(d_model, n_head, dropout_attn, scale=scale)
        self.layers = [TransformerLayer(d_model, copy.deepcopy(self_atten), feedfoward_dim, after_norm, dropout) for _
                       in range(num_layers)]

    def call(self, inputs, mask):
        x = inputs

        x = x + self.pos_embed(mask)

        for layer in self.layers:
            x = layer(x, mask)
        return x


class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_num, embedding_dim, output_size, activation):
        super().__init__()
        self.vocab_num = vocab_num + 1
        self.embedding_dim = embedding_dim
        self.output_size = output_size
        self.vocab_embedding = tf.Variable(tf.random.normal(shape=(self.vocab_num, self.embedding_dim)))
        self.activation = activation
        if self.activation is None:
            self.activation = 'identity'
        assert self.activation.lower() in ['sigmoid', 'tanh', 'relu', 'identity']
        self.GRU = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(self.embedding_dim, activation=self.activation, return_sequences=True,
                                return_state=False))
        self.output_dense = tf.keras.layers.Dense(output_size)

    def call(self, inputs, **kwargs):
        embedded_input = tf.nn.embedding_lookup(self.vocab_embedding, inputs)
        embedded_input = self.GRU(embedded_input)
        embedded_output = self.output_dense(embedded_input)
        return embedded_output


class TENER(tf.keras.Model):
    def __init__(self, tag_vocab, src_vocab, embed_size, num_layers, d_model, n_head, feedforward_dim, dropout,
                 after_norm=True,
                 fc_dropout=0.3, scale=False,
                 dropout_attn=None):
        super().__init__()
        self.embed_size = embed_size
        self.embed = EmbeddingLayer(src_vocab, self.embed_size, self.embed_size, activation='tanh')

        self.in_fc = tf.keras.layers.Dense(d_model)
        self.transformer = TransformerEncoder(num_layers, d_model, n_head, feedforward_dim, dropout,
                                              after_norm=after_norm,
                                              scale=scale, dropout_attn=dropout_attn)
        self.fc_dropout = tf.keras.layers.Dropout(fc_dropout)
        self.out_fc = tf.keras.layers.Dense(tag_vocab + 1)
        self.tag_vocab = tag_vocab + 1

    def call(self, inputs, labels=None):
        seqs = inputs[0]
        mask = inputs[1]

        text_lens = tf.math.reduce_sum(tf.cast(tf.math.not_equal(mask, 0), dtype=tf.int32), axis=-1)
        seqs_x = self.embed(seqs)
        chars = self.in_fc(seqs_x)

        chars = self.transformer(chars, mask)
        chars = self.fc_dropout(chars)
        chars = self.out_fc(chars)
        logits = tf.nn.log_softmax(chars, axis=-1)


        if labels is not None:
            label_sequences = tf.convert_to_tensor(labels, tf.int32)
            log_likelihood, self.transition_params = tfa.text.crf_log_likelihood(logits, label_sequences, text_lens)
            self.transition_params = tf.Variable(self.transition_params, trainable=False)
            return logits, text_lens, log_likelihood
        else:
            return logits, text_lens


