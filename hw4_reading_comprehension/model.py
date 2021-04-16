import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import GRUCell, LSTMCell


def cbow_forward(config, inputs, scope=None):
    with tf.variable_scope(scope or "forward"):

        JX, JQ = config.max_context_size, config.max_ques_size
        d = config.hidden_size
        x, x_len, q, q_len = [inputs[key] for key in ['x', 'x_len', 'q', 'q_len']]
        x_mask = tf.sequence_mask(x_len, JX)
        q_mask = tf.sequence_mask(q_len, JQ)

        # emb_mat = tf.get_variable('emb_mat', shape=[V, d])
        emb_mat = config.emb_mat_ph if config.serve else config.emb_mat
        emb_mat = tf.slice(emb_mat, [2, 0], [-1, -1])
        emb_mat = tf.concat([tf.get_variable('emb_mat', shape=[2, d]), emb_mat], axis=0)
        xx = tf.nn.embedding_lookup(emb_mat, x, name='xx')  # [N, JX, d]
        qq = tf.nn.embedding_lookup(emb_mat, q, name='qq')  # [N, JQ, d]

        qq_avg = tf.reduce_mean(bool_mask(qq, q_mask, expand=True), axis=1)  # [N, d]
        qq_avg_exp = tf.expand_dims(qq_avg, axis=1)  # [N, 1, d]
        qq_avg_tiled = tf.tile(qq_avg_exp, [1, JX, 1])  # [N, JX, d]

        xq = tf.concat([xx, qq_avg_tiled, xx * qq_avg_tiled], axis=2)  # [N, JX, 3d]
        xq_flat = tf.reshape(xq, [-1, 3*d])  # [N * JX, 3*d]

        # Compute logits
        with tf.variable_scope('start'):
            logits1 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp1 = tf.argmax(logits1, axis=1)  # [N]
        with tf.variable_scope('stop'):
            logits2 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp2 = tf.argmax(logits2, axis=1)  # [N]

        outputs = {'logits1': logits1, 'logits2': logits2, 'yp1': yp1, 'yp2': yp2}
        variables = {'emb_mat': emb_mat}
        return variables, outputs



def rnn_forward(config, inputs, scope=None):
    with tf.variable_scope(scope or "forward"):

        JX, JQ = config.max_context_size, config.max_ques_size
        d = config.hidden_size
        x, x_len, q, q_len = [inputs[key] for key in ['x', 'x_len', 'q', 'q_len']]
        x_mask = tf.sequence_mask(x_len, JX)
        q_mask = tf.sequence_mask(q_len, JQ)

        # emb_mat = tf.get_variable('emb_mat', shape=[V, d])
        emb_mat = config.emb_mat_ph if config.serve else config.emb_mat
        emb_mat = tf.slice(emb_mat, [2, 0], [-1, -1])
        emb_mat = tf.concat([tf.get_variable('emb_mat', shape=[2, d]), emb_mat], axis=0)
        xx = tf.nn.embedding_lookup(emb_mat, x, name='xx')  # [N, JX, d]
        qq = tf.nn.embedding_lookup(emb_mat, q, name='qq')  # [N, JQ, d]

        if config.train == True:
            keep_proba = config.keep_prob
        else:
            keep_proba = 1

        fwd_cell1  = DropoutWrapper(GRUCell(config.hidden_size), input_keep_prob=keep_proba, output_keep_prob=keep_proba)
        bwd_cell1 = DropoutWrapper(GRUCell(config.hidden_size), input_keep_prob=keep_proba, output_keep_prob=keep_proba)
        fwd_cell2  = DropoutWrapper(GRUCell(config.hidden_size), input_keep_prob=keep_proba, output_keep_prob=keep_proba)
        bwd_cell2 = DropoutWrapper(GRUCell(config.hidden_size), input_keep_prob=keep_proba, output_keep_prob=keep_proba)


        qq_input = bool_mask(qq, q_mask, expand=True)
        qq_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwd_cell1, cell_bw=bwd_cell1, inputs=qq_input, sequence_length=q_len,
                                                        dtype=tf.float32, scope='rnn_ques')
        final_qq = tf.concat(list(qq_output), axis=2)  # [N, JQ, 2d]

        xx_input = bool_mask(xx, x_mask, expand=True)
        xx_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwd_cell2, cell_bw=bwd_cell2, inputs=xx_input, sequence_length=x_len,
                                                       dtype=tf.float32, scope='rnn_para')
        final_xx = tf.concat(list(xx_output), axis=2)  # [N, JX, 2d]


        qq_avg = tf.reduce_mean(final_qq, axis=1)  # [N, 2d]
        qq_avg_exp = tf.expand_dims(qq_avg, axis=1)  # [N, 1, 2d]
        qq_avg_tiled = tf.tile(qq_avg_exp, [1, JX, 1])  # [N, JX, 2d]

        xq = tf.concat([final_xx, qq_avg_tiled, final_xx * qq_avg_tiled], axis=2)  # [N, JX, 2*3d]
        xq_flat = tf.reshape(xq, [-1, 2*3*d])  # [N * JX, 2*3*d]

        # Compute logits
        with tf.variable_scope('start'):
            logits1 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp1 = tf.argmax(logits1, axis=1)  # [N]
        with tf.variable_scope('stop'):
            logits2 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp2 = tf.argmax(logits2, axis=1)  # [N]

        outputs = {'logits1': logits1, 'logits2': logits2, 'yp1': yp1, 'yp2': yp2}
        variables = {'emb_mat': emb_mat}
        return variables, outputs
        #raise NotImplementedError()



def attention_forward(config, inputs, scope=None):
    with tf.variable_scope(scope or "forward"):

        JX, JQ = config.max_context_size, config.max_ques_size
        d = config.hidden_size
        x, x_len, q, q_len = [inputs[key] for key in ['x', 'x_len', 'q', 'q_len']]
        x_mask = tf.sequence_mask(x_len, JX)
        q_mask = tf.sequence_mask(q_len, JQ)

        # emb_mat = tf.get_variable('emb_mat', shape=[V, d])
        emb_mat = config.emb_mat_ph if config.serve else config.emb_mat
        emb_mat = tf.slice(emb_mat, [2, 0], [-1, -1])
        emb_mat = tf.concat([tf.get_variable('emb_mat', shape=[2, d]), emb_mat], axis=0)
        xx = tf.nn.embedding_lookup(emb_mat, x, name='xx')  # [N, JX, d]
        qq = tf.nn.embedding_lookup(emb_mat, q, name='qq')  # [N, JQ, d]

        if config.train == True:
            keep_proba = config.keep_prob
        else:
            keep_proba = 1

        fwd_cell1  = DropoutWrapper(GRUCell(config.hidden_size), input_keep_prob=keep_proba, output_keep_prob=keep_proba)
        bwd_cell1 = DropoutWrapper(GRUCell(config.hidden_size), input_keep_prob=keep_proba, output_keep_prob=keep_proba)
        fwd_cell2  = DropoutWrapper(GRUCell(config.hidden_size), input_keep_prob=keep_proba, output_keep_prob=keep_proba)
        bwd_cell2 = DropoutWrapper(GRUCell(config.hidden_size), input_keep_prob=keep_proba, output_keep_prob=keep_proba)


        qq_input = bool_mask(qq, q_mask, expand=True)
        qq_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwd_cell1, cell_bw=bwd_cell1, inputs=qq_input, sequence_length=q_len,
                                                       dtype=tf.float32, scope='rnn_ques')
        final_qq = tf.concat(list(qq_output), axis=2)  # [N, JQ, 2d]

        xx_input = bool_mask(xx, x_mask, expand=True)
        xx_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwd_cell2, cell_bw=bwd_cell2, inputs=xx_input, sequence_length=x_len,
                                                       dtype=tf.float32, scope='rnn_para')
        final_xx = tf.concat(list(xx_output), axis=2)  # [N, JX, 2d]


        xx_qq = tf.einsum('aik,ajk->aijk', xx, qq)  # [N, JX, JQ, d])
        qq_tiled = tf.tile(tf.expand_dims(qq, axis=1), [1, JX, 1, 1])  # [N, JX, JQ, d]
        xx_tiled = tf.tile(tf.expand_dims(xx, axis=2), [1, 1, JQ, 1])  # [N, JX, JQ, d]
        final_att = tf.concat([xx_tiled, qq_tiled, xx_qq], axis=3)  # [N, JX, JQ, 3d]

        w_att = tf.Variable(tf.random_normal(shape=[1, 3*d], stddev=0.1), name='w_att')  # [1, 3d]
        b_att = tf.Variable(tf.random_normal(shape=[1], stddev=0.1), name='b_att')  # [1]
        p_att = tf.einsum('ijka,ma->ijk', final_att, w_att) + b_att  # [N, JX, JQ]
        att_weights = tf.nn.softmax(p_att)  # [N, JX, JQ]

        qq_avg = tf.einsum('aij,ajk->aik', att_weights, final_qq)  # [N, JX, 2d]

        xq = tf.concat([final_xx, qq_avg, final_xx * qq_avg], axis=2)  # [N, JX, 2*3d]
        xq_flat = tf.reshape(xq, [-1, 2*3*d])  # [N * JX, 2*3*d]

        # Compute logits
        with tf.variable_scope('start'):
            logits1 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp1 = tf.argmax(logits1, axis=1)  # [N]
        with tf.variable_scope('stop'):
            logits2 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp2 = tf.argmax(logits2, axis=1)  # [N]

        outputs = {'logits1': logits1, 'logits2': logits2, 'yp1': yp1, 'yp2': yp2}
        variables = {'emb_mat': emb_mat}
        return variables, outputs
        # raise NotImplementedError()



def attention_LSTM_forward(config, inputs, scope=None):
    with tf.variable_scope(scope or "forward"):

        JX, JQ = config.max_context_size, config.max_ques_size
        d = config.hidden_size
        x, x_len, q, q_len = [inputs[key] for key in ['x', 'x_len', 'q', 'q_len']]
        x_mask = tf.sequence_mask(x_len, JX)
        q_mask = tf.sequence_mask(q_len, JQ)

        # emb_mat = tf.get_variable('emb_mat', shape=[V, d])
        emb_mat = config.emb_mat_ph if config.serve else config.emb_mat
        emb_mat = tf.slice(emb_mat, [2, 0], [-1, -1])
        emb_mat = tf.concat([tf.get_variable('emb_mat', shape=[2, d]), emb_mat], axis=0)
        xx = tf.nn.embedding_lookup(emb_mat, x, name='xx')  # [N, JX, d]
        qq = tf.nn.embedding_lookup(emb_mat, q, name='qq')  # [N, JQ, d]

        if config.train == True:
            keep_proba = config.keep_prob
        else:
            keep_proba = 1

        fwd_cell1  = DropoutWrapper(LSTMCell(config.hidden_size), input_keep_prob=keep_proba, output_keep_prob=keep_proba)
        bwd_cell1 = DropoutWrapper(LSTMCell(config.hidden_size), input_keep_prob=keep_proba, output_keep_prob=keep_proba)
        fwd_cell2  = DropoutWrapper(LSTMCell(config.hidden_size), input_keep_prob=keep_proba, output_keep_prob=keep_proba)
        bwd_cell2 = DropoutWrapper(LSTMCell(config.hidden_size), input_keep_prob=keep_proba, output_keep_prob=keep_proba)


        qq_input = bool_mask(qq, q_mask, expand=True)
        qq_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwd_cell1, cell_bw=bwd_cell1, inputs=qq_input, sequence_length=q_len,
                                                       dtype=tf.float32, scope='rnn_ques')
        final_qq = tf.concat(list(qq_output), axis=2)  # [N, JQ, 2d]

        xx_input = bool_mask(xx, x_mask, expand=True)
        xx_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwd_cell2, cell_bw=bwd_cell2, inputs=xx_input, sequence_length=x_len,
                                                       dtype=tf.float32, scope='rnn_para')
        final_xx = tf.concat(list(xx_output), axis=2)  # [N, JX, 2d]


        xx_qq = tf.einsum('aik,ajk->aijk', xx, qq)  # [N, JX, JQ, d])
        qq_tiled = tf.tile(tf.expand_dims(qq, axis=1), [1, JX, 1, 1])  # [N, JX, JQ, d]
        xx_tiled = tf.tile(tf.expand_dims(xx, axis=2), [1, 1, JQ, 1])  # [N, JX, JQ, d]
        final_att = tf.concat([xx_tiled, qq_tiled, xx_qq], axis=3)  # [N, JX, JQ, 3d]

        w_att = tf.Variable(tf.random_normal(shape=[1, 3*d], stddev=0.1), name='w_att')  # [1, 3d]
        b_att = tf.Variable(tf.random_normal(shape=[1], stddev=0.1), name='b_att')  # [1]
        p_att = tf.einsum('ijka,ma->ijk', final_att, w_att) + b_att  # [N, JX, JQ]
        att_weights = tf.nn.softmax(p_att)  # [N, JX, JQ]

        qq_avg = tf.einsum('aij,ajk->aik', att_weights, final_qq)  # [N, JX, 2d]

        xq = tf.concat([final_xx, qq_avg, final_xx * qq_avg], axis=2)  # [N, JX, 2*3d]
        xq_flat = tf.reshape(xq, [-1, 2*3*d])  # [N * JX, 2*3*d]

        # Compute logits
        with tf.variable_scope('start'):
            logits1 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp1 = tf.argmax(logits1, axis=1)  # [N]
        with tf.variable_scope('stop'):
            logits2 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp2 = tf.argmax(logits2, axis=1)  # [N]

        outputs = {'logits1': logits1, 'logits2': logits2, 'yp1': yp1, 'yp2': yp2}
        variables = {'emb_mat': emb_mat}
        return variables, outputs
        # raise NotImplementedError()



def get_loss(config, inputs, outputs, scope=None):
    with tf.name_scope(scope or "loss"):
        y1, y2 = inputs['y1'], inputs['y2']
        logits1, logits2 = outputs['logits1'], outputs['logits2']
        loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y1, logits=logits1))
        loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y2, logits=logits2))
        loss = loss1 + loss2
        acc1 = tf.reduce_mean(tf.cast(tf.equal(y1, tf.cast(tf.argmax(logits1, 1), 'int32')), 'float'))
        acc2 = tf.reduce_mean(tf.cast(tf.equal(y2, tf.cast(tf.argmax(logits2, 1), 'int32')), 'float'))
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('acc1', acc1)
        tf.summary.scalar('acc2', acc2)
        return loss


def exp_mask(val, mask, expand=False):
    if expand:
        mask = tf.expand_dims(mask, -1)
    return val - (1.0 - tf.cast(mask, 'float')) * 10.0e10


def bool_mask(val, mask, expand=False):
    if expand:
        mask = tf.expand_dims(mask, -1)
    return val * tf.cast(mask, 'float')
