import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

#TODO: 推論部の記述
class LstmLm(chainer.Chain):
    def __init__(self, vocab, network_prop):
        super().__init__()
        with self.init_scope():
            self.model = ForwardLm(vocab, network_prop)
    
    def __call__(self, batch_input_ref):
        # この段階で既に Variable で cast されている状態、そこまでを multiprocess で事前処理
        # batch_input: (batch_size, sent_length, dim); batch_ref: (batch_size, sent_length)
        batch_input, batch_ref = chainer.dataset.concat_examples(batch_input_ref)
        outputs = list()
        # これいる？
        loss = chainer.Variable(self.np.zeros((), dtype=np.float32))

        # last_scores: (batch_size, sent_length, vocab_size(class))
        last_scores = self.model(batch_input)

        # last_scores, batch_ref: (sent_length, batch_size, dim), 
        # 言語モデルのため、出力文長は一定、padding は必要ない
        last_scores = F.transpose_sequence(last_scores)
        batch_ref = F.transpose_sequence(batch_ref)

        for prd_score, ref in zip(last_scores, batch_ref):
            prd_token = F.argmax(prd_score, axis=1)
            loss += F.softmax_cross_entropy(prd_score, ref)
            outputs.append(prd_token)

        return loss, outputs
    
    def clear_grads(self):
        self.model.clear_grads()
    
    def param_num(self):
        return self.model.param_num()


class ForwardLm(chainer.Chain):
    def __init__(self, vocab, network_prop):
        super().__init__()
        with self.init_scope():
            self.word2embed = L.EmbedID(vocab.size, network_prop.nemb, ignore_label=-1)
            self.forward_lstm = L.NStepLSTM(n_layers=network_prop.layer, in_size=network_prop.nemb, \
                           out_size=network_prop.nhid, dropout=network_prop.dlr)
            self.hidden2word = L.Linear(network_prop.nhid, vocab.size)
        
        self.embed_dropout = network_prop.embed_dropout
        self.dlr = network_prop.dlr
        
    def __call__(self, batch):
        # arg の batch は src 側の 系列がいっぱい入ってる状況
        hx, cx = None, None

        if self.embed_dropout:
            batch_lstm_input = [F.dropout(word2embed(x), ratio=self.dlr) for x in batch]
        else:
            batch_lstm_input = [word2embed(x) for x in batch]
    
        hy, cy, ys = forward_lstm(hx=hx, cs=cx, xs=batch_lstm_input)

        last_scores = [self.hidden2word(x) for x in ys]

        return last_scores
        
    def clear_grads(self):
        self.forward_lstm.cleargrads()
        self.word2embed.cleargrads()
        self.hidden2word.cleargrads()
        
    def param_num(self):
        all_param = 0
        all_param += self.forward_lstm.count_params()
        all_param += self.word2embed.count_params()
        all_param += self.hidden2word.count_params()
        return all_param
        
