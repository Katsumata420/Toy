import cupy
import chainer
import chainer.functions as chainFunc
import chainer.links as chainLinks
import utilities as util
import cupy as xp

class LSTMlabel(chainer.Chain):
    def __init__(self, src_vocab, tgt_vocab, args):
        super().__init__()
        with self.init_scope():
            self.labeler = GRU_labeler(src_vocab, tgt_vocab, args)

    def __call__(self, batch_src, batch_tgt):
        self.reset_state()
        loss, outputs = self.labeler(batch_src, batch_tgt)
        return loss, outputs 

    def generate(self, batch_src):
        self.reset_state()
        hyp = self.labeler.genHyp(batch_src)
        return hyp

    def reset_state(self):
        self.labeler.reset_states()

class GRU_labeler(chainer.Chain):
    def __init__(self, src_vocab, tgt_vocab, args):
        super().__init__()
        with self.init_scope():
            self.word2embed = chainLinks.EmbedID(src_vocab.size, args.nhid, ignore_label=-1)
            # W_1, U
            self.embed2hidden = chainLinks.GRU(args.nhid, args.nhid)
            # W_2
            self.W_2 = chainLinks.Linear(args.nhid, tgt_vocab.size)

        self.vocab_size = src_vocab.size
        self.hidden_size = args.nhid
        if args.useDropout:
            self.use_dropout = args.useDropout
            self.dropoutr = args.dlr
        else:
            self.use_dropout = None
            self.dropoutr = 0
            util.trace('{}'.format(chainer.global_config.__dict__))

    def __call__(self, batch_src, batch_tgt):
        loss = chainer.Variable(self.xp.zeros((), dtype = self.xp.float32))
        first_states = list()
        outputs = list()
        """
        util.trace('batch_src. length: {}'.format(len(batch_src)))
        util.trace('batch_src, shape:{}'.format(batch_src[0].shape))
        util.trace('batch_tgt. length: {}'.format(len(batch_tgt)))
        util.trace('batch_tgt, shape:{}'.format(batch_tgt[0].shape))
        """
        #filstlayer   
        for word in batch_src:
            first_states.append(chainFunc.dropout(self.word2embed(word), ratio=self.dropoutr))
        """
        util.trace('first_states length:{}'.format(len(first_states)))
        util.trace('first state embedding:{}'.format(first_states[0].shape))
        """

        for first_hidden, pos in zip(first_states, batch_tgt):
            forward_hidden = chainFunc.dropout(self.embed2hidden(first_hidden), self.dropoutr)
            score = self.W_2(forward_hidden)
            prd = chainFunc.argmax(score, axis=1)
            loss += chainFunc.softmax_cross_entropy(score, pos, ignore_label=-1)
            outputs.append(prd)

        """
        util.trace('top state embedding:{}'.format(len(prd)))
        util.trace('score:{}'.format(score.shape))
        exit()
        """
        return loss, outputs

    def genHyp(self, batch_src):
        first_states = list()
        outputs = list()

        for word in batch_src:
            first_states.append(self.word2embed(word))

        for first_hidden in first_states:
            forward_hidden = self.embed2hidden(first_hidden)
            prd = chainFunc.argmax(self.W_2(forward_hidden), axis=1)
            outputs.append(prd)
        
        return outputs
    
    def reset_states(self):
        self.embed2hidden.reset_state()
    
