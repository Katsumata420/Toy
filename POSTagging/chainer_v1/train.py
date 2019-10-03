import argparse
import utilities as util
import tag_model
import chainer
import cupy
import random
import numpy as np

def train(args):
    start_epoch = 0
    corpus_file = args.datadir+'/train.'
    source = 'word'
    target = 'pos'

    util.trace('start training...')

    chainer.global_config.train = True
    chainer.global_config.use_cudnn = 'always'
    chainer.global_config.type_check = True
    util.trace('chainer config: {}'.format(chainer.global_config.__dict__))

    """
    util.trace('load vocab...')
    vocab_file = args.datadir+'/vocabulary.'
    source_vocab = util.Vocabulary.load(vocab_file+source)
    target_vocab = util.Vocabulary.load(vocab_file+target)
    """
    util.trace('make vocab...')
    source_vocab = util.Vocabulary.make(corpus_file+source, 50000)
    target_vocab = util.Vocabulary.make(corpus_file+target, 10000)

    util.trace('save vocab...')
    source_vocab.save(args.savedir+'/vocab.'+source)
    target_vocab.save(args.savedir+'/vocab.'+target)
    exit()
        
    util.trace('making model...')
    #initialize model
    model = tag_model.LSTMlabel(source_vocab, target_vocab, args)

    if args.gpunum >= 0:
        import cupy as xp
        chainer.cuda.check_cuda_available() 
        chainer.cuda.get_device(args.gpunum).use()
        model.to_gpu()
        util.trace('use GPU id: {}'.format(args.gpunum))
    else:
        import numpy as xp
        args.gpunum = -1
        util.trace('without GPU')
    
    util.trace('random seed: {}'.format(args.seed_num))
    np.random.seed(args.seed_num)
    xp.random.seed(args.seed_num)
    random.seed(args.seed_num)

    optim = args.optim
    #this is change
    optim = chainer.optimizers.AdaGrad(lr=args.lr)
    optim.setup(model)
    optim.add_hook(chainer.optimizer.GradientClipping(args.grad_clip))

    for epoch in range(start_epoch, args.epoch):
        util.trace('Epoch {}/{}'.format(epoch+1, args.epoch))
        accum_loss = 0.0
        num_sent = 0
        for batch_src, batch_tgt in util.miniBatch(corpus_file+source, corpus_file+target,\
                                    source_vocab, target_vocab, args.batch, args.pooling):
            model.zerograds()
            util.trace('forward')
            loss, batch_hyp = model(batch_src, batch_tgt)
            accum_loss += loss.data
            util.trace('backward')
            loss.backward()
            optim.update()

            for src, tgt, hyp in zip(util.convert_b2w(batch_src, source_vocab), util.convert_b2w(batch_tgt, target_vocab), \
                util.convert_b2w(batch_hyp, target_vocab)):
                util.trace('Epoch {}/{}, {} sent'.format(epoch+1, args.epoch, num_sent+1))
                util.trace('text: {}'.format(src))
                util.trace('refPOS: {}'.format(tgt))
                util.trace('hypPOS: {}'.format(hyp))
                num_sent += 1
        util.trace('accum_loss: {}'.format(accum_loss))
        util.trace('Save model ...')
        model_name = '{}.{:03d}'.format(args.name, epoch+1)
        chainer.serializers.save_npz(args.savedir+'/{}.weights'.format(model_name), model)
        chainer.serializers.save_npz(args.savedir+'/{}.optimizer'.format(model_name), optim)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='sorry, look at readme.', description='arg description', epilog='end')
    parser.add_argument('-datadir', help='data directory to use corpus and vocab', default='')
    parser.add_argument('-savedir', help='save directory for weight', default='')
    parser.add_argument('-nhid', help='hidden size for model', type=int, default=512)
    parser.add_argument('-optim', help='select optimizer', default='AdaGrad')
    parser.add_argument('-lr', help='learning rate for optimizer', type=float, default=0.01)
    parser.add_argument('-gpunum', help='GPU number (negative value is using CPU)', type=int, default=-1)
    parser.add_argument('-epoch', help='max epoch during training', type=int, default=50)
    parser.add_argument('-useDropout', help='max epoch during training', action='store_true')
    parser.add_argument('-dlr', help='dropout rate', type=float, default=0.2)
    parser.add_argument('-batch', help='batch size', type=int, default=100)
    parser.add_argument('-pooling', help='pooling size', type=int, default=100)
    parser.add_argument('-grad_clip', help='gradient cliping', type=float, default=5.0)
    parser.add_argument('-useSeed', help='use random seed or not?', action='store_true')
    parser.add_argument('-seed_num', help='random seed number', type=int, default=2434)
    parser.add_argument('-name', help='model name, default is "sample"', default='sample') 
    args = parser.parse_args()
    
    train(args)
    util.trace('finish training!')
