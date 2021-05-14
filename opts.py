import argparse
import torch


def parse_opt():
    parser = argparse.ArgumentParser()

    # train settings
    parser.add_argument('--train_mode', type=str, default='xe', choices=['xe', 'rl'])
    parser.add_argument('--learning_rate', type=float, default=4e-5)  # 4e-4 for xe, 4e-5 for rl
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--max_epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=20)

    parser.add_argument('--idx2word', type=str, default='prepare/idx2word.json')
    parser.add_argument('--captions', type=str, default='prepare/captions.json')
    parser.add_argument('--att_feats', type=str, default='prepare/features/features_att.h5')
    parser.add_argument('--word_feats', type=str, default='prepare/features/Image_words.h5')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint')
    parser.add_argument('--result', type=str, default='./result/')
    parser.add_argument('--grad_clip', type=float, default=0.1)
    parser.add_argument('--label_smoothing', type=float, default=0.1)  # 0 means the cross entropy loss
    parser.add_argument('--beam_size', type=int, default=3)

    # encoder settings
    parser.add_argument('--resnet101_file', type=str, default='prepare/resnet101.pth', help='Pre-trained resnet101 network for extracting image features')

    # eval settings
    parser.add_argument('-e', '--eval_model', type=str, default='')
    parser.add_argument('-r', '--result_file', type=str, default='')

    # test setting
    parser.add_argument('-t', '--test_model', type=str, default='')
    parser.add_argument('-i', '--image_file', type=str, default='')

    args = parser.parse_args()

    # decoder settings
    settings = dict()
    settings['att_feat_dim'] = 2048
    settings['word_feat_dim'] = 1000
    settings['d_model'] = 512  # model dim
    settings['d_ff'] = 2048  # feed forward dim
    settings['h'] = 8  # multi heads num
    settings['N_enc'] = 4  # encoder layers num # 6
    settings['N_dec'] = 4  # decoder layers num # 6
    settings['dropout_p'] = 0.1
    settings['max_seq_len'] = 16

    args.settings = settings
    args.use_gpu = torch.cuda.is_available()
    args.device = torch.device('cuda:0') if args.use_gpu else torch.device('cpu')
    return args
