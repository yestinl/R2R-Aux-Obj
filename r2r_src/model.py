
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from param import args
import random

class EncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx,
                            dropout_ratio, bidirectional=False, num_layers=1, glove=None):
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        if bidirectional:
            print("Using Bidir in EncoderLSTM")
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.use_glove = glove is not None
        if self.use_glove:
            print('Using GloVe embedding')
            self.embedding.weight.data[...] = torch.from_numpy(glove)
            self.embedding.weight.requires_grad = False
        input_size = embedding_size
        self.lstm = nn.LSTM(input_size, hidden_size, self.num_layers,
                            batch_first=True, dropout=dropout_ratio,
                            bidirectional=bidirectional)
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
                                         hidden_size * self.num_directions)

    def init_state(self, inputs):
        ''' Initialize to zero cell states and hidden states.'''
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)

        return h0.cuda(), c0.cuda()

    def forward(self, inputs, lengths):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        embeds = self.embedding(inputs)  # (batch, seq_len, embedding_size)
        if not self.use_glove:
            embeds = self.drop(embeds)
        h0, c0 = self.init_state(inputs)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

        if self.num_directions == 2:    # The size of enc_h_t is (num_layers * num_directions, batch, hidden_size)
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1] # (batch, hidden_size)

        ctx, _ = pad_packed_sequence(enc_h, batch_first=True)

        if args.sub_out == "max":
            ctx_max, _ = ctx.max(1)
            decoder_init = nn.Tanh()(self.encoder2decoder(ctx_max))
        elif args.sub_out == "tanh":
            decoder_init = nn.Tanh()(self.encoder2decoder(h_t))
        else:
            assert False

        ctx = self.drop(ctx)
        if args.zero_init:
            return ctx, torch.zeros_like(decoder_init), torch.zeros_like(c_t)
        else:
            return ctx, decoder_init, c_t  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)


class SoftDotAttention(nn.Module):
    '''Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, query_dim, ctx_dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(query_dim + ctx_dim, query_dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        logit = attn

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask.bool(), -float('inf'))
        attn = self.sm(attn)    # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, h), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, attn
        else:
            return weighted_context, attn


class AttnDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, embedding_size, hidden_size,
                       dropout_ratio):
        super(AttnDecoderLSTM, self).__init__()

        if args.sparseObj and (not args.denseObj):
            print("Train in sparseObj mode")
            feature_size = args.glove_emb+args.angle_bbox_size  # 308
            if args.catRN:
                print("Train in sparseObj+RN mode")
                feature_size = args.glove_emb+args.angle_bbox_size+args.feature_size+args.angle_feat_size # 2484
        elif args.denseObj and (not args.sparseObj):
            print("Train in denseObj mode")
            feature_size = args.feature_size+args.angle_feat_size   # 2176
            if args.catRN:
                print("Train in denseObj+RN mode")
                feature_size = args.feature_size*2+args.angle_feat_size*2 # 4352
                # self.att_fc = nn.Linear(feature_size, args.feature_size) # run denseObj_RN_FC_0
        elif args.denseObj and args.sparseObj:
            print("Train in sparseObj + denseObj mode")
            feature_size = args.feature_size+args.angle_feat_size+args.glove_emb+args.angle_bbox_size # 2484
            if args.catRN:
                print("Train in sparseObj+denseObj+RN mode")
                feature_size = args.feature_size*2+args.angle_feat_size*2+args.glove_emb+args.angle_bbox_size # 4660
        else:
            print("Train in RN mode")
            feature_size = args.feature_size+args.angle_feat_size # 2176
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Sequential(
            nn.Linear(args.angle_feat_size, self.embedding_size),
            nn.Tanh()
        )
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop_env = nn.Dropout(p=args.featdropout)
        self.lstm = nn.LSTMCell(embedding_size+feature_size, hidden_size)
        # self.lstm = nn.LSTMCell(args.feature_size+embedding_size, hidden_size) ## run denseObj_RN_FC_0
        self.feat_att_layer = SoftDotAttention(hidden_size, args.feature_size+args.angle_feat_size)
        if args.denseObj:
            self.dense_att_layer = SoftDotAttention(hidden_size, args.feature_size + args.angle_feat_size)
        if args.sparseObj:
            self.sparse_att_layer = SoftDotAttention(hidden_size, args.glove_emb+args.angle_bbox_size)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.candidate_att_layer = SoftDotAttention(hidden_size, args.feature_size+args.angle_feat_size)

    def forward(self, action, cand_feat,
                    prev_h1, c_0,
                ctx, ctx_mask=None,feature=None, sparseObj=None,denseObj=None,ObjFeature_mask=None,
                already_dropfeat=False):
        '''
        Takes a single step in the decoder LSTM (allowing sampling).
        action: batch x angle_feat_size
        feature: batch x 36 x (feature_size + angle_feat_size)
        cand_feat: batch x cand x (feature_size + angle_feat_size)
        h_0: batch x hidden_size
        prev_h1: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        already_dropfeat: used in EnvDrop
        '''
        action_embeds = self.embedding(action) #(64,64)

        # Adding Dropout
        action_embeds = self.drop(action_embeds)
        prev_h1_drop = self.drop(prev_h1)

        if not already_dropfeat:
            if sparseObj is not None:
                sparseObj[..., :-args.angle_bbox_size] = self.drop_env(sparseObj[..., :-args.angle_bbox_size])
                sparse_attn_feat, _ = self.sparse_att_layer(prev_h1_drop, sparseObj, mask=ObjFeature_mask,output_tilde=False)
            if denseObj is not None:
                denseObj[..., :-args.angle_feat_size] = self.drop_env(denseObj[..., :-args.angle_feat_size])
                dense_attn_feat, _ = self.dense_att_layer(prev_h1_drop, denseObj, mask=ObjFeature_mask,output_tilde=False)
            if feature is not None:
                # Dropout the raw feature as a common regularization
                feature[..., :-args.angle_feat_size] = self.drop_env(feature[..., :-args.angle_feat_size])   # Do not drop the last args.angle_feat_size (position feat)
                RN_attn_feat, _ = self.feat_att_layer(prev_h1_drop, feature, output_tilde=False)  # input: (64,512), (64,36,2176)

        if args.sparseObj and(not args.denseObj):
            attn_feat = sparse_attn_feat
            if args.catRN:
                attn_feat = torch.cat([RN_attn_feat,sparse_attn_feat],1)
        elif args.denseObj and (not args.sparseObj):
            attn_feat = dense_attn_feat
            if args.catRN:
                attn_feat = torch.cat([RN_attn_feat, dense_attn_feat], 1)
                # attn_feat = self.att_fc(attn_feat) # run denseObj_RN_FC_0
        elif args.denseObj and args.sparseObj:
            attn_feat = torch.cat([dense_attn_feat,sparse_attn_feat], 1)
            if args.catRN:
                attn_feat = torch.cat([RN_attn_feat, dense_attn_feat, sparse_attn_feat],1)
        else:
            attn_feat = RN_attn_feat

        concat_input = torch.cat((action_embeds, attn_feat), 1) # (batch, embedding_size+feature_size)  input:(64,64) output:(64,2240)
        h_1, c_1 = self.lstm(concat_input, (prev_h1, c_0)) #

        h_1_drop = self.drop(h_1)
        h_tilde, _ = self.attention_layer(h_1_drop, ctx, ctx_mask)

        # Adding Dropout
        h_tilde_drop = self.drop(h_tilde)

        if not already_dropfeat:
            cand_feat[..., :-args.angle_feat_size] = self.drop_env(cand_feat[..., :-args.angle_feat_size])

        _, logit = self.candidate_att_layer(h_tilde_drop, cand_feat, output_prob=False)

        return h_1, c_1, logit, h_tilde


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(args.rnn_dim, args.rnn_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.rnn_dim, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()

class SpeakerEncoder(nn.Module):
    def __init__(self, feature_size, hidden_size, dropout_ratio, bidirectional):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.feature_size = feature_size

        if bidirectional:
            print("BIDIR in speaker encoder!!")

        self.lstm = nn.LSTM(feature_size, self.hidden_size // self.num_directions, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop3 = nn.Dropout(p=args.featdropout)
        self.attention_layer = SoftDotAttention(self.hidden_size, feature_size)

        self.post_lstm = nn.LSTM(self.hidden_size, self.hidden_size // self.num_directions, self.num_layers,
                                 batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)

    def forward(self, action_embeds, feature, lengths, already_dropfeat=False):
        """
        :param action_embeds: (batch_size, length, 2052). The feature of the view
        :param feature: (batch_size, length, 36, 2052). The action taken (with the image feature)
        :param lengths: Not used in it
        :return: context with shape (batch_size, length, hidden_size)
        """
        x = action_embeds
        if not already_dropfeat:
            x[..., :-args.angle_feat_size] = self.drop3(x[..., :-args.angle_feat_size])            # Do not dropout the spatial features

        # LSTM on the action embed
        ctx, _ = self.lstm(x)
        ctx = self.drop(ctx)

        # Att and Handle with the shape
        batch_size, max_length, _ = ctx.size()
        if not already_dropfeat:
            feature[..., :-args.angle_feat_size] = self.drop3(feature[..., :-args.angle_feat_size])   # Dropout the image feature
        x, _ = self.attention_layer(                        # Attend to the feature map
            ctx.contiguous().view(-1, self.hidden_size),    # (batch, length, hidden) --> (batch x length, hidden)
            feature.view(batch_size * max_length, -1, self.feature_size),        # (batch, length, # of images, feature_size) --> (batch x length, # of images, feature_size)
        )
        x = x.view(batch_size, max_length, -1)
        x = self.drop(x)

        # Post LSTM layer
        x, _ = self.post_lstm(x)
        x = self.drop(x)

        return x

class SpeakerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx, hidden_size, dropout_ratio, glove=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.use_glove = glove is not None
        if self.use_glove:
            print('Using GloVe embedding')
            self.embedding.weight.data[...] = torch.from_numpy(glove)
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.drop = nn.Dropout(dropout_ratio)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size)
        self.baseline_projection = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(128, 1)
        )

    def forward(self, words, ctx, ctx_mask, h0, c0):
        embeds = self.embedding(words)
        if not self.use_glove:
            embeds = self.drop(embeds)
        x, (h1, c1) = self.lstm(embeds, (h0, c0))

        x = self.drop(x)

        # Get the size
        batchXlength = words.size(0) * words.size(1)
        multiplier = batchXlength // ctx.size(0)         # By using this, it also supports the beam-search

        # Att and Handle with the shape
        # Reshaping x          <the output> --> (b(word)*l(word), r)
        # Expand the ctx from  (b, a, r)    --> (b(word)*l(word), a, r)
        # Expand the ctx_mask  (b, a)       --> (b(word)*l(word), a)
        x, _ = self.attention_layer(
            x.contiguous().view(batchXlength, self.hidden_size),   # (5120, 512)
            ctx.unsqueeze(1).expand(-1, multiplier, -1, -1).contiguous(). view(batchXlength, -1, self.hidden_size), #(5120,22,512)
            mask=ctx_mask.unsqueeze(1).expand(-1, multiplier, -1).contiguous().view(batchXlength, -1) # (5120,22)
        )
        x = x.view(words.size(0), words.size(1), self.hidden_size)

        # Output the prediction logit
        x = self.drop(x)
        logit = self.projection(x)

        return logit, h1, c1

class AuxAng(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.LeakyReLU(),
                        nn.Linear(hidden_size, 4),
                        nn.Sigmoid()
        )

    def forward(self, f_t_hat):
        """
        :param action_embeds: (batch_size, length, 2052). The feature of the view
        :param feature: (batch_size, length, 36, 2052). The action taken (with the image feature)
        :param lengths: Not used in it
        :return: context with shape (batch_size, length, hidden_size)
        """
        ang_pre = self.fc(f_t_hat)
        return ang_pre

class AuxFea(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = args.rnn_dim
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, args.feature_size)

    def forward(self, h):
        h = self.relu1(self.fc1(h))
        h = self.fc2(h)
        return h

class AuxPro(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Sequential(
                        nn.Linear(hidden_size,hidden_size),
                        nn.LeakyReLU(),
                        nn.Linear(hidden_size, 1),
                        nn.Sigmoid()
                  )
    def forward(self, f_t_hat):
        """
        :param action_embeds: (batch_size, length, 2052). The feature of the view
        :param feature: (batch_size, length, 36, 2052). The action taken (with the image feature)
        :param lengths: Not used in it
        :return: context with shape (batch_size, length, hidden_size)
        """
        score = self.fc(f_t_hat)
        return score

class AuxMat(nn.Module):
    def __init__(self, hidden_size,prob):
        super().__init__()
        self.prob = prob
        self.fc = nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, f_t_hat, f_wl_avr, ended):
        batchsize = f_t_hat.size(0)
        idx = []
        shuffle_num = 0
        for f_idx, f in enumerate(ended):
            if(f==False):
                idx.append(f_idx)
                shuffle_num += 1
        # shuffle
        random.shuffle(idx)
        idx1 = idx[0:int(shuffle_num * self.prob / 2)]
        idx2 = idx[-int(shuffle_num * self.prob / 2):]
        mt = torch.zeros(batchsize).cuda()

        perm1 = [i for i in range(batchsize)]
        for i in range(int(shuffle_num*self.prob/2)):
            t = perm1[idx1[i]]
            perm1[idx1[i]] = perm1[idx2[i]]
            perm1[idx2[i]] = t
            mt[idx1[i]] = mt[idx2[i]] = 1
        # concat
        f_wl_avr = f_wl_avr[perm1]
        input = torch.cat((f_t_hat, f_wl_avr),1)
        score = self.fc(input).squeeze(1)
        return score, mt

class Comfc(nn.Module):
    def __init__(self, hidden_size, dropout_ratio):
        super().__init__()
        self.fc = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size, bias=False),
                        nn.Dropout(p=dropout_ratio)
                  )
    def forward(self, f_t_hat):
        """
        :param action_embeds: (batch_size, length, 2052). The feature of the view
        :param feature: (batch_size, length, 36, 2052). The action taken (with the image feature)
        :param lengths: Not used in it
        :return: context with shape (batch_size, length, hidden_size)
        """
        f_t_hat_fc = self.fc(f_t_hat)
        return f_t_hat_fc

