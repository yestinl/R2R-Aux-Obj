
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from param import args


class EncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx, 
                            dropout_ratio, bidirectional=False, num_layers=1):
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        if bidirectional:
            print("Using Bidir in EncoderLSTM")
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        input_size = embedding_size
        self.lstm = nn.LSTM(input_size, hidden_size, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, 
                            bidirectional=bidirectional)
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
            hidden_size * self.num_directions
        )

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

class MultiHeadSelfAttention(nn.Module):
    '''Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self,num_heads, query_dim, ctx_dim):
        '''Initialize layer.'''
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.linear_in = []
        for i in range(self.num_heads):
            self.linear_in.append(nn.Linear(query_dim,ctx_dim,bias=False).cuda())
        self.sm = nn.Softmax()
        self.linear_concat_out = nn.Linear(self.num_heads*ctx_dim+query_dim, query_dim, bias=False)
        # self.linear_out = nn.Linear(self.num_heads*ctx_dim, ctx_dim,bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        append_logit = []
        append_weighted_context = []
        append_attn = []

        for i in range(self.num_heads):

            target = self.linear_in[i](h).unsqueeze(2)  # batch x dim x 1

            # Get attention
            attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
            logit = attn

            if mask is not None:
                # -Inf masking prior to the softmax
                attn.masked_fill_(mask.bool(), -float('inf'))
            attn = self.sm(attn)    # There will be a bug here, but it's actually a problem in torch source code.
            attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

            weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
            append_logit.append(logit)                        # num_head x batch x seq_len
            append_weighted_context.append(weighted_context)
            append_attn.append(attn)

        output_logit = torch.stack(append_logit)
        output_weighted_context = torch.cat(append_weighted_context,1)
        output_attn = torch.stack(append_attn)

        output_logit = output_logit.mean(dim=0)
        output_attn = output_attn.mean(dim=0)

        if not output_prob:
            output_attn = output_logit
        if output_tilde:
            h_tilde = torch.cat((output_weighted_context, h), 1)
            h_tilde = self.tanh(self.linear_concat_out(h_tilde))
            return h_tilde, output_attn
        else:
            # output_weighted_context = self.linear_out(output_weighted_context)
            return output_weighted_context, output_attn

class Gate(nn.Module):
    '''Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, query_dim, ctx_dim):
        '''Initialize layer.'''
        super(Gate, self).__init__()
        self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sg = nn.Sigmoid()
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
        if args.objInputMode == 'sg':
            attn = self.sg(attn)    # There will be a bug here, but it's actually a problem in torch source code.
        elif args.objInputMode == 'tanh':
            attn = self.tanh(attn)
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
                print("Train in denseObj cat RN mode")

                print("Train in denseObj cat%s mode"%args.catfeat)
                if args.catfeat == 'none':
                    feature_size = args.feature_size * 2 + args.angle_feat_size  # 4352
                elif args.catfeat == 'bboxAngle':
                    feature_size = args.feature_size * 2 + args.angle_feat_size * 3  # 4352
                else:
                    feature_size = args.feature_size * 2 + args.angle_feat_size * 2
                # self.att_fc = nn.Linear(feature_size, args.feature_size) # run denseObj_RN_FC_0
            if args.addRN:
                print("Train in denseObj add RN mode")
                feature_size = args.feature_size + args.angle_feat_size


        elif args.denseObj and args.sparseObj:
            print("Train in sparseObj + denseObj mode")
            feature_size = args.feature_size+args.angle_feat_size+args.glove_emb+args.angle_bbox_size # 2484
            if args.catRN:
                print("Train in sparseObj+denseObj+RN mode")
                feature_size = args.feature_size*2+args.angle_feat_size*2+args.glove_emb+args.angle_bbox_size # 4660
        else:
            print("Train in RN mode")
            feature_size = args.feature_size+args.angle_feat_size # 2176
        print('feature_size: %d'%feature_size)
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Sequential(
            nn.Linear(args.angle_feat_size, self.embedding_size),
            nn.Tanh()
        )
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop_env = nn.Dropout(p=args.featdropout)

        if args.multiMode == "vis":
            self.feat_att_layer = MultiHeadSelfAttention(args.headNum, hidden_size, args.feature_size + args.angle_feat_size)
            self.lstm = nn.LSTMCell(embedding_size+feature_size*args.headNum, hidden_size)
        else:
            self.lstm = nn.LSTMCell(embedding_size + feature_size, hidden_size)
            self.feat_att_layer = SoftDotAttention(hidden_size, args.feature_size + args.angle_feat_size)
        if args.denseObj:
            print("Train in %s mode."%args.objInputMode)
            if (args.objInputMode == 'sg') or (args.objInputMode == 'tanh'):
                if args.catfeat == 'none':
                    self.dense_input_layer = Gate(hidden_size, args.feature_size)
                elif args.catfeat == 'bboxAngle':
                    self.dense_input_layer = Gate(hidden_size, args.feature_size + args.angle_feat_size*2)
                else:
                    self.dense_input_layer = Gate(hidden_size, args.feature_size + args.angle_feat_size)
            elif args.objInputMode == 'sm':
                if args.catfeat == 'none':
                    self.dense_input_layer = SoftDotAttention(hidden_size, args.feature_size)
                elif args.catfeat == 'bboxAngle':
                    self.dense_input_layer = SoftDotAttention(hidden_size, args.feature_size + args.angle_feat_size * 2)
                else:
                    self.dense_input_layer = SoftDotAttention(hidden_size, args.feature_size + args.angle_feat_size)
            # self.dense_att_layer = SoftDotAttention(hidden_size, args.feature_size)
        if args.sparseObj:
            print("Train in %s mode"%args.objInputMode)
            if (args.objInputMode == 'sg') or (args.objInputMode == 'tanh'):
                self.sparse_input_layer = Gate(hidden_size, args.glove_emb + args.angle_bbox_size)
            elif args.objInputMode == 'sm':
                self.sparse_input_layer = SoftDotAttention(hidden_size, args.glove_emb+args.angle_bbox_size)

        if args.multiMode == 'ins':
            self.attention_layer = MultiHeadSelfAttention(args.headNum, hidden_size, hidden_size)
        else:
            self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        if args.multiMode == 'can':
            self.candidate_att_layer = MultiHeadSelfAttention(args.headNum, hidden_size, args.feature_size+args.angle_feat_size)
        else:
            self.candidate_att_layer = SoftDotAttention(hidden_size, args.feature_size+args.angle_feat_size)


    def forward(self, action, cand_feat,
                prev_h1, c_0,
                ctx, ctx_mask=None,feature=None, sparseObj=None,denseObj=None,ObjFeature_mask=None,
                already_dropfeat=False,multi=False):
        '''
        Takes a single step in the decoder LSTM (allowing sampling).
        action: batch x angle_feat_size
        feature: batch x 36 x (feature_size + angle_feat_size)
        cand_feat: batch x cand x (feature_size + angle_feat_size)
        prev_h1: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        already_dropfeat: used in EnvDrop
        '''
        action_embeds = self.embedding(action)

        # Adding Dropout
        action_embeds = self.drop(action_embeds)

        if not already_dropfeat:
            if sparseObj is not None:
                sparseObj[..., :-args.angle_bbox_size] = self.drop_env(sparseObj[..., :-args.angle_bbox_size])
            if denseObj is not None:
                if args.catfeat == 'none':
                    denseObj = self.drop_env(denseObj)
                elif args.catfeat == 'bboxAngle':
                    denseObj[..., -args.angle_feat_size*2] = self.drop_env(denseObj[..., -args.angle_feat_size*2])
                else:
                    denseObj[..., -args.angle_feat_size] = self.drop_env(denseObj[..., -args.angle_feat_size])
            if feature is not None:
                # Dropout the raw feature as a common regularization
                feature[..., :-args.angle_feat_size] = self.drop_env(
                    feature[..., :-args.angle_feat_size])  # Do not drop the last args.angle_feat_size (position feat)

        prev_h1_drop = self.drop(prev_h1)

        if sparseObj is not None:
            sparse_attn_feat, _ = self.sparse_input_layer(prev_h1_drop, sparseObj, mask=ObjFeature_mask,
                                                        output_tilde=False)
        if denseObj is not None:
            dense_input_feat, _ = self.dense_input_layer(prev_h1_drop, denseObj, mask=ObjFeature_mask,
                                                          output_tilde=False)  # input:(64,512)(64,k,2176) output:(64,2176)

        if feature is not None:
            # Dropout the raw feature as a common regularization
            RN_attn_feat, _ = self.feat_att_layer(prev_h1_drop, feature,
                                                  output_tilde=False)  # input: (64,512), (64,36,2176) output:(64,2176)

        if args.sparseObj and(not args.denseObj):
            attn_feat = sparse_attn_feat
            if args.catRN:
                attn_feat = torch.cat([RN_attn_feat,sparse_attn_feat],1)
        elif args.denseObj and (not args.sparseObj):
            attn_feat = dense_input_feat
            if args.catRN:
                attn_feat = torch.cat([RN_attn_feat, dense_input_feat], 1) #(64,4352)
                # attn_feat = self.att_fc(attn_feat) # run denseObj_RN_FC_0
            if args.addRN:
                attn_feat = dense_input_feat+RN_attn_feat[:,:args.feature_size]
                attn_feat = torch.cat([attn_feat, RN_attn_feat[:,args.feature_size:]], 1)
        elif args.denseObj and args.sparseObj:
            attn_feat = torch.cat([dense_input_feat,sparse_attn_feat], 1)
            if args.catRN:
                attn_feat = torch.cat([RN_attn_feat, dense_input_feat, sparse_attn_feat],1)
        else:
            attn_feat = RN_attn_feat
        # attn_feat, _ = self.feat_att_layer(prev_h1_drop, feature, output_tilde=False)

        concat_input = torch.cat((action_embeds, attn_feat), 1) # (batch, embedding_size+feature_size)
        h_1, c_1 = self.lstm(concat_input, (prev_h1, c_0))

        h_1_drop = self.drop(h_1)

        if multi:
            h_tilde = torch.zeros_like(h_1_drop).cuda()
            for i in range(args.multiNum):
                h_tilde_i, _ = self.attention_layer(h_1_drop, ctx[i], ctx_mask[i])
                h_tilde += h_tilde_i
            h_tilde /= args.multiNum
        else:
            h_tilde, _ = self.attention_layer(h_1_drop, ctx, ctx_mask)

        # Adding Dropout
        h_tilde_drop = self.drop(h_tilde)

        if not already_dropfeat:
            cand_feat[..., :-args.angle_feat_size] = self.drop_env(cand_feat[..., :-args.angle_feat_size])

        _, logit = self.candidate_att_layer(h_tilde_drop, cand_feat, output_prob=False)

        return h_1, c_1, logit, h_tilde

class AttnDecoderLSTM_LongCat(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, embedding_size, hidden_size,
                       dropout_ratio):
        super(AttnDecoderLSTM_LongCat, self).__init__()
        if args.sparseObj and (not args.denseObj):
            print("Train in sparseObj mode")
            feature_size = args.glove_emb+args.angle_bbox_size  # 308
            if args.catRN:
                print("Train in sparseObj+RN mode")
                feature_size = args.glove_emb+args.angle_bbox_size+args.feature_size+args.angle_feat_size # 2484
        elif args.denseObj and (not args.sparseObj):
            print("Train in denseObj mode")
            # feature_size = args.feature_size+args.angle_feat_size   # 2176
            if args.catRN:
                print("Train in denseObj long cat RN mode")
                # feature_size = args.feature_size*2+args.angle_feat_size*2 # 4352
                feature_size = args.feature_size + args.angle_feat_size  # 2176
                # self.att_fc = nn.Linear(feature_size, args.feature_size) # run denseObj_RN_FC_0
        #     if args.addRN:
        #         print("Train in denseObj add RN mode")
        #         feature_size = args.feature_size + args.angle_feat_size
        # elif args.denseObj and args.sparseObj:
        #     print("Train in sparseObj + denseObj mode")
        #     feature_size = args.feature_size+args.angle_feat_size+args.glove_emb+args.angle_bbox_size # 2484
        #     if args.catRN:
        #         print("Train in sparseObj+denseObj+RN mode")
        #         feature_size = args.feature_size*2+args.angle_feat_size*2+args.glove_emb+args.angle_bbox_size # 4660
        # else:
        #     print("Train in RN mode")
        #     feature_size = args.feature_size+args.angle_feat_size # 2176
        print('feature_size: %d'%feature_size)
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Sequential(
            nn.Linear(args.angle_feat_size, self.embedding_size),
            nn.Tanh()
        )
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop_env = nn.Dropout(p=args.featdropout)
        self.lstm_v = nn.LSTMCell(embedding_size+feature_size, hidden_size)
        self.lstm_o = nn.LSTMCell(embedding_size+feature_size, hidden_size)
        self.feat_att_layer = SoftDotAttention(hidden_size, args.feature_size+args.angle_feat_size)
        if args.denseObj:
            self.dense_att_layer = SoftDotAttention(hidden_size, args.feature_size + args.angle_feat_size)
            # self.dense_att_layer = SoftDotAttention(hidden_size, args.feature_size)
        if args.sparseObj:
            self.sparse_att_layer = SoftDotAttention(hidden_size, args.glove_emb+args.angle_bbox_size)
        self.attention_layer_v = SoftDotAttention(hidden_size, hidden_size)
        self.attention_layer_o = SoftDotAttention(hidden_size, hidden_size)
        self.candidate_att_layer = SoftDotAttention(hidden_size*2, args.feature_size+args.angle_feat_size)

    def forward(self, action, cand_feat,
                prev_h1_v,prev_h1_o, c_0_v, c_0_o,
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
        action_embeds = self.embedding(action)

        # Adding Dropout
        action_embeds = self.drop(action_embeds)

        if not already_dropfeat:
            if sparseObj is not None:
                sparseObj[..., :-args.angle_bbox_size] = self.drop_env(sparseObj[..., :-args.angle_bbox_size])
            if denseObj is not None:
                # denseObj[..., :-args.angle_feat_size] = self.drop_env(denseObj[..., :-args.angle_feat_size])
                denseObj[..., -args.angle_feat_size] = self.drop_env(denseObj[..., -args.angle_feat_size])
            if feature is not None:
                # Dropout the raw feature as a common regularization
                feature[..., :-args.angle_feat_size] = self.drop_env(
                    feature[..., :-args.angle_feat_size])  # Do not drop the last args.angle_feat_size (position feat)

        prev_h1_v_drop = self.drop(prev_h1_v)
        prev_h1_o_drop = self.drop(prev_h1_o)

        # if sparseObj is not None:
        #     sparse_attn_feat, _ = self.sparse_att_layer(prev_h1_drop, sparseObj, mask=ObjFeature_mask,
        #                                                 output_tilde=False)
        if denseObj is not None:
            # denseObj[..., :-args.angle_feat_size] = self.drop_env(denseObj[..., :-args.angle_feat_size])
            dense_attn_feat, _ = self.dense_att_layer(prev_h1_o_drop, denseObj, mask=ObjFeature_mask,
                                                      output_tilde=False)  # input:(64,512)(64,k,2176) output:(64,2176)
        if feature is not None:
            # Dropout the raw feature as a common regularization
            RN_attn_feat, _ = self.feat_att_layer(prev_h1_v_drop, feature,
                                                  output_tilde=False)  # input: (64,512), (64,36,2176) output:(64,2176)

        # if args.sparseObj and(not args.denseObj):
        #     attn_feat = sparse_attn_feat
        #     if args.catRN:
        #         attn_feat = torch.cat([RN_attn_feat,sparse_attn_feat],1)
        # elif args.denseObj and (not args.sparseObj):
        #     attn_feat = dense_attn_feat
        #     if args.catRN:
        #         attn_feat = torch.cat([RN_attn_feat, dense_attn_feat], 1) #(64,4352)
        #         # attn_feat = self.att_fc(attn_feat) # run denseObj_RN_FC_0
        #     if args.addRN:
        #         attn_feat = dense_attn_feat+RN_attn_feat[:,:args.feature_size]
        #         attn_feat = torch.cat([attn_feat, RN_attn_feat[:,args.feature_size:]], 1)
        # elif args.denseObj and args.sparseObj:
        #     attn_feat = torch.cat([dense_attn_feat,sparse_attn_feat], 1)
        #     if args.catRN:
        #         attn_feat = torch.cat([RN_attn_feat, dense_attn_feat, sparse_attn_feat],1)
        # else:
        #     attn_feat = RN_attn_feat
        # attn_feat, _ = self.feat_att_layer(prev_h1_drop, feature, output_tilde=False)

        concat_input_v = torch.cat((action_embeds, RN_attn_feat), 1) # (batch, embedding_size+feature_size)
        h_1_v, c_1_v = self.lstm_v(concat_input_v, (prev_h1_v, c_0_o))
        h_1_drop_v = self.drop(h_1_v)
        h_tilde_v, _= self.attention_layer_v(h_1_drop_v, ctx, ctx_mask)
        # h_tilde_v_drop = self.drop(h_tilde_v)

        concat_input_o = torch.cat((action_embeds, dense_attn_feat), 1)
        h_1_o, c_1_o = self.lstm_o(concat_input_o, (prev_h1_o, c_0_o))
        h_1_o_drop = self.drop(h_1_o)
        h_tilde_o, _ = self.attention_layer_o(h_1_o_drop, ctx, ctx_mask)
        # h_tilde_o_drop = self.drop(h_tilde_o)

        h_tilde = torch.cat((h_tilde_v, h_tilde_o), 1)
        h_tilde_drop = self.drop(h_tilde)


        if not already_dropfeat:
            cand_feat[..., :-args.angle_feat_size] = self.drop_env(cand_feat[..., :-args.angle_feat_size])

        _, logit = self.candidate_att_layer(h_tilde_drop, cand_feat, output_prob=False)

        return h_1_v, h_1_o, c_1_v, c_1_o, logit, h_tilde_v, h_tilde_o


class MatchingNetwork(nn.Module):
    def __init__(self):
        super(MatchingNetwork, self).__init__()
        hidden_size = args.rnn_dim
        self.fc1 = nn.Linear(hidden_size*2, hidden_size)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        h = self.relu1(self.fc1(h))
        h = self.sigmoid(self.fc2(h))
        # h = torch.mean(h, dim=1) # pooling, harm performance
        return h

class FeaturePredictor(nn.Module):
    def __init__(self):
        super(FeaturePredictor, self).__init__()
        hidden_size = args.rnn_dim
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, args.feature_size)

    def forward(self, h):
        h = self.relu1(self.fc1(h))
        h = self.fc2(h)
        return h

class AnglePredictor(nn.Module):
    def __init__(self):
        super(AnglePredictor, self).__init__()
        hidden_size = args.rnn_dim
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, 4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        h = self.relu1(self.fc1(h))
        h = self.sigmoid(self.fc2(h))
        return h

class ProgressIndicator(nn.Module):
    def __init__(self):
        super(ProgressIndicator, self).__init__()
        hidden_size = args.rnn_dim
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        h = self.relu1(self.fc1(h))
        h = self.sigmoid(self.fc2(h))
        return h

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

        if args.denseObj:
            self.obj_gate = Gate(self.hidden_size, args.feature_size)
            self.post_lstm = nn.LSTM(self.hidden_size*2, self.hidden_size // self.num_directions, self.num_layers,
                                 batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        else:
            self.post_lstm = nn.LSTM(self.hidden_size, self.hidden_size // self.num_directions, self.num_layers,
                                     batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        self.lstm = nn.LSTM(feature_size, self.hidden_size // self.num_directions, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop3 = nn.Dropout(p=args.featdropout)
        self.attention_layer = SoftDotAttention(self.hidden_size, feature_size)



    def forward(self, action_embeds, feature, lengths, already_dropfeat=False, objMask=None, objFeat=None):
        """
        :param action_embeds: (batch_size, length, 2052). The feature of the view
        :param feature: (batch_size, length, 36, 2052). The action taken (with the image feature)
        :param lengths: Not used in it
        :return: context with shape (batch_size, length, hidden_size)
        """
        x = action_embeds
        if not already_dropfeat:
            x[..., :-args.angle_feat_size] = self.drop3(x[..., :-args.angle_feat_size])            # Do not dropout the spatial features
            if args.denseObj:
                objFeat = self.drop3(objFeat)

        # LSTM on the action embed
        ctx, _ = self.lstm(x)
        ctx = self.drop(ctx)

        # Att and Handle with the shape
        batch_size, max_length, _ = ctx.size()
        if not already_dropfeat:
            feature[..., :-args.angle_feat_size] = self.drop3(feature[..., :-args.angle_feat_size])   # Dropout the image feature

        x, _ = self.attention_layer(                        # Attend to the feature map
            ctx.contiguous().view(-1, self.hidden_size),    # (batch, length, hidden) --> (batch x length, hidden)
            feature.view(batch_size * max_length, -1, self.feature_size)        # (batch, length, # of images, feature_size) --> (batch x length, # of images, feature_size)
        )
        if objFeat is not None:
            x2, _ = self.obj_gate(
                ctx.contiguous().view(-1, self.hidden_size),
                objFeat.view(batch_size*max_length, -1, objFeat.size(-1)),
                objMask.view(batch_size*max_length,-1)
            )
            x = torch.cat((x,x2),dim=1)

        x = x.view(batch_size, max_length, -1)
        x = self.drop(x)

        # Post LSTM layer
        x, _ = self.post_lstm(x)
        x = self.drop(x)

        return x

class SpeakerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx, hidden_size, dropout_ratio):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.drop = nn.Dropout(dropout_ratio)
        # if args.denseObj:
        #     self.attention_layer = SoftDotAttention(hidden_size*2, hidden_size)
        # else:
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size)
        self.baseline_projection = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(128, 1)
        )

    def forward(self, words, ctx, ctx_mask, h0, c0, objMask=None, objFeat=None):
        embeds = self.embedding(words)
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
            x.contiguous().view(batchXlength, self.hidden_size),
            ctx.unsqueeze(1).expand(-1, multiplier, -1, -1).contiguous(). view(batchXlength, -1, self.hidden_size),
            mask=ctx_mask.unsqueeze(1).expand(-1, multiplier, -1).contiguous().view(batchXlength, -1)
        )
        x = x.view(words.size(0), words.size(1), self.hidden_size)

        # Output the prediction logit
        x = self.drop(x)
        logit = self.projection(x)

        return logit, h1, c1

class SpeakerDecoder_SameLSTM(SpeakerDecoder):
    def __init__(self, vocab_size, embedding_size, padding_idx, hidden_size, dropout_ratio):
        super(SpeakerDecoder_SameLSTM, self).__init__(vocab_size, embedding_size, padding_idx, hidden_size, dropout_ratio)

    def forward(self, words, ctx, ctx_mask, x):
        # embeds = self.embedding(words)
        # embeds = self.drop(embeds)
        # x, (h1, c1) = self.lstm(embeds, (h0, c0))

        # x = self.drop(x)

        # Get the size
        batchXlength = words.size(0) * words.size(1)
        multiplier = batchXlength // ctx.size(0)         # By using this, it also supports the beam-search

        # Att and Handle with the shape
        # Reshaping x          <the output> --> (b(word)*l(word), r)
        # Expand the ctx from  (b, a, r)    --> (b(word)*l(word), a, r)
        # Expand the ctx_mask  (b, a)       --> (b(word)*l(word), a)
        x, _ = self.attention_layer(
            x.contiguous().view(batchXlength, self.hidden_size),
            ctx.unsqueeze(1).expand(-1, multiplier, -1, -1).contiguous(). view(batchXlength, -1, self.hidden_size),
            mask=ctx_mask.unsqueeze(1).expand(-1, multiplier, -1).contiguous().view(batchXlength, -1)
        )
        x = x.view(words.size(0), words.size(1), self.hidden_size)

        # Output the prediction logit
        x = self.drop(x)
        logit = self.projection(x)

        return logit




