
import json
import os
import sys
import numpy as np
import random
import math
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from env import R2RBatch
from utils import padding_idx, add_idx, Tokenizer,get_sync_dir
import utils
import model
import param
from param import args
from collections import defaultdict

from polyaxon_client.tracking import get_outputs_refs_paths

# refs_paths = get_outputs_refs_paths()['experiments']
SPEAKER_DIR = 'lyx/speaker/state_dict/best_val_unseen_bleu'

class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents
    
    def write_results(self):
        output = [{'instr_id':k, 'trajectory': v} for k,v in self.results.items()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def get_results(self):
        output = [{'instr_id': k, 'trajectory': v} for k, v in self.results.items()]
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        self.auxloss=0
        self.acc=0
        self.acc_num = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(**kwargs, val_spe=args.spe_weight):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj['path']
        else:   # Do a full round
            while True:
                for traj in self.rollout(**kwargs, val_spe=args.spe_weight):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = traj['path']
                if looped:
                    break
        # if args.spe_weight:
        #     print("acc", self.acc/self.acc_num)
        #     self.logs['word_acc'].append(self.acc/self.acc_num)

class Seq2SeqAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    env_actions = {
      'left': (0,-1, 0), # left
      'right': (0, 1, 0), # right
      'up': (0, 0, 1), # up
      'down': (0, 0,-1), # down
      'forward': (1, 0, 0), # forward
      '<end>': (0, 0, 0), # <end>
      '<start>': (0, 0, 0), # <start>
      '<ignore>': (0, 0, 0)  # <ignore>
    }

    def __init__(self, env, results_path, tok, episode_len=20):
        super(Seq2SeqAgent, self).__init__(env, results_path)
        self.tok = tok
        self.episode_len = episode_len
        self.feature_size = self.env.feature_size

        self.hidden_size = args.rnn_dim
        if args.useGlove:
            glove = np.load(args.GLOVE_PATH)
            word_embedding_size = 300
        else:
            word_embedding_size = args.wemb
            glove = None
        # R2R Models
        enc_hidden_size = args.rnn_dim//2 if args.bidir else args.rnn_dim
        self.encoder = model.EncoderLSTM(self.tok.vocab_size(), word_embedding_size, enc_hidden_size, padding_idx,
                                         args.dropout, bidirectional=args.bidir, glove=glove).cuda()
        self.decoder = model.AttnDecoderLSTM(args.aemb, args.rnn_dim, args.dropout).cuda()
        self.critic = model.Critic().cuda()
        self.models = (self.encoder, self.decoder, self.critic)

        # Auxiliary Task Model
        if args.spe_weight:
            self.Spe_decoder = model.SpeakerDecoder(self.tok.vocab_size(), word_embedding_size, padding_idx, args.rnn_dim, args.dropout, glove=glove).cuda()
            # print('yes:',refs_paths[0])
            # speaker_model = os.path.join(refs_paths[0], 'snap/speaker/state_dict/best_val_unseen_bleu')
            speaker_model = get_sync_dir(SPEAKER_DIR)
            print('use speaker model in %s'%(speaker_model))
            states = torch.load(speaker_model)
            # states = torch.load("snap/speaker/state_dict/best_val_unseen_bleu")
            self.Spe_decoder.load_state_dict(states["decoder"]["state_dict"])
        if args.pro_weight:
            self.Pro_fc = model.AuxPro(self.hidden_size).cuda()
        if args.ang_weight:
            self.Ang_fc = model.AuxAng(self.hidden_size).cuda()
        if args.mat_weight:
            self.Mat_fc = model.AuxMat(self.hidden_size, args.shuffleprob).cuda()
        if args.fea_weight:
            self.Fea_fc = model.AuxFea().cuda()

        self.aux_models = (self.Spe_decoder, self.Pro_fc, self.Ang_fc, self.Fea_fc)


        # Optimizers
        self.encoder_optimizer = args.optimizer(self.encoder.parameters(), lr=args.lr)
        self.decoder_optimizer = args.optimizer(self.decoder.parameters(), lr=args.lr)
        self.critic_optimizer = args.optimizer(self.critic.parameters(), lr=args.lr)
        if args.spe_weight:
            self.Spe_decoder_optimizer = args.optimizer(self.Spe_decoder.parameters(), lr=args.lr)
        if args.pro_weight:
            self.Pro_fc_optimizer = args.optimizer(self.Pro_fc.parameters(), lr=args.lr)
        if args.mat_weight:
            self.Mat_fc_optimizer = args.optimizer(self.Mat_fc.parameters(), lr=args.lr)
        if args.ang_weight:
            self.Ang_fc_optimizer = args.optimizer(self.Ang_fc.parameters(), lr=args.lr)
        if args.fea_weight:
            self.Fea_fc_optimizer = args.optimizer(self.Fea_fc.parameters(), lr=args.lr)

        self.aux_optimizer = (self.Spe_decoder_optimizer, self.Pro_fc_optimizer,
                                            self.Ang_fc_optimizer, self.Fea_fc_optimizer)
        self.optimizers = (self.encoder_optimizer, self.decoder_optimizer, self.critic_optimizer)
        # Evaluations
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignoreid, size_average=False)
        if args.spe_weight:
            self.spe_criterion = nn.CrossEntropyLoss(ignore_index=self.tok.word_to_index['<PAD>'])
        if args.pro_weight:
            self.pro_criterion = nn.BCELoss()
        if args.mat_weight:
            self.mat_criterion = nn.BCELoss()
        if args.ang_weight:
            self.ang_criterion = nn.MSELoss()
        if args.fea_weight:
            self.fea_criterion = nn.MSELoss()
        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)

    def _sort_batch(self, obs):
        ''' Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). '''

        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]     # Full length

        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        seq_lengths, perm_idx = seq_lengths.sort(0, True)       # True -> descending
        sorted_tensor = seq_tensor[perm_idx]
        mask = (sorted_tensor == padding_idx)[:,:seq_lengths[0]]    # seq_lengths[0] is the Maximum length

        return Variable(sorted_tensor, requires_grad=False).long().cuda(), \
               mask.byte().cuda(),  \
               list(seq_lengths), list(perm_idx)

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        if args.sparseObj and (not args.denseObj):
            Obj_leng = [ob['obj_s_feature'].shape[0] for ob in obs]
            if not args.catRN:
                sparseObj = np.zeros((len(obs), max(Obj_leng), args.glove_emb + args.angle_bbox_size), dtype=np.float32)
                for i, ob in enumerate(obs):
                    sparseObj[i, :Obj_leng[i], :args.glove_emb] = ob['obj_s_feature']
                    sparseObj[i, :Obj_leng[i], -args.angle_bbox_size:] = np.append(ob['bbox_angle_e'], ob['bbox_angle_h'], axis=1)
                return Variable(torch.from_numpy(sparseObj), requires_grad=False).cuda(), Obj_leng
            else:
                sparseObj = np.zeros((len(obs), max(Obj_leng), args.glove_emb + args.angle_bbox_size),
                                     dtype=np.float32)
                for i, ob in enumerate(obs):
                    sparseObj[i, :Obj_leng[i], :args.glove_emb] = ob['obj_s_feature']
                    sparseObj[i, :Obj_leng[i], -args.angle_bbox_size:] = np.append(ob['bbox_angle_e'],
                                                                                   ob['bbox_angle_h'], axis=1)
                features = np.empty((len(obs), args.views, self.feature_size + args.angle_feat_size),
                                    dtype=np.float32)
                for i, ob in enumerate(obs):
                    features[i, :, :] = ob['feature']
                return Variable(torch.from_numpy(sparseObj), requires_grad=False).cuda(), Obj_leng, Variable(torch.from_numpy(features), requires_grad=False).cuda()
        elif args.denseObj and (not args.sparseObj):
            Obj_leng = [ob['obj_d_feature'].shape[0] for ob in obs]
            if not args.catRN:
                denseObj = np.zeros((len(obs), max(Obj_leng), args.feature_size+args.angle_feat_size), dtype=np.float32)
                for i, ob in enumerate(obs):
                    denseObj[i, :Obj_leng[i], :args.feature_size] = ob['obj_d_feature']
                    denseObj[i, :Obj_leng[i], -args.angle_feat_size:] = np.repeat(np.append(ob['bbox_angle_e'],
                                                                                            ob['bbox_angle_h'], axis=1), args.angle_feat_size//8, axis=1)
                return Variable(torch.from_numpy(denseObj), requires_grad=False).cuda(), Obj_leng
            else:
                denseObj = np.zeros((len(obs), max(Obj_leng), args.feature_size+args.angle_feat_size),
                                     dtype=np.float32)
                for i, ob in enumerate(obs):
                    denseObj[i, :Obj_leng[i], :args.feature_size] = ob['obj_d_feature']
                    denseObj[i, :Obj_leng[i], -args.angle_feat_size:] = np.repeat(np.append(ob['bbox_angle_e'],
                                                                                            ob['bbox_angle_h'], axis=1), args.angle_feat_size//8, axis=1)
                features = np.empty((len(obs), args.views, self.feature_size + args.angle_feat_size),
                                    dtype=np.float32)
                for i, ob in enumerate(obs):
                    features[i, :, :] = ob['feature']
                return Variable(torch.from_numpy(denseObj), requires_grad=False).cuda(), Obj_leng, Variable(
                    torch.from_numpy(features), requires_grad=False).cuda()
        elif args.denseObj and args.sparseObj:
            Obj_leng = [ob['obj_d_feature'].shape[0] for ob in obs]
            if not args.catRN:
                denseObj = np.zeros((len(obs), max(Obj_leng), args.feature_size + args.angle_feat_size),
                                    dtype=np.float32)
                for i, ob in enumerate(obs):
                    denseObj[i, :Obj_leng[i], :args.feature_size] = ob['obj_d_feature']
                    denseObj[i, :Obj_leng[i], -args.angle_feat_size:] = np.repeat(np.append(ob['bbox_angle_e'],
                                                                                            ob['bbox_angle_h'], axis=1), args.angle_feat_size//8, axis=1)
                sparseObj = np.zeros((len(obs), max(Obj_leng), args.glove_emb + args.angle_bbox_size), dtype=np.float32)
                for i, ob in enumerate(obs):
                    sparseObj[i, :Obj_leng[i], :args.glove_emb] = ob['obj_s_feature']
                    sparseObj[i, :Obj_leng[i], -args.angle_bbox_size:] = np.append(ob['bbox_angle_e'],
                                                                                   ob['bbox_angle_h'], axis=1)
                return Variable(torch.from_numpy(sparseObj), requires_grad=False).cuda(),Variable(torch.from_numpy(denseObj), requires_grad=False).cuda(),  Obj_leng
            else:
                denseObj = np.zeros((len(obs), max(Obj_leng), args.feature_size + args.angle_feat_size),
                                    dtype=np.float32)
                for i, ob in enumerate(obs):
                    denseObj[i, :Obj_leng[i], :args.feature_size] = ob['obj_d_feature']
                    denseObj[i, :Obj_leng[i], -args.angle_feat_size:] = np.repeat(np.append(ob['bbox_angle_e'],
                                                                                            ob['bbox_angle_h'], axis=1), args.angle_feat_size//8, axis=1)
                sparseObj = np.zeros((len(obs), max(Obj_leng), args.glove_emb + args.angle_bbox_size), dtype=np.float32)
                for i, ob in enumerate(obs):
                    sparseObj[i, :Obj_leng[i], :args.glove_emb] = ob['obj_s_feature']
                    sparseObj[i, :Obj_leng[i], -args.angle_bbox_size:] = np.append(ob['bbox_angle_e'],
                                                                                   ob['bbox_angle_h'], axis=1)
                features = np.empty((len(obs), args.views, self.feature_size + args.angle_feat_size),
                                    dtype=np.float32)
                for i, ob in enumerate(obs):
                    features[i, :, :] = ob['feature']
                return Variable(torch.from_numpy(sparseObj), requires_grad=False).cuda(), Variable(torch.from_numpy(denseObj), requires_grad=False).cuda(), Obj_leng, Variable(
                    torch.from_numpy(features), requires_grad=False).cuda()
        else:
            features = np.empty((len(obs), args.views, self.feature_size + args.angle_feat_size), dtype=np.float32)
            for i, ob in enumerate(obs):
                features[i, :, :] = ob['feature']  # Image feat
            return Variable(torch.from_numpy(features), requires_grad=False).cuda()

    def _candidate_variable(self, obs):
        candidate_leng = [len(ob['candidate']) + 1 for ob in obs]       # +1 is for the end
        candidate_feat = np.zeros((len(obs), max(candidate_leng), self.feature_size + args.angle_feat_size), dtype=np.float32)
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, c in enumerate(ob['candidate']):
                candidate_feat[i, j, :] = c['feature']                         # Image feat
        return torch.from_numpy(candidate_feat).cuda(), candidate_leng

    def get_input_feat(self, obs):
        input_a_t = np.zeros((len(obs), args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            input_a_t[i] = utils.angle_feature(ob['heading'], ob['elevation'])
        input_a_t = torch.from_numpy(input_a_t).cuda()
        candidate_feat, candidate_leng = self._candidate_variable(obs)
        if args.sparseObj and (not args.denseObj):
            if not args.catRN:
                sparseObj, Obj_leng = self._feature_variable(obs)
                return input_a_t, sparseObj, Obj_leng, candidate_feat, candidate_leng
            else:
                sparseObj, Obj_leng, features = self._feature_variable(obs)
                return input_a_t, sparseObj, Obj_leng,features, candidate_feat, candidate_leng
        elif args.denseObj and (not args.sparseObj):
            if not args.catRN:
                denseObj, Obj_leng = self._feature_variable(obs)
                return input_a_t, denseObj, Obj_leng, candidate_feat, candidate_leng
            else:
                denseObj, Obj_leng, features = self._feature_variable(obs)
                return input_a_t, denseObj, Obj_leng, features, candidate_feat, candidate_leng
        elif args.denseObj and args.sparseObj:
            if not args.catRN:
                sparseObj, denseObj, Obj_leng = self._feature_variable(obs)
                return input_a_t, sparseObj, denseObj, Obj_leng, candidate_feat, candidate_leng
            else:
                sparseObj, denseObj, Obj_leng, features = self._feature_variable(obs)
                return input_a_t, sparseObj, denseObj, Obj_leng, features, candidate_feat,candidate_leng
        else:
            f_t = self._feature_variable(obs)      # Image features from obs
            return input_a_t, f_t, candidate_feat, candidate_leng

    def _teacher_action(self, obs, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)  # (64,)
        # e = np.zeros((len(obs), args.angle_feat_size), dtype=np.float32)
        for i, ob in enumerate(obs):
            if ended[i]:  # Just ignore this index
                a[i] = args.ignoreid
                # e[i, :] = 0.0
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:  # Next view point
                        a[i] = k
                        # e[i, :] = utils.angle_feature(candidate['heading'], candidate['elevation'])
                        break
                else:  # Stop here
                    assert ob['teacher'] == ob['viewpoint']  # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate'])
                    # e[i, :] = 0.0
        return torch.from_numpy(a).cuda()

    def rollout(self, train_ml=None, train_rl=True, reset=True, speaker=None, val_spe=None, bt=None):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment
        :param speaker:     Speaker used in back translation.
                            If the speaker is not None, use back translation.
                            O.w., normal training
        :return:
        """
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False
        if reset:
            # Reset env
            obs = np.array(self.env.reset())
        else:
            obs = np.array(self.env._get_obs())

        batch_size = len(obs)

        if speaker is not None:         # Trigger the self_train mode!
            noise = self.decoder.drop_env(torch.ones(self.feature_size).cuda())
            batch = self.env.batch.copy()
            speaker.env = self.env
            insts = speaker.infer_batch(featdropmask=noise)     # Use the same drop mask in speaker

            # Create fake environments with the generated instruction
            boss = np.ones((batch_size, 1), np.int64) * self.tok.word_to_index['<BOS>']  # First word is <BOS>
            insts = np.concatenate((boss, insts), 1)
            for i, (datum, inst) in enumerate(zip(batch, insts)):
                if inst[-1] != self.tok.word_to_index['<PAD>']: # The inst is not ended!
                    inst[-1] = self.tok.word_to_index['<EOS>']
                datum.pop('instructions')
                datum.pop('instr_encoding')
                datum['instructions'] = self.tok.decode_sentence(inst)
                datum['instr_encoding'] = inst
            obs = np.array(self.env.reset(batch))

        # Reorder the language input for the encoder (do not ruin the original code)
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx]

        # Vision_Language Embedding
        f_wl_til, h0, c1 = self.encoder(seq, seq_lengths) # input: (64,80) (64,) output:(64,80,512)(64,512)
        # f_wl_avr = torch.sum(f_wl_til,dim=1)/torch.tensor(seq_lengths).unsqueeze(1).cuda().float()      #(64,512)
        f_wl_avr = h0.clone()

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(perm_obs):   # The init distance from the view point to the target
            last_dist[i] = ob['distance']

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in perm_obs]

        # For test result submission
        visited = [set() for _ in perm_obs]

        # Initialization the tracking state
        ended = np.array([False] * batch_size)  # Indices match permuation of the model, not env

        # Initialization the Trajectory
        # f_oT_til = torch.zeros(batch_size, self.episode_len, self.hidden_size).cuda()  # (64, 35, 512)
        # pro_score_T = torch.zeros(batch_size, self.episode_len).cuda()  # (64,35)
        pro_score_T = []
        f_oT_til = []

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        ml_loss = 0.
        spe_loss = 0.
        ang_loss = 0.
        fea_loss = 0.
        pro_loss = 0.
        # mat_loss = torch.zeros(batch_size).cuda()
        mat_loss = 0.
        traj_len = np.zeros(batch_size)  # (64,)
        r_t = torch.zeros(batch_size, self.episode_len).cuda()  # (64,35)

        h_last_t = h0
        for t in range(self.episode_len):
            ObjFeature_mask = None
            sparseObj = None
            denseObj = None
            f_t = None
            Obj_leng = None
            if args.sparseObj and (not args.denseObj):
                if not args.catRN:
                    input_a_t, sparseObj, Obj_leng, c_t, candidate_leng = self.get_input_feat(perm_obs)
                else:
                    input_a_t, sparseObj, Obj_leng, f_t, c_t, candidate_leng = self.get_input_feat(perm_obs)
            elif args.denseObj and (not args.sparseObj):
                if not args.catRN:
                    input_a_t, denseObj, Obj_leng, c_t, candidate_leng = self.get_input_feat(perm_obs)
                else:
                    input_a_t, denseObj, Obj_leng, f_t, c_t, candidate_leng = self.get_input_feat(perm_obs)
            elif args.denseObj and args.sparseObj:
                if not args.catRN:
                    input_a_t, sparseObj, denseObj, Obj_leng, c_t, candidate_leng = self.get_input_feat(perm_obs)
                else:
                    input_a_t, sparseObj, denseObj, Obj_leng, f_t, c_t, candidate_leng = self.get_input_feat(perm_obs)
            else:
                input_a_t, f_t, c_t, candidate_leng = self.get_input_feat(perm_obs)
            if Obj_leng is not None:
                ObjFeature_mask = utils.length2mask(Obj_leng)

            if speaker is not None:       # Apply the env drop mask to the feat
                c_t[..., :-args.angle_feat_size] *= noise
                f_t[..., :-args.angle_feat_size] *= noise


            f_ot_til, c1, logit, f_t_hat = self.decoder(input_a_t,c_t,
                                               h_last_t, c1,
                                               f_wl_til, seq_mask,feature=f_t,
                                               sparseObj=sparseObj,denseObj=denseObj,
                                               ObjFeature_mask=ObjFeature_mask,already_dropfeat=(speaker is not None))
            h_last_t = f_t_hat
            hidden_states.append(f_ot_til) #[(64,512)]

            if args.spe_weight:
                f_oT_til.append(f_ot_til)

            # Mask outputs where agent can't move forward
            # Here the logit is [b, max_candidate]
            candidate_mask = utils.length2mask(candidate_leng)
            if args.submit:     # Avoding cyclic path
                for ob_id, ob in enumerate(perm_obs):
                    visited[ob_id].add(ob['viewpoint'])
                    for c_id, c in enumerate(ob['candidate']):
                        if c['viewpointId'] in visited[ob_id]:
                            candidate_mask[ob_id][c_id] = 1
            logit.masked_fill_(candidate_mask.bool(), -float('inf'))
            target= self._teacher_action(perm_obs, ended)  # next viewpoint of current viewpoint,(64,)(64,128)

            # Supervised training
            if train_ml:
                ml_loss += self.criterion(logit, target)

            # Angle Prediction Task & feature Prediction Task
            mask = target == -100  #
            target_aux = target.clone()
            target_aux[mask] = 0
            target_aux = target_aux.unsqueeze(1).unsqueeze(2)  # (64,1,1)
            target_aux = target_aux.expand(-1, 1, c_t.size(2))  # (64,1,2176)
            selected_feat = torch.gather(c_t, 1, target_aux)  # (64,1,2176)
            selected_feat = selected_feat.squeeze(1)  # batch_size, feat_size (64,2176)
            selected_feat[mask] = 0
            feature_label = selected_feat[:, :-args.angle_feat_size]
            angle_label = selected_feat[:, -4:]
            if args.fea_weight:
                feature_pred = self.Fea_fc(f_t_hat)
                fea_loss += self.fea_criterion(feature_pred, feature_label)
            if args.ang_weight:
                ang_pred = self.Ang_fc(f_t_hat)
                ang_loss += self.ang_criterion(ang_pred, angle_label)

            # Progress Estimation Task
            if args.pro_weight:
                pro_score = self.Pro_fc(f_t_hat)  # (64,1)
                pro_score_T.append(pro_score)

            # Cross-modal Matching Task
            if args.mat_weight:
                mat_score, mt = self.Mat_fc(f_t_hat, f_wl_avr, ended)  # (64,) (64,)
                mat_loss += self.mat_criterion(mat_score, mt)
                # for f_idx, f in enumerate(ended):
                #     if (f == False):
                #         mat_loss[f_idx] += self.mat_criterion(mat_score[f_idx], mt[f_idx])

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = logit.max(1)        # student forcing - argmax
                log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))   # Gather the log_prob for each batch
            elif self.feedback == 'sample':
                probs = F.softmax(logit, 1)    # sampling an action from model #(64,9)
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())      # For log
                entropys.append(c.entropy())                                # For optimization
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id == args.ignoreid or next_id == (candidate_leng[i]-1):    # The last action is <end>
                    cpu_a_t[i] = -1             # Change the <end> and ignore action to -1

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj,t)
            obs = np.array(self.env._get_obs())
            perm_obs = obs[perm_idx]                    # Perm the obs for the resu

            # Calculate the mask and reward
            dist = np.zeros(batch_size, np.float32)
            reward = np.zeros(batch_size, np.float32)
            mask = np.ones(batch_size, np.float32)
            for i, ob in enumerate(perm_obs):
                dist[i] = ob['distance']
                if ended[i]:            # If the action is already finished BEFORE THIS ACTION.
                    reward[i] = 0.
                    mask[i] = 0.
                else:       # Calculate the reward
                    traj_len[i] += 1
                    action_idx = cpu_a_t[i]
                    if action_idx == -1:        # If the action now is end
                        if dist[i] < 3:         # Correct
                            reward[i] = 2.
                        else:                   # Incorrect
                            reward[i] = -2.
                    else:                       # The action is not end
                        reward[i] = - (dist[i] - last_dist[i])      # Change of distance
                        if reward[i] > 0:                           # Quantification
                            reward[i] = 1
                        elif reward[i] < 0:
                            reward[i] = -1
                        else:
                            raise NameError("The action doesn't change the move")
            rewards.append(reward)
            masks.append(mask)
            last_dist[:] = dist

            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break

        if train_ml:
            ml_loss = ml_loss*train_ml/batch_size
            self.logs['step'].append(t)
            self.logs['ml_loss'].append(ml_loss.detach())
            self.loss += ml_loss

            # Trajectory Retelling Task
            if args.spe_weight or val_spe:
                f_oT_til = torch.stack(f_oT_til, dim=1)
                decode_mask = [torch.tensor(mask) for mask in masks]
                decode_mask = (1 - torch.stack(decode_mask, dim=1)).bool().cuda()  # different definition about mask
                # ctx_mask = utils.length2mask(traj_len)
                h_o = f_ot_til.unsqueeze(0)
                c_o = c1.unsqueeze(0)
                insts = seq
                spe_pre, _, _ = self.Spe_decoder(insts, f_oT_til, decode_mask, h_o,
                                                 c_o)  # input:(64,80) (64,T,512) (64,T) (1,64,512) (1,64,512) output:(64,80,992)
                # # Because the softmax_loss only allow dim-1 to be logit,
                # # So permute the output (batch_size, length, logit) --> (batch_size, logit, length)
                spe_pre = spe_pre.permute(0, 2, 1).contiguous()  # (64,992,80)
                spe_loss = self.spe_criterion(
                    input=spe_pre[:, :, :-1],  # -1 for aligning
                    target=insts[:, 1:]  # "1:" to ignore the word <BOS>
                )
                spe_loss *= args.spe_weight*args.ml_spe
                self.logs['spe_loss'].append(spe_loss.detach())
                self.auxloss += spe_loss
                self.loss += spe_loss

                # Evaluation
                if val_spe:
                    correct_num = torch.sum((insts[:, 1:] == torch.argmax(spe_pre[:, :, :-1], dim=1))) \
                        .cpu().numpy()
                    all_num = insts.size(0) * (insts.size(1) - 1)
                    self.acc += correct_num / all_num
                    self.acc_num += 1

        if train_rl:
            # Last action in A2C
            # input_a_t, f_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)
            ObjFeature_mask = None
            sparseObj = None
            denseObj = None
            f_t = None
            Obj_leng = None
            if args.sparseObj and (not args.denseObj):
                if not args.catRN:
                    input_a_t, sparseObj, Obj_leng, c_t, candidate_leng = self.get_input_feat(perm_obs)
                else:
                    input_a_t, sparseObj, Obj_leng, f_t, c_t, candidate_leng = self.get_input_feat(perm_obs)
            elif args.denseObj and (not args.sparseObj):
                if not args.catRN:
                    input_a_t, denseObj, Obj_leng, c_t, candidate_leng = self.get_input_feat(perm_obs)
                else:
                    input_a_t, denseObj, Obj_leng, f_t, c_t, candidate_leng = self.get_input_feat(perm_obs)
            elif args.denseObj and args.sparseObj:
                if not args.catRN:
                    input_a_t, sparseObj, denseObj, Obj_leng, c_t, candidate_leng = self.get_input_feat(perm_obs)
                else:
                    input_a_t, sparseObj, denseObj, Obj_leng, f_t, c_t, candidate_leng = self.get_input_feat(perm_obs)
            else:
                input_a_t, f_t, c_t, candidate_leng = self.get_input_feat(perm_obs)
            if Obj_leng is not None:
                ObjFeature_mask = utils.length2mask(Obj_leng)

            if speaker is not None:  # Apply the env drop mask to the feat
                c_t[..., :-args.angle_feat_size] *= noise
                f_t[..., :-args.angle_feat_size] *= noise



            last_h_, _, _, _ = self.decoder(input_a_t, c_t,
                                               h_last_t, c1,
                                               f_wl_til, seq_mask,feature=f_t,
                                               sparseObj=sparseObj,denseObj=denseObj,
                                               ObjFeature_mask=ObjFeature_mask,already_dropfeat=(speaker is not None))#input: (64,128)(64,36,2176)(64,11,2176)(64,512)(64,512)(64,80,512)(64,80)
                                                                #output:(64,512)

            rl_loss = 0.

            # NOW, A2C!!!
            # Calculate the final discounted reward
            last_value__ = self.critic(last_h_).detach()    # The value esti of the last state, remove the grad for safety (64,)
            discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero (64,)
            for i in range(batch_size):
                if not ended[i]:        # If the action is not ended, use the value function as the last reward
                    discount_reward[i] = last_value__[i]

            length = len(rewards)            #21
            total = 0
            for x in range(length-1, -1, -1):
                discount_reward = discount_reward * args.gamma + rewards[x]   # If it ended, the reward will be 0 #(64)
                mask_ = Variable(torch.from_numpy(masks[x]), requires_grad=False).cuda()         #(64,)
                clip_reward = discount_reward.copy()                                             #
                r_ = Variable(torch.from_numpy(clip_reward), requires_grad=False).cuda()         #(64)
                v_ = self.critic(hidden_states[x])                                               #(64)
                a_ = (r_ - v_).detach()

                # r_: The higher, the better. -ln(p(action)) * (discount_reward - value)
                rl_loss += (-policy_log_probs[x] * a_ * mask_).sum() # input:(21,64)[t,:]*(64)
                rl_loss += (((r_ - v_) ** 2) * mask_).sum() * 0.5*args.rl_weight     # 1/2 L2 loss
                if self.feedback == 'sample':
                    rl_loss += (- 0.01 * entropys[x] * mask_).sum()
                self.logs['critic_loss'].append((((r_ - v_) ** 2) * mask_).sum().item())
                total = total + np.sum(masks[x])
            self.logs['total'].append(total)

            # Normalize the loss function
            if args.normalize_loss == 'total':
                rl_loss /= total
            elif args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert args.normalize_loss == 'none'

            self.logs['rl_loss'].append(rl_loss.detach())
            self.loss += rl_loss

            # Trajectory Retelling Task
            if args.spe_weight or val_spe:
                f_oT_til = torch.stack(f_oT_til, dim=1)
                decode_mask = [torch.tensor(mask) for mask in masks]
                decode_mask = (1 - torch.stack(decode_mask, dim=1)).bool().cuda()  # different definition about mask
                # ctx_mask = utils.length2mask(traj_len)
                h_o = f_ot_til.unsqueeze(0)
                c_o = c1.unsqueeze(0)
                insts = seq
                spe_pre, _, _ = self.Spe_decoder(insts, f_oT_til, decode_mask, h_o,
                                                 c_o)  # input:(64,80) (64,T,512) (64,T) (1,64,512) (1,64,512) output:(64,80,992)
                # # Because the softmax_loss only allow dim-1 to be logit,
                # # So permute the output (batch_size, length, logit) --> (batch_size, logit, length)
                spe_pre = spe_pre.permute(0, 2, 1).contiguous()  # (64,992,80)
                spe_loss = self.spe_criterion(
                    input=spe_pre[:, :, :-1],  # -1 for aligning
                    target=insts[:, 1:]  # "1:" to ignore the word <BOS>
                )
                spe_loss *= args.spe_weight * args.rl_spe
                self.logs['spe_loss'].append(spe_loss.detach())
                self.auxloss += spe_loss
                self.loss += spe_loss

                # Evaluation
                if val_spe:
                    correct_num = torch.sum((insts[:, 1:] == torch.argmax(spe_pre[:, :, :-1], dim=1))) \
                        .cpu().numpy()
                    all_num = insts.size(0) * (insts.size(1) - 1)
                    self.acc += correct_num / all_num
                    self.acc_num += 1

        # Angle Prediction Task
        if args.ang_weight:
            ang_loss *= args.ang_weight
            self.logs['ang_loss'].append(ang_loss.detach())
            self.auxloss += ang_loss
            self.loss += ang_loss

        # Feature Prediction Task
        if args.fea_weight:
            fea_loss = fea_loss * args.fea_weight
            self.auxloss += fea_loss
            self.loss += fea_loss
            self.logs['fea_loss'].append(fea_loss.detach())

        # Cross-modal Matching Task
        if args.mat_weight:
            # mat_loss = torch.sum(mat_loss / torch.from_numpy(traj_len).float().cuda()) * args.mat_weight
            mat_loss *= args.mat_weight
            self.logs['mat_loss'].append(mat_loss.detach())
            self.auxloss += mat_loss
            self.loss += mat_loss

        # Progress Estimation Task
        if args.pro_weight:
            # pro_score_T = pro_score_T[:, 0:t + 1]
            pro_score_T = torch.stack(pro_score_T, dim=1).squeeze(2)
            for len_id, l in enumerate(traj_len):
                for i in range(int(l)):
                    r_t[len_id, i] = float(i + 1) / l
            # for i in range(batch_size):
            # pro_loss += self.pro_criterion(pro_score_T[i, :], r_t[i, :]) / traj_len[i]  # criterion,input: (1,35)
            r_t = r_t[:, 0:t + 1]
            pro_loss += self.pro_criterion(pro_score_T, r_t)
            pro_loss *= args.pro_weight
            self.logs['pro_loss'].append(pro_loss.detach())
            self.auxloss += pro_loss
            self.loss += pro_loss

        # self.logs['auxloss'].append(self.auxloss.detach())
        self.logs['loss'].append(self.loss)
        return traj

    def train(self, n_iters, feedback='teacher', speaker=None, **kwargs):
        ''' Train for a given number of iterations '''
        self.feedback = feedback

        self.encoder.train()
        self.decoder.train()
        self.critic.train()

        if args.spe_weight:
            self.Spe_decoder.train()
        if args.pro_weight:
            self.Pro_fc.train()
        if args.ang_weight:
            self.Ang_fc.train()
        if args.mat_weight:
            self.Mat_fc.train()
        if args.fea_weight:
            self.Fea_fc.train()

        self.losses = []
        iter_time = time.time()
        for iter in range(1, n_iters + 1):

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            if args.spe_weight:
                self.Spe_decoder_optimizer.zero_grad()
            if args.pro_weight:
                self.Pro_fc_optimizer.zero_grad()
            if args.ang_weight:
                self.Ang_fc_optimizer.zero_grad()
            if args.mat_weight:
                self.Mat_fc_optimizer.zero_grad()
            if args.fea_weight:
                self.Fea_fc_optimizer.zero_grad()
            self.loss = 0
            self.auxloss = 0
            if feedback == 'teacher':
                self.feedback = 'teacher'
                self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
            elif feedback == 'sample':
                if args.ml_weight != 0:
                    self.feedback = 'teacher'
                    start = time.time()
                    self.rollout(train_ml=args.ml_weight, train_rl=False, speaker=speaker, **kwargs)
                    # print('Train in ML use %0.4f seconds' % (time.time() - start))
                self.feedback = 'sample'
                start = time.time()
                self.rollout(train_ml=None, train_rl=True, **kwargs)
                # print('Train in RL use %0.4f seconds' % (time.time() - start))
            else:
                assert False

            self.loss.backward()

            torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
            torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 40.)
            if args.spe_weight:
                torch.nn.utils.clip_grad_norm(self.Spe_decoder.parameters(), 40.)

            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
            self.critic_optimizer.step()

            if args.spe_weight:
                self.Spe_decoder_optimizer.step()
            if args.pro_weight:
                self.Pro_fc_optimizer.step()
            if args.ang_weight:
                self.Ang_fc_optimizer.step()
            if args.mat_weight:
                self.Mat_fc_optimizer.step()
            if args.fea_weight:
                self.Fea_fc_optimizer.step()

        print('Train 100 iter for %0.4f seconds' % (time.time() - iter_time))

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        if use_dropout:
            self.encoder.train()
            self.decoder.train()
            self.critic.train()
            if args.spe_weight:
                self.Spe_decoder.train()
            if args.pro_weight:
                self.Pro_fc.train()
            if args.ang_weight:
                self.Ang_fc.train()
            if args.mat_weight:
                self.Mat_fc.train()
            if args.fea_weight:
                self.Fea_fc.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
            self.critic.eval()
            if args.spe_weight:
                self.Spe_decoder.eval()
            if args.pro_weight:
                self.Pro_fc.eval()
            if args.ang_weight:
                self.Ang_fc.eval()
            if args.mat_weight:
                self.Mat_fc.eval()
            if args.fea_weight:
                self.Fea_fc.eval()
        super(Seq2SeqAgent, self).test(iters)

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                     ("decoder", self.decoder, self.decoder_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        if args.spe_weight:
            all_tuple.append(("Spe_decoder", self.Spe_decoder, self.Spe_decoder_optimizer))
        if args.pro_weight:
            all_tuple.append(("Pro_fc", self.Pro_fc, self.Pro_fc_optimizer))
        if args.mat_weight:
            all_tuple.append(("Mat_fc", self.Mat_fc, self.Mat_fc_optimizer))
        if args.ang_weight:
            all_tuple.append(("Ang_fc", self.Ang_fc, self.Ang_fc_optimizer))
        if args.fea_weight:
            all_tuple.append(("Fea_fc", self.Fea_fc, self.Fea_fc_optimizer))
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)
        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            if args.loadOptim:
                optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                     ("decoder", self.decoder, self.decoder_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        if args.spe_weight:
            all_tuple.append(("Spe_decoder", self.Spe_decoder, self.Spe_decoder_optimizer))
        if args.pro_weight:
            all_tuple.append(("Pro_fc", self.Pro_fc, self.Pro_fc_optimizer))
        if args.mat_weight:
            all_tuple.append(("Mat_fc", self.Mat_fc, self.Mat_fc_optimizer))
        if args.ang_weight:
            all_tuple.append(("Ang_fc", self.Ang_fc, self.Ang_fc_optimizer))
        if args.fea_weight:
            all_tuple.append(("Fea_fc", self.Fea_fc, self.Fea_fc_optimizer))
        for param in all_tuple:
            recover_state(*param)
        return states['encoder']['epoch'] - 1

    def _dijkstra(self):
        """
        The dijkstra algorithm.
        Was called beam search to be consistent with existing work.
        But it actually finds the Exact K paths with smallest listener log_prob.
        :return:
        [{
            "scan": XXX
            "instr_id":XXX,
            'instr_encoding": XXX
            'dijk_path': [v1, v2, ..., vn]      (The path used for find all the candidates)
            "paths": {
                    "trajectory": [viewpoint_id1, viewpoint_id2, ..., ],
                    "action": [act_1, act_2, ..., ],
                    "listener_scores": [log_prob_act1, log_prob_act2, ..., ],
                    "visual_feature": [(f1_step1, f2_step2, ...), (f1_step2, f2_step2, ...)
            }
        }]
        """
        def make_state_id(viewpoint, action):     # Make state id
            return "%s_%s" % (viewpoint, str(action))
        def decompose_state_id(state_id):     # Make state id
            viewpoint, action = state_id.split("_")
            action = int(action)
            return viewpoint, action

        # Get first obs
        obs = self.env._get_obs()

        # Prepare the state id
        batch_size = len(obs)
        results = [{"scan": ob['scan'],
                    "instr_id": ob['instr_id'],
                    "instr_encoding": ob["instr_encoding"],
                    "dijk_path": [ob['viewpoint']],
                    "paths": []} for ob in obs]

        # Encoder
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        recover_idx = np.zeros_like(perm_idx)
        for i, idx in enumerate(perm_idx):
            recover_idx[idx] = i
        ctx, h_t, c_t = self.encoder(seq, seq_lengths)
        ctx, h_t, c_t, ctx_mask = ctx[recover_idx], h_t[recover_idx], c_t[recover_idx], seq_mask[recover_idx]    # Recover the original order

        # Dijk Graph States:
        id2state = [
            {make_state_id(ob['viewpoint'], -95):
                 {"next_viewpoint": ob['viewpoint'],
                  "running_state": (h_t[i], h_t[i], c_t[i]),
                  "location": (ob['viewpoint'], ob['heading'], ob['elevation']),
                  "feature": None,
                  "from_state_id": None,
                  "score": 0,
                  "scores": [],
                  "actions": [],
                  }
             }
            for i, ob in enumerate(obs)
        ]    # -95 is the start point
        visited = [set() for _ in range(batch_size)]
        finished = [set() for _ in range(batch_size)]
        graphs = [utils.FloydGraph() for _ in range(batch_size)]        # For the navigation path
        ended = np.array([False] * batch_size)

        # Dijk Algorithm
        for _ in range(300):
            # Get the state with smallest score for each batch
            # If the batch is not ended, find the smallest item.
            # Else use a random item from the dict  (It always exists)
            smallest_idXstate = [
                max(((state_id, state) for state_id, state in id2state[i].items() if state_id not in visited[i]),
                    key=lambda item: item[1]['score'])
                if not ended[i]
                else
                next(iter(id2state[i].items()))
                for i in range(batch_size)
            ]

            # Set the visited and the end seqs
            for i, (state_id, state) in enumerate(smallest_idXstate):
                assert (ended[i]) or (state_id not in visited[i])
                if not ended[i]:
                    viewpoint, action = decompose_state_id(state_id)
                    visited[i].add(state_id)
                    if action == -1:
                        finished[i].add(state_id)
                        if len(finished[i]) >= args.candidates:     # Get enough candidates
                            ended[i] = True

            # Gather the running state in the batch
            h_ts, h1s, c_ts = zip(*(idXstate[1]['running_state'] for idXstate in smallest_idXstate))
            h_t, h1, c_t = torch.stack(h_ts), torch.stack(h1s), torch.stack(c_ts)

            # Recover the env and gather the feature
            for i, (state_id, state) in enumerate(smallest_idXstate):
                next_viewpoint = state['next_viewpoint']
                scan = results[i]['scan']
                from_viewpoint, heading, elevation = state['location']
                self.env.env.sims[i].newEpisode(scan, next_viewpoint, heading, elevation) # Heading, elevation is not used in panoramic
            obs = self.env._get_obs()

            # Update the floyd graph
            # Only used to shorten the navigation length
            # Will not effect the result
            for i, ob in enumerate(obs):
                viewpoint = ob['viewpoint']
                if not graphs[i].visited(viewpoint):    # Update the Graph
                    for c in ob['candidate']:
                        next_viewpoint = c['viewpointId']
                        dis = self.env.distances[ob['scan']][viewpoint][next_viewpoint]
                        graphs[i].add_edge(viewpoint, next_viewpoint, dis)
                    graphs[i].update(viewpoint)
                results[i]['dijk_path'].extend(graphs[i].path(results[i]['dijk_path'][-1], viewpoint))

            input_a_t, f_t, c_t, candidate_leng = self.get_input_feat(obs)

            # Run one decoding step
            h_t, c_t, alpha, logit, h1 = self.decoder(input_a_t, f_t, c_t,
                                                      h_t, h1, c_t,
                                                      ctx, ctx_mask,
                                                      False)

            # Update the dijk graph's states with the newly visited viewpoint
            candidate_mask = utils.length2mask(candidate_leng)
            logit.masked_fill_(candidate_mask.bool(), -float('inf'))
            log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
            _, max_act = log_probs.max(1)

            for i, ob in enumerate(obs):
                current_viewpoint = ob['viewpoint']
                candidate = ob['candidate']
                current_state_id, current_state = smallest_idXstate[i]
                old_viewpoint, from_action = decompose_state_id(current_state_id)
                assert ob['viewpoint'] == current_state['next_viewpoint']
                if from_action == -1 or ended[i]:       # If the action is <end> or the batch is ended, skip it
                    continue
                for j in range(len(ob['candidate']) + 1):               # +1 to include the <end> action
                    # score + log_prob[action]
                    modified_log_prob = log_probs[i][j].detach().cpu().item()
                    new_score = current_state['score'] + modified_log_prob
                    if j < len(candidate):                        # A normal action
                        next_id = make_state_id(current_viewpoint, j)
                        next_viewpoint = candidate[j]['viewpointId']
                        trg_point = candidate[j]['pointId']
                        heading = (trg_point % 12) * math.pi / 6
                        elevation = (trg_point // 12 - 1) * math.pi / 6
                        location = (next_viewpoint, heading, elevation)
                    else:                                                 # The end action
                        next_id = make_state_id(current_viewpoint, -1)    # action is -1
                        next_viewpoint = current_viewpoint                # next viewpoint is still here
                        location = (current_viewpoint, ob['heading'], ob['elevation'])

                    if next_id not in id2state[i] or new_score > id2state[i][next_id]['score']:
                        id2state[i][next_id] = {
                            "next_viewpoint": next_viewpoint,
                            "location": location,
                            "running_state": (h_t[i], h1[i], c_t[i]),
                            "from_state_id": current_state_id,
                            "feature": (f_t[i].detach().cpu(), c_t[i][j].detach().cpu()),
                            "score": new_score,
                            "scores": current_state['scores'] + [modified_log_prob],
                            "actions": current_state['actions'] + [len(candidate)+1],
                        }

            # The active state is zero after the updating, then setting the ended to True
            for i in range(batch_size):
                if len(visited[i]) == len(id2state[i]):     # It's the last active state
                    ended[i] = True

            # End?
            if ended.all():
                break

        # Move back to the start point
        for i in range(batch_size):
            results[i]['dijk_path'].extend(graphs[i].path(results[i]['dijk_path'][-1], results[i]['dijk_path'][0]))
        """
            "paths": {
                "trajectory": [viewpoint_id1, viewpoint_id2, ..., ],
                "action": [act_1, act_2, ..., ],
                "listener_scores": [log_prob_act1, log_prob_act2, ..., ],
                "visual_feature": [(f1_step1, f2_step2, ...), (f1_step2, f2_step2, ...)
            }
        """
        # Gather the Path
        for i, result in enumerate(results):
            assert len(finished[i]) <= args.candidates
            for state_id in finished[i]:
                path_info = {
                    "trajectory": [],
                    "action": [],
                    "listener_scores": id2state[i][state_id]['scores'],
                    "listener_actions": id2state[i][state_id]['actions'],
                    "visual_feature": []
                }
                viewpoint, action = decompose_state_id(state_id)
                while action != -95:
                    state = id2state[i][state_id]
                    path_info['trajectory'].append(state['location'])
                    path_info['action'].append(action)
                    path_info['visual_feature'].append(state['feature'])
                    state_id = id2state[i][state_id]['from_state_id']
                    viewpoint, action = decompose_state_id(state_id)
                state = id2state[i][state_id]
                path_info['trajectory'].append(state['location'])
                for need_reverse_key in ["trajectory", "action", "visual_feature"]:
                    path_info[need_reverse_key] = path_info[need_reverse_key][::-1]
                result['paths'].append(path_info)

        return results

    def beam_search(self, speaker):
        """
        :param speaker: The speaker to be used in searching.
        :return:
        {
            "scan": XXX
            "instr_id":XXX,
            "instr_encoding": XXX
            "dijk_path": [v1, v2, ...., vn]
            "paths": [{
                "trajectory": [viewoint_id0, viewpoint_id1, viewpoint_id2, ..., ],
                "action": [act_1, act_2, ..., ],
                "listener_scores": [log_prob_act1, log_prob_act2, ..., ],
                "speaker_scores": [log_prob_word1, log_prob_word2, ..., ],
            }]
        }
        """
        self.env.reset()
        results = self._dijkstra()
        """
        return from self._dijkstra()
        [{
            "scan": XXX
            "instr_id":XXX,
            "instr_encoding": XXX
            "dijk_path": [v1, v2, ...., vn]
            "paths": [{
                    "trajectory": [viewoint_id0, viewpoint_id1, viewpoint_id2, ..., ],
                    "action": [act_1, act_2, ..., ],
                    "listener_scores": [log_prob_act1, log_prob_act2, ..., ],
                    "visual_feature": [(f1_step1, f2_step2, ...), (f1_step2, f2_step2, ...)
            }]
        }]
        """

        # Compute the speaker scores:
        for result in results:
            lengths = []
            num_paths = len(result['paths'])
            for path in result['paths']:
                assert len(path['trajectory']) == (len(path['visual_feature']) + 1)
                lengths.append(len(path['visual_feature']))
            max_len = max(lengths)
            img_feats = torch.zeros(num_paths, max_len, 36, self.feature_size + args.angle_feat_size)
            can_feats = torch.zeros(num_paths, max_len, self.feature_size + args.angle_feat_size)
            for j, path in enumerate(result['paths']):
                for k, feat in enumerate(path['visual_feature']):
                    img_feat, can_feat = feat
                    img_feats[j][k] = img_feat
                    can_feats[j][k] = can_feat
            img_feats, can_feats = img_feats.cuda(), can_feats.cuda()
            features = ((img_feats, can_feats), lengths)
            insts = np.array([result['instr_encoding'] for _ in range(num_paths)])
            seq_lengths = np.argmax(insts == self.tok.word_to_index['<EOS>'], axis=1)   # len(seq + 'BOS') == len(seq + 'EOS')
            insts = torch.from_numpy(insts).cuda()
            speaker_scores = speaker.teacher_forcing(train=True, features=features, insts=insts, for_listener=True)
            for j, path in enumerate(result['paths']):
                path.pop("visual_feature")
                path['speaker_scores'] = -speaker_scores[j].detach().cpu().numpy()[:seq_lengths[j]]
        return results

    def beam_search_test(self, speaker):
        self.encoder.eval()
        self.decoder.eval()
        self.critic.eval()

        looped = False
        self.results = {}
        while True:
            for traj in self.beam_search(speaker):
                if traj['instr_id'] in self.results:
                    looped = True
                else:
                    self.results[traj['instr_id']] = traj
            if looped:
                break

    def zero_grad(self):
        self.loss = 0.
        self.auxloss=0.
        self.losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()

        for model, optimizer in zip(self.aux_models, self.aux_optimizer):
            model.train()
            optimizer.zero_grad()

    def accumulate_gradient(self, feedback='teacher', **kwargs):
        if feedback == 'teacher':
            self.feedback = 'teacher'
            self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
        elif feedback == 'sample':
            self.feedback = 'teacher'
            self.rollout(train_ml=args.ml_weight, train_rl=False, **kwargs)
            self.feedback = 'sample'
            self.rollout(train_ml=None, train_rl=True, **kwargs)
        else:
            assert False

    def optim_step(self):
        self.loss.backward()

        torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
        torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 40.)
        if args.spe_weight:
            self.Spe_decoder_optimizer.step()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.critic_optimizer.step()
        if args.pro_weight:
            self.Pro_fc_optimizer.step()
        if args.ang_weight:
            self.Ang_fc_optimizer.step()
        if args.mat_weight:
            self.Mat_fc_optimizer.step()
        if args.fea_weight:
            self.Fea_fc_optimizer.step()

    def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None, step=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        def take_action(i, idx, name):
            if type(name) is int:       # Go to the next view
                self.env.env.sims[idx].makeAction(name, 0, 0)
            else:                       # Adjust
                self.env.env.sims[idx].makeAction(*self.env_actions[name])
            state = self.env.env.sims[idx].getState()
            if traj is not None:
                traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))
        if perm_idx is None:
            perm_idx = range(len(perm_obs))
        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12   # The point idx started from 0
                trg_level = (trg_point ) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, idx, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, idx, 'down')
                    src_level -= 1
                while self.env.env.sims[idx].getState().viewIndex != trg_point:    # Turn right until the target
                    take_action(i, idx, 'right')
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[idx].getState().navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, idx, select_candidate['idx'])

