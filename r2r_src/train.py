import sys
sys.path.insert(0, '/R2R-Aux/build')
import torch

import os
import time
import json
import numpy as np
from collections import defaultdict
from speaker import Speaker

from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince, read_img_features, read_obj_features, get_sync_dir
import utils
from env import R2RBatch
from agent import Seq2SeqAgent
from eval import Evaluation
from param import args

import tarfile
from polyaxon_client.tracking import get_outputs_path

import warnings
warnings.filterwarnings("ignore")


from tensorboardX import SummaryWriter

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

output_dir = get_outputs_path()
# log_dir = 'snap/%s' % args.name
log_dir = os.path.join(output_dir, "snap", args.name)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'

IMAGENET_FEATURES = 'lyx/img_features/ResNet-152-imagenet.tsv'
PLACE365_FEATURES = 'lyx/img_features/ResNet-152-places365.tsv'

SPARSE_OBJ_FEATURES = 'lyx/obj_features/%s/panorama_objs_Features_nms_%s.npy'%(args.objdir, args.objdir)
DENSE_OBJ_FEATURES1 = 'lyx/obj_features/%s/panorama_objs_DenseFeatures_nms1_%s.npy'%(args.objdir, args.objdir)
DENSE_OBJ_FEATURES2 = 'lyx/obj_features/%s/panorama_objs_DenseFeatures_nms2_%s.npy'%(args.objdir, args.objdir)



if args.features == 'imagenet':
    features = get_sync_dir(IMAGENET_FEATURES)
    sparse_obj_feat = get_sync_dir(SPARSE_OBJ_FEATURES)
    dense_obj_feat1 = get_sync_dir(DENSE_OBJ_FEATURES1)
    dense_obj_feat2 = get_sync_dir(DENSE_OBJ_FEATURES2)


if args.fast_train:
    name, ext = os.path.splitext(features)
    features = name + "-fast" + ext

feedback_method = args.feedback # teacher or sample

print(args)


def train_speaker(train_env, tok, n_iters, log_every=500, val_envs={}):
    writer = SummaryWriter(logdir=log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)
    speaker = Speaker(train_env, listner, tok)

    if args.load is not None:
        print("LOAD THE Speaker from %s" % args.load)
        speaker.load(os.path.join(args.load))

    if args.fast_train:
        log_every = 40

    best_bleu = defaultdict(lambda: 0)
    best_loss = defaultdict(lambda: 1232)
    for idx in range(0, n_iters, log_every):
        interval = min(log_every, n_iters - idx)

        # Train for log_every interval
        speaker.env = train_env
        speaker.train(interval)   # Train interval iters

        print()
        print("Iter: %d" % idx)

        # Evaluation
        for env_name, (env, evaluator) in val_envs.items():
            if 'train' in env_name: # Ignore the large training set for the efficiency
                continue

            print("............ Evaluating %s ............." % env_name)
            speaker.env = env
            path2inst, loss, word_accu, sent_accu = speaker.valid()
            path_id = next(iter(path2inst.keys()))
            print('path_id:', path_id)
            print("Inference: ", tok.decode_sentence(path2inst[path_id]))
            print("GT: ", evaluator.gt[str(path_id)]['instructions'])
            bleu_score, precisions = evaluator.bleu_score(path2inst)

            # Tensorboard log
            writer.add_scalar("bleu/%s" % (env_name), bleu_score, idx)
            writer.add_scalar("loss/%s" % (env_name), loss, idx)
            writer.add_scalar("word_accu/%s" % (env_name), word_accu, idx)
            writer.add_scalar("sent_accu/%s" % (env_name), sent_accu, idx)
            writer.add_scalar("bleu4/%s" % (env_name), precisions[3], idx)

            # Save the model according to the bleu score
            if bleu_score > best_bleu[env_name]:
                best_bleu[env_name] = bleu_score
                print('Save the model with %s BEST env bleu %0.4f' % (env_name, bleu_score))
                speaker.save(idx, os.path.join(log_dir, 'state_dict', 'best_%s_bleu' % env_name))

            if loss < best_loss[env_name]:
                best_loss[env_name] = loss
                print('Save the model with %s BEST env loss %0.4f' % (env_name, loss))
                speaker.save(idx, os.path.join(log_dir, 'state_dict', 'best_%s_loss' % env_name))

            # Screen print out
            print("Bleu 1: %0.4f Bleu 2: %0.4f, Bleu 3 :%0.4f,  Bleu 4: %0.4f" % tuple(precisions))


def train(train_env, tok, n_iters, log_every=100, val_envs={}, aug_env=None):
    writer = SummaryWriter(logdir=log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)
    speaker = None
    if args.self_train:
        speaker = Speaker(train_env, listner, tok)
        if args.speaker is not None:
            print("Load the speaker from %s." % args.speaker)
            speaker.load(args.speaker)
            # listner.load_speaker(speaker)

    start_iter = 0
    if args.load is not None:
        print("LOAD THE listener from %s" % args.load)
        start_iter = listner.load(os.path.join(args.load))

    start = time.time()

    best_val = {'val_seen': {"accu": 0., "state":"", 'update':False},
                'val_unseen': {"accu": 0., "state":"", 'update':False}}
    best_spl = {'val_seen': {"spl":0, "state":"", 'update':False},
                   'val_unseen': {"spl": 0., "state":"", 'update':False}}
    if args.fast_train:
        log_every = 40
    for idx in range(start_iter, start_iter+n_iters, log_every):
        listner.logs = defaultdict(list)
        interval = min(log_every, start_iter+n_iters-idx)
        iter = idx + interval

        # Train for log_every interval
        if aug_env is None:     # The default training process
            listner.env = train_env
            listner.train(interval, feedback=feedback_method)   # Train interval iters
        else:
            if args.accumulate_grad:
                for _ in range(interval // 2):
                    listner.zero_grad()
                    listner.env = train_env

                    # Train with GT data
                    args.ml_weight = 0.2
                    listner.accumulate_gradient(feedback_method)
                    listner.env = aug_env

                    # Train with Back Translation
                    args.ml_weight = 0.6        # Sem-Configuration
                    listner.accumulate_gradient(feedback_method, speaker=speaker)
                    listner.optim_step()
            else:
                for _ in range(interval // 2):
                    # Train with GT data
                    listner.env = train_env
                    args.ml_weight = 0.2
                    listner.train(1, feedback=feedback_method)

                    # Train with Back Translation
                    listner.env = aug_env
                    args.ml_weight = 0.6
                    listner.train(1, feedback=feedback_method, speaker=speaker)

        # Log the training stats to tensorboard
        total = max(sum(listner.logs['total']), 1)
        length = max(len(listner.logs['rl_loss']), 1)
        critic_loss = sum(listner.logs['critic_loss'])/total
        rl_loss = sum(listner.logs['rl_loss']) / total #/ length / args.batchSize
        entropy = sum(listner.logs['entropy']) / total #/ length / args.batchSize
        ml_loss = sum(listner.logs['ml_loss']) / total
        ang_loss = sum(listner.logs['ang_loss']) / total
        fea_loss = sum(listner.logs['fea_loss']) / total
        spe_loss = sum(listner.logs['spe_loss']) / total
        mat_loss = sum(listner.logs['mat_loss']) / total
        pro_loss = sum(listner.logs['pro_loss']) / total
        aux_loss = sum(listner.logs['auxloss'])/total/2
        sum_loss = rl_loss+ml_loss+aux_loss
        # sum_loss = sum(listner.logs['loss'])/total


        writer.add_scalar("loss/ml_loss", ml_loss, idx)
        writer.add_scalar("loss/fea_loss", fea_loss, idx)
        writer.add_scalar("loss/ang_loss", ang_loss, idx)
        writer.add_scalar("loss/spe_loss", spe_loss, idx)
        writer.add_scalar("loss/mat_loss", mat_loss, idx)
        writer.add_scalar("loss/pro_loss", pro_loss, idx)
        writer.add_scalar("loss/aux_loss", aux_loss, idx)
        writer.add_scalar("loss/sum_loss", sum_loss, idx)
        writer.add_scalar("loss/rl_loss", rl_loss, idx)
        writer.add_scalar("loss/critic_loss", critic_loss, idx)
        writer.add_scalar("policy_entropy", entropy, idx)
        writer.add_scalar("total_actions", total, idx)
        writer.add_scalar("max_length", length, idx)

        print("total_actions", total)
        print("max_length", length)

        # Run validation
        loss_str = ""
        test_time = time.time()
        for env_name, (env, evaluator) in val_envs.items():
            listner.env = env

            # Get validation loss under the same conditions as training
            iters = None if args.fast_train or env_name != 'train' else 20     # 20 * 64 = 1280

            # Get validation distance from goal under test evaluation conditions
            with torch.no_grad():
                listner.test(use_dropout=False, feedback='argmax', iters=iters)
                # listner.test(use_dropout=False, feedback='argmax', iters=iters)
            result= listner.get_results()
            score_summary, _ = evaluator.score(result)
            loss_str += "%s " % env_name
            for metric,val in score_summary.items():
                if metric in ['success_rate']:
                    writer.add_scalar("%s/accuracy" % env_name, val, idx)
                    if env_name in best_val:
                        if val > best_val[env_name]['accu']:
                            best_val[env_name]['accu'] = val
                            best_val[env_name]['update'] = True
                if metric in ['nav_error']:
                    writer.add_scalar("%s/nav_error" % env_name, val, idx)
                if metric in ['oracle_error']:
                    writer.add_scalar("%s/oracle_error" % env_name, val, idx)
                if metric in ['spl']:
                    writer.add_scalar("%s/spl" % env_name, val, idx)
                    if env_name in best_spl:
                        if val > best_spl[env_name]['spl']:
                            best_spl[env_name]['spl'] = val
                            best_val[env_name]['update'] = True
                if metric in ['oracle_rate']:
                    writer.add_scalar("%s/oracle_rate" % env_name, val, idx)
                loss_str += ', %s: %.3f' % (metric, val)
            loss_str += '\n'
        print("Test use %0.4f seconds" % (time.time() - test_time))
        loss_str += 'ml_weight: %.1f' % (args.ml_weight)
        loss_str += ', rl_weight: %.1f' % (args.rl_weight)
        loss_str += ', spe_weight: %.1f' % (args.spe_weight)
        loss_str += ', pro_weight: %.1f' % (args.pro_weight)
        loss_str += ', mat_weight: %.1f' % (args.mat_weight)
        loss_str += ', ang_weight: %.1f' % (args.ang_weight)
        loss_str += ', fea_weight: %.1f' % (args.fea_weight)
        loss_str += '\n'


        for env_name in best_val:
            if best_val[env_name]['update']:
                best_val[env_name]['state'] = 'Iter %d \n%s' % (iter, loss_str)
                best_val[env_name]['update'] = False
                file_dir = os.path.join(output_dir, "snap", args.name, "state_dict", "best_%s" % (env_name))
                listner.save(idx, file_dir)
                # listner.save(idx, os.path.join("snap", args.name, "state_dict", "best_%s" % (env_name)))

        for env_name in best_spl:
            if best_spl[env_name]['update']:
                best_spl[env_name]['state'] = 'Itere %d \n%s' % (iter, loss_str)
                best_spl[env_name]['update'] = False
                file_dir = os.path.join(output_dir, "snap", args.name, "state_dict", "best_spl_%s" % (env_name))
                listner.save(idx, file_dir)
                # listner.save(idx, os.path.join("snap", args.name, "state_dict", "best_spl_%s" % (env_name)))

        print(('%s (%d %d%%) \n%s' % (timeSince(start, float(iter)/n_iters),
                                             iter, float(iter)/n_iters*100, loss_str)))

        if args.load is not None:
            print("BEST RESULT TILL NOW")
            for env_name in best_val:
                print(env_name, best_val[env_name]['state'])


        if iter % 1000 == 0:
            print("BEST RESULT TILL NOW")
            for env_name in best_val:
                print(env_name, best_val[env_name]['state'])

        if iter % 20000 == 0:
            file_dir = os.path.join(output_dir, "snap", args.name, "state_dict", "Iter_%06d" % (iter))
            listner.save(idx,file_dir)
            # listner.save(idx, os.path.join("snap", args.name, "state_dict", "Iter_%06d" % (iter)))




    listner.save(idx, os.path.join("snap", args.name, "state_dict", "LAST_iter%d" % (idx)))


def valid(train_env, tok, val_envs={}):
    agent = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    print("Loaded the listener model at iter %d from %s" % (agent.load(args.load), args.load))

    for env_name, (env, evaluator) in val_envs.items():
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        with torch.no_grad():
            agent.test(use_dropout=False, feedback='argmax', iters=iters)
        result = agent.get_results()

        if env_name != '':
            score_summary, _ = evaluator.score(result)
            loss_str = "Env name: %s" % env_name
            for metric,val in score_summary.items():
                loss_str += ', %s: %.4f' % (metric, val)
            print(loss_str)

        if args.submit:
            json.dump(
                result,
                open(os.path.join(log_dir, "submit_%s.json" % env_name), 'w'),
                sort_keys=True, indent=4, separators=(',', ': ')
            )


def beam_valid(train_env, tok, val_envs={}):
    listener = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    speaker = Speaker(train_env, listener, tok)
    if args.speaker is not None:
        print("Load the speaker from %s." % args.speaker)
        speaker.load(args.speaker)

    print("Loaded the listener model at iter % d" % listener.load(args.load))

    final_log = ""
    for env_name, (env, evaluator) in val_envs.items():
        listener.logs = defaultdict(list)
        listener.env = env

        listener.beam_search_test(speaker)
        results = listener.results

        def cal_score(x, alpha, avg_speaker, avg_listener):
            speaker_score = sum(x["speaker_scores"]) * alpha
            if avg_speaker:
                speaker_score /= len(x["speaker_scores"])
            # normalizer = sum(math.log(k) for k in x['listener_actions'])
            normalizer = 0.
            listener_score = (sum(x["listener_scores"]) + normalizer) * (1-alpha)
            if avg_listener:
                listener_score /= len(x["listener_scores"])
            return speaker_score + listener_score

        if args.param_search:
            # Search for the best speaker / listener ratio
            interval = 0.01
            logs = []
            for avg_speaker in [False, True]:
                for avg_listener in [False, True]:
                    for alpha in np.arange(0, 1 + interval, interval):
                        result_for_eval = []
                        for key in results:
                            result_for_eval.append({
                                "instr_id": key,
                                "trajectory": max(results[key]['paths'],
                                                  key=lambda x: cal_score(x, alpha, avg_speaker, avg_listener)
                                                  )['trajectory']
                            })
                        score_summary, _ = evaluator.score(result_for_eval)
                        for metric,val in score_summary.items():
                            if metric in ['success_rate']:
                                print("Avg speaker %s, Avg listener %s, For the speaker weight %0.4f, the result is %0.4f" %
                                      (avg_speaker, avg_listener, alpha, val))
                                logs.append((avg_speaker, avg_listener, alpha, val))
            tmp_result = "Env Name %s\n" % (env_name) + \
                    "Avg speaker %s, Avg listener %s, For the speaker weight %0.4f, the result is %0.4f\n" % max(logs, key=lambda x: x[3])
            print(tmp_result)
            # print("Env Name %s" % (env_name))
            # print("Avg speaker %s, Avg listener %s, For the speaker weight %0.4f, the result is %0.4f" %
            #       max(logs, key=lambda x: x[3]))
            final_log += tmp_result
            print()
        else:
            avg_speaker = True
            avg_listener = True
            alpha = args.alpha

            result_for_eval = []
            for key in results:
                result_for_eval.append({
                    "instr_id": key,
                    "trajectory": [(vp, 0, 0) for vp in results[key]['dijk_path']] + \
                                  max(results[key]['paths'],
                                   key=lambda x: cal_score(x, alpha, avg_speaker, avg_listener)
                                  )['trajectory']
                })
            # result_for_eval = utils.add_exploration(result_for_eval)
            score_summary, _ = evaluator.score(result_for_eval)

            if env_name != 'test':
                loss_str = "Env Name: %s" % env_name
                for metric, val in score_summary.items():
                    if metric in ['success_rate']:
                        print("Avg speaker %s, Avg listener %s, For the speaker weight %0.4f, the result is %0.4f" %
                              (avg_speaker, avg_listener, alpha, val))
                    loss_str += ",%s: %0.4f " % (metric, val)
                print(loss_str)
            print()

            if args.submit:
                json.dump(
                    result_for_eval,
                    open(os.path.join(log_dir, "submit_%s.json" % env_name), 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )
    print(final_log)


def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # Check for vocabs
    if not os.path.exists(TRAIN_VOCAB):
        write_vocab(build_vocab(splits=['train']), TRAIN_VOCAB)
    if not os.path.exists(TRAINVAL_VOCAB):
        write_vocab(build_vocab(splits=['train','val_seen','val_unseen']), TRAINVAL_VOCAB)


def train_val():
    ''' Train on the training set, and validate on seen and unseen splits. '''
    # args.fast_train = True
    setup()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)

    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)


    feat_dict = read_img_features(features)

    # load object feature
    obj_s_feat = None
    if args.sparseObj:
        print("Start loading the object sparse feature")
        start = time.time()
        obj_s_feat = np.load(sparse_obj_feat,allow_pickle=True).item()
        print("Finish Loading the object sparse feature from %s in %0.4f seconds" % (sparse_obj_feat, time.time() - start))

    obj_d_feat = None
    if args.denseObj:
        print("Start loading the object dense feature")
        start = time.time()
        obj_d_feat1 = np.load(dense_obj_feat1,allow_pickle=True).item()
        obj_d_feat2 = np.load(dense_obj_feat2,allow_pickle=True).item()
        obj_d_feat = {**obj_d_feat1, **obj_d_feat2}
        print("Finish Loading the dense object dense feature from %s and %s in %0.4f seconds" % (dense_obj_feat1,dense_obj_feat2,
                                                                                                 time.time() - start))

    featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

    train_env = R2RBatch(feat_dict , obj_d_feat=obj_d_feat, obj_s_feat=obj_s_feat, batch_size=args.batchSize, splits=['train'], tokenizer=tok)
    from collections import OrderedDict

    val_env_names = ['val_unseen', 'val_seen']
    if args.submit:
        val_env_names.append('test')
    else:
        pass
        #val_env_names.append('train')

    if not args.beam:
        val_env_names.append("train")

    val_envs = OrderedDict(
        ((split,
          (R2RBatch(feat_dict, obj_d_feat=obj_d_feat, obj_s_feat = obj_s_feat, batch_size=args.batchSize, splits=[split], tokenizer=tok),
           Evaluation([split], featurized_scans, tok))
          )
         for split in val_env_names
         )
    )

    if args.train == 'listener':
        train(train_env, tok, args.iters, val_envs=val_envs)
    elif args.train == 'validlistener':
        if args.beam:
            beam_valid(train_env, tok, val_envs=val_envs)
        else:
            valid(train_env, tok, val_envs=val_envs)
    elif args.train == 'speaker':
        train_speaker(train_env, tok, args.iters, val_envs=val_envs)
    elif args.train == 'validspeaker':
        valid_speaker(tok, val_envs)
    else:
        assert False


def valid_speaker(tok, val_envs):
    import tqdm
    listner = Seq2SeqAgent(None, "", tok, args.maxAction)
    speaker = Speaker(None, listner, tok)
    speaker.load(args.load)

    for env_name, (env, evaluator) in val_envs.items():
        if env_name == 'train':
            continue
        print("............ Evaluating %s ............." % env_name)
        speaker.env = env
        path2inst, loss, word_accu, sent_accu = speaker.valid(wrapper=tqdm.tqdm)
        path_id = next(iter(path2inst.keys()))
        print("Inference: ", tok.decode_sentence(path2inst[path_id]))
        print("GT: ", evaluator.gt[path_id]['instructions'])
        pathXinst = list(path2inst.items())
        name2score = evaluator.lang_eval(pathXinst, no_metrics={'METEOR'})
        score_string = " "
        for score_name, score in name2score.items():
            score_string += "%s_%s: %0.4f " % (env_name, score_name, score)
        print("For env %s" % env_name)
        print(score_string)
        print("Average Length %0.4f" % utils.average_length(path2inst))


def train_val_augment():
    """
    Train the listener with the augmented data
    """
    setup()

    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    # Load the env img features
    feat_dict = read_img_features(features)
    featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

    # Load the augmentation data
    aug_path = args.aug

    # Create the training environment
    train_env = R2RBatch(feat_dict, batch_size=args.batchSize,
                         splits=['train'], tokenizer=tok)
    aug_env   = R2RBatch(feat_dict, batch_size=args.batchSize,
                         splits=[aug_path], tokenizer=tok, name='aug')

    # Printing out the statistics of the dataset
    stats = train_env.get_statistics()
    print("The training data_size is : %d" % train_env.size())
    print("The average instruction length of the dataset is %0.4f." % (stats['length']))
    print("The average action length of the dataset is %0.4f." % (stats['path']))
    stats = aug_env.get_statistics()
    print("The augmentation data size is %d" % aug_env.size())
    print("The average instruction length of the dataset is %0.4f." % (stats['length']))
    print("The average action length of the dataset is %0.4f." % (stats['path']))

    # Setup the validation data
    val_envs = {split: (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split],
                                 tokenizer=tok), Evaluation([split], featurized_scans, tok))
                for split in ['train', 'val_seen', 'val_unseen']}

    # Start training
    train(train_env, tok, args.iters, val_envs=val_envs, aug_env=aug_env)


if __name__ == "__main__":
    if args.train in ['speaker', 'rlspeaker', 'validspeaker',
                      'listener', 'validlistener']:
        train_val()
    elif args.train == 'auglistener':
        train_val_augment()
    else:
        assert False

