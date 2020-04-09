''' Batched Room-to-Room navigation environment '''

import sys
sys.path.append('buildpy36')
import MatterSim
import csv
import numpy as np
import math
import base64
import utils
import json
import os
import random
import networkx as nx
from param import args

from utils import load_datasets, load_nav_graphs, Tokenizer

csv.field_size_limit(sys.maxsize)


class EnvBatch():
    ''' A simple wrapper for a batch of MatterSim environments, 
        using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store=None, batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feature_store: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        if feature_store:
            if type(feature_store) is dict:     # A silly way to avoid multiple reading
                self.features = feature_store
                self.image_w = 640
                self.image_h = 480
                self.vfov = 60
                self.feature_size = next(iter(self.features.values())).shape[-1]
                print('The feature size is %d' % self.feature_size)
        else:
            print('Image features not provided')
            self.features = None
            self.image_w = 640
            self.image_h = 480
            self.vfov = 60
        self.featurized_scans = set([key.split("_")[0] for key in list(self.features.keys())])
        self.sims = []
        for i in range(batch_size):
            sim = MatterSim.Simulator()
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)   # Set increment/decrement to 30 degree. (otherwise by radians)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.init()
            self.sims.append(sim)

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId   

    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            # print("New episode %d" % i)
            # sys.stdout.flush()
            self.sims[i].newEpisode(scanId, viewpointId, heading, 0)
  
    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((30, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        for i, sim in enumerate(self.sims):
            state = sim.getState()

            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if self.features:
                feature = self.features[long_id]     # Get feature for
                feature_states.append((feature, state))
            else:
                feature_states.append((None, state))
        return feature_states

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input). 
            Every action element should be an (index, heading, elevation) tuple. '''
        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction(index, heading, elevation)

class ObjEnvBatch(EnvBatch):
    def __init__(self, feature_store=None,obj_d_feat = None,obj_s_feat=None,batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feature_store: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        super(ObjEnvBatch, self).__init__(feature_store, batch_size)
        if obj_d_feat is not None:
           self.obj_d_feat = obj_d_feat
        else:
           self.obj_d_feat = None
        if obj_s_feat is not None:
            self.obj_s_feat = obj_s_feat
        else:
            self.obj_s_feat = None
        # self.obj_d_feat = obj_d_feat
    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((30, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        obj_d_feat_states = []
        obj_s_feat_states = []
        for i, sim in enumerate(self.sims):
            state = sim.getState()

            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if self.features:
                feature = self.features[long_id]     # Get feature for
                feature_states.append((feature, state))
            else:
                feature_states.append((None, state))
            if self.obj_d_feat:
                obj_d_feat = self.obj_d_feat[long_id]
                obj_d_feat_states.append(obj_d_feat)
            else:
                obj_d_feat_states.append(None)
            if self.obj_s_feat:
                obj_s_feat = self.obj_s_feat[long_id]
                obj_s_feat_states.append(obj_s_feat)
            else:
                obj_s_feat_states.append(None)
        return feature_states, obj_d_feat_states, obj_s_feat_states


class R2RBatch():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store, obj_d_feat = None, obj_s_feat=None, batch_size=100, seed=10, splits=['train'], tokenizer=None,
                 name=None):
        if obj_d_feat or obj_s_feat:
            self.env = ObjEnvBatch(feature_store=feature_store, obj_d_feat = obj_d_feat, obj_s_feat= obj_s_feat, batch_size=batch_size)
        else:
            self.env = EnvBatch(feature_store=feature_store, batch_size=batch_size)
        if feature_store:
            self.feature_size = self.env.feature_size
        # if obj_d_feat:
        #     self.obj_d_feature_size = self.env.obj_d_feat_size
        self.data = []
        if tokenizer:
            self.tok = tokenizer
        scans = []
        for split in splits:
            for item in load_datasets([split]):
                # Split multiple instructions into separate entries
                for j,instr in enumerate(item['instructions']):
                    if item['scan'] not in self.env.featurized_scans:   # For fast training
                        continue
                    new_item = dict(item)
                    new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                    new_item['instructions'] = instr
                    if tokenizer:
                        new_item['instr_encoding'] = tokenizer.encode_sentence(instr)
                    if not tokenizer or new_item['instr_encoding'] is not None:  # Filter the wrong data
                        self.data.append(new_item)
                        scans.append(item['scan'])
        if name is None:
            self.name = splits[0] if len(splits) > 0 else "FAKE"
        else:
            self.name = name

        self.scans = set(scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)

        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()

        self.angle_feature = utils.get_all_point_angle_feature()
        self.sim = utils.new_simulator()
        self.buffered_state_dict = {}

        # It means that the fake data is equals to data in the supervised setup
        self.fake_data = self.data
        print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

    def size(self):
        return len(self.data)

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self, tile_one=False, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        :param tile_one: Tile the one into batch_size
        :return: None
        """
        if batch_size is None:
            batch_size = self.batch_size
        if tile_one:
            batch = [self.data[self.ix]] * batch_size
            self.ix += 1
            if self.ix >= len(self.data):
                random.shuffle(self.data)
                self.ix -= len(self.data)
        else:
            batch = self.data[self.ix: self.ix+batch_size]
            if len(batch) < batch_size:
                random.shuffle(self.data)
                self.ix = batch_size - len(batch)
                batch += self.data[:self.ix]
            else:
                self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing. 
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return goalViewpointId      # Just stop here
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    def make_candidate(self, feature, scanId, viewpointId, viewId, obj_d_feat=None, obj_s_feat=None):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
        base_heading = (viewId % 12) * math.radians(30)
        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)
        if long_id not in self.buffered_state_dict:
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode(scanId, viewpointId, 0, math.radians(-30))
                elif ix % 12 == 0:
                    self.sim.makeAction(0, 1.0, 1.0)
                else:
                    self.sim.makeAction(0, 1.0, 0)

                state = self.sim.getState()
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation

                visual_feat = feature[ix]
                if obj_d_feat:
                    odf = obj_d_feat[ix]

                # if obj_s_feat:
                #     num_obj = 0
                #     obj_index = []
                #     for n_obj, viewIndex in enumerate(obj_s_feat['concat_viewIndex']):
                #         if viewIndex == ix:
                #             num_obj += 1
                #             obj_index.append(n_obj)
                #     concat_angles_h = obj_s_feat['concat_angles_h'][obj_index]
                #     concat_angles_e = obj_s_feat['concat_angles_e'][obj_index]
                #     concat_feature = obj_s_feat['concat_feature'][obj_index]
                #     odf = concat_feature
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)

                    # Heading and elevation for for the loc
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = utils.angle_feature(loc_heading, loc_elevation)
                    if (loc.viewpointId not in adj_dict or
                            distance < adj_dict[loc.viewpointId]['distance']):
                        adj_dict[loc.viewpointId] = {
                            'heading': loc_heading,
                            'elevation': loc_elevation,
                            "normalized_heading": state.heading + loc.rel_heading,
                            'scanId':scanId,
                            'viewpointId': loc.viewpointId, # Next viewpoint id
                            'pointId': ix,
                            'distance': distance,
                            'idx': j + 1,
                            'feature': np.concatenate((visual_feat, angle_feat), -1),
                            'ang_feat': angle_feat
                        }
                        if obj_d_feat:
                            adj_dict[loc.viewpointId]['obj_d_feat'] = odf
            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {key: c[key]
                 for key in
                    ['normalized_heading', 'elevation', 'scanId', 'viewpointId',
                     'pointId', 'idx']}
                for c in candidate
            ]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                ix = c_new['pointId']
                normalized_heading = c_new['normalized_heading']
                visual_feat = feature[ix]
                loc_heading = normalized_heading - base_heading
                c_new['heading'] = loc_heading
                angle_feat = utils.angle_feature(c_new['heading'], c_new['elevation'])
                c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)
                c_new['ang_feat'] = angle_feat
                if obj_d_feat:
                    c_new['obj_feat'] = obj_d_feat[ix]
                c_new.pop('normalized_heading')
                candidate_new.append(c_new)
            return candidate_new

    def _get_obs(self):
        obs = []
        # for i, (feature, state) in enumerate(self.env.getStates()):
        if args.sparseObj or args.denseObj:
            F, obj_d_feat, obj_s_feat = self.env.getStates()
        else:
            F = self.env.getStates()
        for i in range(len(F)):
            feature = F[i][0]
            state = F[i][1]
            # odf = obj_d_feat[i]
            if args.sparseObj:
                osf = obj_s_feat[i]
            if args.denseObj:
                odf = obj_d_feat[i]
            item = self.batch[i]
            base_view_id = state.viewIndex

            # Full features
            candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)

            # (visual_feature, angel_feature) for views
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)
            ang_feat = self.angle_feature[base_view_id]
            obs_dict = {
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature,
                'ang_feat': ang_feat,
                'candidate': candidate,
                'navigableLocations' : state.navigableLocations,
                'instructions' : item['instructions'],
                'teacher' : self._shortest_path_action(state, item['path'][-1]),
                'path_id' : item['path_id']
            }
            if args.sparseObj:
                obs_dict['obj_s_feature'] = osf['concat_feature']
                obs_dict['bbox_angle_h'] = osf['concat_angles_h']
                obs_dict['bbox_angle_e'] = osf['concat_angles_e']
            if args.denseObj:
                obs_dict['obj_d_feature'] = odf['concat_feature']
                obs_dict['bbox_angle_h'] = odf['concat_angles_h']
                obs_dict['bbox_angle_e'] = odf['concat_angles_e']
            obs.append(obs_dict)

            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
            # A2C reward. The negative distance between the state and the final state
            obs[-1]['distance'] = self.distances[state.scanId][state.location.viewpointId][item['path'][-1]]
        return obs

    def reset(self, batch=None, inject=False, **kwargs):
        ''' Load a new minibatch / episodes. '''
        if batch is None:       # Allow the user to explicitly define the batch
            self._next_minibatch(**kwargs)
        else:
            if inject:          # Inject the batch into the next minibatch
                self._next_minibatch(**kwargs)
                self.batch[:len(batch)] = batch
            else:               # Else set the batch to the current batch
                self.batch = batch
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()

    def get_statistics(self):
        stats = {}
        length = 0
        path = 0
        for datum in self.data:
            length += len(self.tok.split_sentence(datum['instructions']))
            path += self.distances[datum['scan']][datum['path'][0]][datum['path'][-1]]
        stats['length'] = length / len(self.data)
        stats['path'] = path / len(self.data)
        return stats

# class BottomUpImageFeatures(ImageFeatures):
#     PAD_ITEM = ("<pad>",)
#     feature_dim = ImageFeatures.MEAN_POOLED_DIM
#
#     def __init__(self, number_of_detections, precomputed_cache_path=None, precomputed_cache_dir=None, image_width=640, image_height=480):
#         self.number_of_detections = number_of_detections
#         self.index_to_attributes, self.attribute_to_index = BottomUpImageFeatures.read_visual_genome_vocab(paths.bottom_up_attribute_path, BottomUpImageFeatures.PAD_ITEM, add_null=True)
#         self.index_to_objects, self.object_to_index = BottomUpImageFeatures.read_visual_genome_vocab(paths.bottom_up_object_path, BottomUpImageFeatures.PAD_ITEM, add_null=False)
#
#         self.num_attributes = len(self.index_to_attributes)
#         self.num_objects = len(self.index_to_objects)
#
#         self.attribute_pad_index = self.attribute_to_index[BottomUpImageFeatures.PAD_ITEM]
#         self.object_pad_index = self.object_to_index[BottomUpImageFeatures.PAD_ITEM]
#
#         self.image_width = image_width
#         self.image_height = image_height
#
#         self.precomputed_cache = {}
#         def add_to_cache(key, viewpoints):
#             assert len(viewpoints) == ImageFeatures.NUM_VIEWS
#             viewpoint_feats = []
#             for viewpoint in viewpoints:
#                 params = {}
#                 for param_key, param_value in viewpoint.items():
#                     if param_key == 'cls_prob':
#                         # make sure it's in descending order
#                         assert np.all(param_value[:-1] >= param_value[1:])
#                     if param_key == 'boxes':
#                         # TODO: this is for backward compatibility, remove it
#                         param_key = 'spatial_features'
#                         param_value = spatial_feature_from_bbox(param_value, self.image_height, self.image_width)
#                     assert len(param_value) >= self.number_of_detections
#                     params[param_key] = param_value[:self.number_of_detections]
#                 viewpoint_feats.append(BottomUpViewpoint(**params))
#             self.precomputed_cache[key] = viewpoint_feats
#
#         if precomputed_cache_dir:
#             self.precomputed_cache = {}
#             import glob
#             for scene_dir in glob.glob(os.path.join(precomputed_cache_dir, "*")):
#                 scene_id = os.path.basename(scene_dir)
#                 pickle_file = os.path.join(scene_dir, "d={}.pkl".format(number_of_detections))
#                 with open(pickle_file, 'rb') as f:
#                     data = pickle.load(f)
#                     for (viewpoint_id, viewpoints) in data.items():
#                         key = (scene_id, viewpoint_id)
#                         add_to_cache(key, viewpoints)
#         elif precomputed_cache_path:
#             self.precomputed_cache = {}
#             with open(precomputed_cache_path, 'rb') as f:
#                 data = pickle.load(f)
#                 for (key, viewpoints) in data.items():
#                     add_to_cache(key, viewpoints)
#
#     @staticmethod
#     def read_visual_genome_vocab(fname, pad_name, add_null=False):
#         # one-to-many mapping from indices to names (synonyms)
#         index_to_items = []
#         item_to_index = {}
#         start_ix = 0
#         items_to_add = [pad_name]
#         if add_null:
#             null_tp = ()
#             items_to_add.append(null_tp)
#         for item in items_to_add:
#             index_to_items.append(item)
#             item_to_index[item] = start_ix
#             start_ix += 1
#
#         with open(fname) as f:
#             for index, line in enumerate(f):
#                 this_items = []
#                 for synonym in line.split(','):
#                     item = tuple(synonym.split())
#                     this_items.append(item)
#                     item_to_index[item] = index + start_ix
#                 index_to_items.append(this_items)
#         assert len(index_to_items) == max(item_to_index.values()) + 1
#         return index_to_items, item_to_index
#
#     def batch_features(self, feature_list):
#         def transform(lst, wrap_with_var=True):
#             features = np.stack(lst)
#             x = torch.from_numpy(features)
#             if wrap_with_var:
#                 x = Variable(x, requires_grad=False)
#             return try_cuda(x)
#
#         return BottomUpViewpoint(
#             cls_prob=transform([f.cls_prob for f in feature_list]),
#             image_features=transform([f.image_features for f in feature_list]),
#             attribute_indices=transform([f.attribute_indices for f in feature_list]),
#             object_indices=transform([f.object_indices for f in feature_list]),
#             spatial_features=transform([f.spatial_features for f in feature_list]),
#             no_object_mask=transform([f.no_object_mask for f in feature_list], wrap_with_var=False),
#         )
#
#     def parse_attribute_objects(self, tokens):
#         parse_options = []
#         # allow blank attribute, but not blank object
#         for split_point in range(0, len(tokens)):
#             attr_tokens = tuple(tokens[:split_point])
#             obj_tokens = tuple(tokens[split_point:])
#             if attr_tokens in self.attribute_to_index and obj_tokens in self.object_to_index:
#                 parse_options.append((self.attribute_to_index[attr_tokens], self.object_to_index[obj_tokens]))
#         assert parse_options, "didn't find any parses for {}".format(tokens)
#         # prefer longer objects, e.g. "electrical outlet" over "electrical" "outlet"
#         return parse_options[0]
#
#     @functools.lru_cache(maxsize=20000)
#     def _get_viewpoint_features(self, scan_id, viewpoint_id):
#         if self.precomputed_cache:
#             return self.precomputed_cache[(scan_id, viewpoint_id)]
#
#         fname = os.path.join(paths.bottom_up_feature_store_path, scan_id, "{}.p".format(viewpoint_id))
#         with open(fname, 'rb') as f:
#             data = pickle.load(f, encoding='latin1')
#
#         viewpoint_features = []
#         for viewpoint in data:
#             top_indices = k_best_indices(viewpoint['cls_prob'], self.number_of_detections, sorted=True)[::-1]
#
#             no_object = np.full(self.number_of_detections, True, dtype=np.uint8) # will become torch Byte tensor
#             no_object[0:len(top_indices)] = False
#
#             cls_prob = np.zeros(self.number_of_detections, dtype=np.float32)
#             cls_prob[0:len(top_indices)] = viewpoint['cls_prob'][top_indices]
#             assert cls_prob[0] == np.max(cls_prob)
#
#             image_features = np.zeros((self.number_of_detections, ImageFeatures.MEAN_POOLED_DIM), dtype=np.float32)
#             image_features[0:len(top_indices)] = viewpoint['features'][top_indices]
#
#             spatial_feats = np.zeros((self.number_of_detections, 5), dtype=np.float32)
#             spatial_feats[0:len(top_indices)] = spatial_feature_from_bbox(viewpoint['boxes'][top_indices], self.image_height, self.image_width)
#
#             object_indices = np.full(self.number_of_detections, self.object_pad_index)
#             attribute_indices = np.full(self.number_of_detections, self.attribute_pad_index)
#
#             for i, ix in enumerate(top_indices):
#                 attribute_ix, object_ix = self.parse_attribute_objects(list(viewpoint['captions'][ix].split()))
#                 object_indices[i] = object_ix
#                 attribute_indices[i] = attribute_ix
#
#             viewpoint_features.append(BottomUpViewpoint(cls_prob, image_features, attribute_indices, object_indices, spatial_feats, no_object))
#         return viewpoint_features
#
#     def get_features(self, state):
#         viewpoint_features = self._get_viewpoint_features(state.scanId, state.location.viewpointId)
#         return viewpoint_features[state.viewIndex]
#
#     def get_name(self):
#         return "bottom_up_attention_d={}".format(self.number_of_detections)