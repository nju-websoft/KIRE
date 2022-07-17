import numpy as np
import os
import torch
import copy


class KG_data_loader():

    def __init__(self, prefix, dataset):
        self.dataset = dataset
        if dataset == "docred":
            self.max_length = 512
            kg_in_path = './data/DocRED/processed'
        elif dataset== 'dwie':
            kg_in_path = './data/DWIE/processed'
            self.max_length = 1800

        self.kg_ent_adj = np.load(os.path.join(kg_in_path, prefix + '_kg_subg_entity_adj.npy'), allow_pickle=True)
        self.kg_ent_adj_nums = np.load(os.path.join(kg_in_path, prefix + '_kg_subg_entity_adj_nums.npy'),
                                       allow_pickle=True)
        self.kg_ent_mask = np.load(os.path.join(kg_in_path, prefix + '_kg_subg_entity_mask.npy'))
        self.kg_ent_nids = np.load(os.path.join(kg_in_path, prefix + '_kg_subg_entity_nids.npy'), allow_pickle=True)
        self.kg_ent_attrs = np.load(os.path.join(kg_in_path, prefix + '_kg_subg_entity_attrs.npy'))
        self.kg_ent_attr_lens = np.load(os.path.join(kg_in_path, prefix + '_kg_subg_entity_attr_lens.npy'))
        self.kg_ent_attr_indexs = np.load(os.path.join(kg_in_path, prefix + '_kg_subg_entity_attr_indexs.npy'),
                                          allow_pickle=True)
        self.kg_ent_attr_nums = np.load(os.path.join(kg_in_path, prefix + '_kg_subg_entity_attr_nums.npy'),
                                        allow_pickle=True)

        kg_ent_attr_cnts = []
        for i in range(len(self.kg_ent_attr_nums)):
            kg_ent_attr_cnts.append(sum(self.kg_ent_attr_nums[i]))
        self.kg_ent_attr_cnts = [0]
        for cnt in kg_ent_attr_cnts:
            self.kg_ent_attr_cnts.append(self.kg_ent_attr_cnts[-1] + cnt)

        # add self-loop
        for pmid in range(len(self.kg_ent_adj)):
            kg_ent_adj = self.kg_ent_adj[pmid]
            adj_node_nums = self.kg_ent_adj_nums[0][int(pmid)]
            for i in range(int(adj_node_nums)):
                kg_ent_adj.append((i, i, 1060))

        self.wikidatarel2id = {}
        with open('./knowledge_injection_layer/wikidatarel2id.txt') as f:
            for line in f.readlines():
                items = line.strip().split('\t')
                self.wikidatarel2id[items[0]] = int(items[1])
        self.id2wikidatarel = {v: k for k,v in self.wikidatarel2id.items()}

        self.coref_h_mapping = np.load(os.path.join(kg_in_path, prefix + '_coref_h_mapping_dense.npy'),
                                           allow_pickle=True)
        self.coref_t_mapping = np.load(os.path.join(kg_in_path, prefix + '_coref_t_mapping_dense.npy'),
                                           allow_pickle=True)
        self.coref_label = np.load(os.path.join(kg_in_path, prefix + '_coref_label.npy'))
        self.coref_lens = np.load(os.path.join(kg_in_path, prefix + '_coref_lens.npy'))
        self.coref_label_mask = np.load(os.path.join(kg_in_path, prefix + '_coref_label_mask.npy'))
        self.coref_mention_position = np.load(os.path.join(kg_in_path, prefix + '_coref_mention_position.npy'))

        self.prefix = prefix

        self.dis2idx_dir = np.zeros((2*self.max_length), dtype='int64')
        self.dis2idx_dir[1] = 1
        self.dis2idx_dir[2:] = 2
        self.dis2idx_dir[4:] = 3
        self.dis2idx_dir[8:] = 4
        self.dis2idx_dir[16:] = 5
        self.dis2idx_dir[32:] = 6
        self.dis2idx_dir[64:] = 7
        self.dis2idx_dir[128:] = 8
        self.dis2idx_dir[256:] = 9
        self.dis_size = 20

    def get_kg_batch(self, dids, batch_max_length):
        batch_size = len(dids)
        if self.dataset == 'docred':
            max_attr_len = 128
            max_entity_size = 42
            max_attr_size = 12000
            if batch_size <= 4:
                max_attr_size = 15000
            max_length = 512
            max_coref_mention_size = 250
            max_pair_cnt = 3200
        elif self.dataset == 'dwie':
            max_attr_len = 128
            max_entity_size = 100
            max_attr_size = 15000
            if batch_size <= 2:
                max_attr_size = 45000
            elif batch_size <= 4:
                max_attr_size = 45000
            max_length = 1800
            max_coref_mention_size = 800
            max_pair_cnt = 9000

        IGNORE_INDEX = -100

        kg_ent_mask = torch.LongTensor(batch_size, max_length).cuda()
        kg_ent_labels = torch.LongTensor(batch_size, max_length).cuda()
        coref_h_mapping = torch.FloatTensor(batch_size, max_pair_cnt, max_length).cuda()
        coref_t_mapping = torch.FloatTensor(batch_size, max_pair_cnt, max_length).cuda()
        coref_dis = torch.LongTensor(batch_size, max_pair_cnt).cuda()
        coref_lens = torch.LongTensor(batch_size, max_coref_mention_size).cuda()
        coref_label = torch.FloatTensor(batch_size, max_pair_cnt, 2).cuda()
        coref_label_mask = torch.BoolTensor(batch_size, max_pair_cnt).cuda()
        coref_mention_position = torch.FloatTensor(batch_size, max_coref_mention_size, max_length).cuda()
        kg_ent_attrs = torch.LongTensor(batch_size * max_attr_size, max_attr_len).cuda()
        kg_ent_attr_lens = torch.LongTensor(batch_size * max_attr_size)

        kg_ent_labels.fill_(IGNORE_INDEX)
        coref_label_mask.fill_(False)
        kg_ent_adj = []
        kg_ent_adj_edges = []
        kg_ent_attr_nums = []
        kg_ent_attr_indexs = np.zeros((batch_size, max_entity_size), dtype=object)
        for k in [kg_ent_mask, coref_h_mapping, coref_t_mapping, coref_lens, coref_label, coref_mention_position,
                  kg_ent_attrs, kg_ent_attr_lens, coref_dis]:
            k.zero_()

        max_coref_pair = 0
        attr_index = 0
        max_ent_adj_nums = max([int(self.kg_ent_adj_nums[0][index]) for index in dids])
        for i, index in enumerate(dids):
            def _convert2sparse(adj, shape, entity_node_size):
                visit = set()
                row_ids = []
                col_ids = []
                vals = []
                edges = []
                for h, t, r in adj:
                    if ((h, t) not in visit):
                        if self.prefix != "dev_test":
                            visit.add((h, t))
                            row_ids.append(h)
                            col_ids.append(t)
                            vals.append(1.0)
                            edges.append(r)
                        elif (h == t or h >= entity_node_size or t >= entity_node_size):
                            visit.add((h, t))
                            row_ids.append(h)
                            col_ids.append(t)
                            vals.append(1.0)
                            edges.append(r)
                indices = torch.from_numpy(np.vstack((row_ids, col_ids)).astype(np.int64))
                values = torch.FloatTensor(vals)
                edges = torch.LongTensor(edges)
                return torch.sparse.FloatTensor(indices, values, shape), torch.sparse.LongTensor(indices, edges, shape)

            _ent_adj, _ent_edge = _convert2sparse(self.kg_ent_adj[index], shape=(max_ent_adj_nums, max_ent_adj_nums), entity_node_size=len(self.kg_ent_nids[index]))

            kg_ent_adj.append(_ent_adj.cuda())
            kg_ent_adj_edges.append(_ent_edge.cuda())
            kg_ent_mask[i].copy_(torch.from_numpy(self.kg_ent_mask[index]))

            for j, nid in enumerate(self.kg_ent_nids[index]):
                for index1 in self.kg_ent_attr_indexs[index][j]:
                    kg_ent_labels[i, index1] = nid
            attr_nums = copy.deepcopy(self.kg_ent_attr_nums[index])
            attr_s = self.kg_ent_attr_cnts[index]
            attr_e = self.kg_ent_attr_cnts[index + 1]
            assert attr_e - attr_s == sum(attr_nums)
            kg_ent_attrs[attr_index: attr_index + attr_e - attr_s].copy_(
                torch.from_numpy(self.kg_ent_attrs[attr_s: attr_e]))
            kg_ent_attr_lens[attr_index: attr_index + attr_e - attr_s].copy_(
                torch.from_numpy(self.kg_ent_attr_lens[attr_s: attr_e]))
            kg_ent_attr_nums.append(attr_nums)
            kg_ent_attr_indexs[i] = self.kg_ent_attr_indexs[index]
            attr_index += sum(attr_nums)

            coref_h_mapping[i].copy_(torch.from_numpy(self.coref_h_mapping[index].todense()))
            coref_t_mapping[i].copy_(torch.from_numpy(self.coref_t_mapping[index].todense()))
            h_t_dis = (self.coref_t_mapping[index].todense()>0).argmax(axis=1) - (self.coref_h_mapping[index].todense()>0).argmax(axis=1) + self.max_length
            h_t_dis = np.asarray([self.dis2idx_dir[x] for x in h_t_dis])
            coref_dis[i].copy_(torch.from_numpy(h_t_dis).squeeze())
            coref_label[i].copy_(torch.from_numpy(self.coref_label[index]))
            coref_lens[i].copy_(torch.from_numpy(self.coref_lens[index]))
            coref_label_mask[i].copy_(torch.from_numpy(self.coref_label_mask[index]))
            coref_mention_position[i].copy_(torch.from_numpy(self.coref_mention_position[index]))
            max_coref_pair = max(max_coref_pair, int(coref_label_mask[i].sum().item()))
        return{
            'kg_ent_mask': kg_ent_mask[:batch_size, :batch_max_length].contiguous(),
            'kg_ent_attrs': kg_ent_attrs[:attr_index].contiguous(),
            'kg_ent_attr_lens': kg_ent_attr_lens[:attr_index].contiguous(),
            'kg_ent_attr_nums': kg_ent_attr_nums,
            'kg_ent_attr_indexs': kg_ent_attr_indexs,
            'kg_ent_labels': kg_ent_labels[:batch_size, :batch_max_length].contiguous(),
            'kg_adj': kg_ent_adj, 'kg_adj_edges': kg_ent_adj_edges,
            'coref_h_mapping': coref_h_mapping[:batch_size, :max_coref_pair, :batch_max_length].contiguous(),
            'coref_t_mapping': coref_t_mapping[:batch_size, :max_coref_pair, :batch_max_length].contiguous(),
            'coref_dis': coref_dis[:batch_size, :max_coref_pair].contiguous(),
            'coref_lens': coref_lens[:batch_size, :].contiguous(),
            'coref_label': coref_label[:batch_size, :max_coref_pair].contiguous(),
            'coref_label_mask': coref_label_mask[:batch_size, :max_coref_pair].contiguous(),
            'coref_mention_position': coref_mention_position[:batch_size, :, :batch_max_length].contiguous()
        }