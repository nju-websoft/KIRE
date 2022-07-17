import bz2file
import os
import json
import numpy as np
import pickle
import requests
import time
from bs4 import BeautifulSoup
from tqdm import tqdm
import sys

sys.path.append('./wikidata_server')
from api import query_one_hop_relation_triplets, query_property_triplets, query_entity_label


def tokenizer(sentence):
    sentence = spacynlp(sentence)
    results = []
    for token in sentence:
        results.append(token.text)
    return results


def tokenizer_1(sentence):
    for char in [',', '.', '|', '?', '？', '\'', '\\', '"']:
        sentence = sentence.replace(char, ' ' + char + ' ')
    results = sentence.split()
    return results


class KGMapping(object):

    def __init__(self):
        super().__init__()
        self.entityMappingFile = out_path + '/label2kgids.pkl'
        if os.path.exists(self.entityMappingFile):
            self.label2kgids = pickle.load(open(self.entityMappingFile, 'rb'))
        else:
            self.label2kgids = self._getEntityMappingFile()

    def _getEntityMappingFile(self):
        entityLabel2WikidataId = {}
        with open(out_path + '/entityLabel2WikidataId.txt', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                item = line.strip().split('\t')
                label = self._handleLabel(item[0])
                eid = item[1].split('/')[-1]
                entityLabel2WikidataId[label] = eid

        docred_entities_labels = set()
        data_file_names = [in_path + '/train_annotated.json', in_path + '/dev.json', in_path + '/test.json']
        for data_file_name in data_file_names:
            ori_data = json.load(open(data_file_name))

            for i in range(len(ori_data)):
                vertexSet = ori_data[i]['vertexSet']
                for vertex in vertexSet:
                    for v in vertex:
                        label = self._handleLabel(v['name'])
                        docred_entities_labels.add(label)

        entity_labels_nt3 = {}
        with open('/data1/pub/wikidata/entity_labels.nt3', 'r', encoding='utf-8') as f:
            for line in f:
                item = line.strip().split('\t')
                if len(item) < 2:
                    continue
                id = item[0]
                label = self._handleLabel(item[1])
                if label in docred_entities_labels:
                    entity_labels_nt3[label] = id

        label2kgids = dict()

        missing_num = 0
        for label in docred_entities_labels:
            if label in entityLabel2WikidataId:
                id = entityLabel2WikidataId[label]
                label2kgids[label] = id
            elif label in entity_labels_nt3:
                id = entity_labels_nt3[label]
                label2kgids[label] = id
            else:
                missing_num += 1
                print("%s missing ids" % label)
        print("entity linking end，missing num / total num =%d / %d" % (
            missing_num, len(docred_entities_labels)))  # missing / total = 4113 / 63256

        pickle.dump(label2kgids, open(self.entityMappingFile, 'wb'))
        return label2kgids

    @staticmethod
    def _handleLabel(label):
        return str(label).replace('\n', ' ').replace('.', '').replace('-', ' ').lower()

    def getEntityId(self, label):
        label = self._handleLabel(label)
        if label in self.label2kgids:
            return self.label2kgids[label]
        else:
            return None

    @staticmethod
    def get_entity_wikidataid():
        entity_set = set()
        entity2wikidataid = {}  # key=label  value=["<http://www.wikidata.org/entity/Q148>","China"]
        with open(out_path + '/entityLabel2WikidataId_wikidata.txt', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                items = line.strip().split('\t')
                if items[1] != 'None':
                    entity_set.add(items[0])
                    entity2wikidataid[items[0]] = '\t'.join(items[1:])

        data_file_names = [in_path + '/test.json', in_path + '/train_annotated.json', in_path + '/dev.json']
        f_out = open(out_path + '/entityLabel2WikidataId_wikidata.txt', 'w', encoding='utf-8')
        for data_file_name in data_file_names:
            ori_data = json.load(open(data_file_name, encoding='utf-8'))
            for i in range(len(ori_data)):
                vertexSet = ori_data[i]['vertexSet']
                for x in vertexSet:
                    for x1 in x:
                        label = x1['name']
                        label = str(label).replace('.', '').replace('-', ' ')
                        if label in entity_set:
                            continue
                        entity_set.add(label)
                        wikidataid = KGMapping._get_wikidata_idby_wikidata(label)

                        if wikidataid is None:
                            new_label = KGMapping._get_wikipedia_redirect(label)
                            if new_label is not None:
                                print("redirect\t", label, '\t', new_label)
                                wikidataid = KGMapping._get_wikidata_idby_wikidata(new_label)

                        entity2wikidataid[label] = wikidataid

                if len(entity2wikidataid) > 50:
                    for key in entity2wikidataid.keys():
                        f_out.write(key + "\t" + str(entity2wikidataid[key]) + "\n")
                    f_out.flush()
                    entity2wikidataid = {}

        for key in entity2wikidataid.keys():
            f_out.write(key + "\t" + str(entity2wikidataid[key]) + "\n")
        f_out.close()

    @staticmethod
    def _get_wikidata_id_by_falcon2(label):

        try:
            falcon2_url = 'https://labs.tib.eu/falcon/falcon2/api?mode=long'
            results = requests.post(url=falcon2_url, json={"text": label},
                                    headers={"Content-Type": "application/json;charset=UTF-8",
                                             # "Accept-Encoding": "gzip,deflate,br",
                                             # "Accept-Language": "zh-CN,zh;q=0.9",
                                             # "X-Requested-With": "XMLHttpRequest",
                                             # "Cookie": "_ga = GA1.2.1831385661.1600737557;_gid = GA1.2.1658241475.1600737557;_gat_gtag_UA_129277158_1 = 1",
                                             # "Referer": "https://labs.tib.eu/falcon/falcon2/",
                                             # "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.109 Safari/537.3"
                                             })
        except Exception as e:
            print(e)
            time.sleep(30)
            return None
        print(results.text)
        try:
            results_json = json.loads(results.text, encoding='utf-8')
            if len(results_json['entities_wikidata']) > 0:
                return results_json['entities_wikidata'][0]
            else:
                return "None"
        except Exception as e:
            print("label=%s error" % label)
            print(e)
        return "None"

    @staticmethod
    def _get_wikidata_idby_wikidata(label):
        try:
            wikidata_url = 'https://www.wikidata.org/w/index.php?sort=relevance&search=' + label + '&title=Special%3ASearch&profile=advanced&fulltext=1&advancedSearch-current=%7B%7D&ns0=1&ns120=1'
            results = requests.get(url=wikidata_url,
                                   headers={"Content-Type": "application/json;charset=UTF-8",
                                            # "Accept-Encoding": "gzip,deflate,br",
                                            # "Accept-Language": "zh-CN,zh;q=0.9",
                                            # "X-Requested-With": "XMLHttpRequest",
                                            # "Cookie": "_ga = GA1.2.1831385661.1600737557;_gid = GA1.2.1658241475.1600737557;_gat_gtag_UA_129277158_1 = 1",
                                            # "Referer": "https://labs.tib.eu/falcon/falcon2/",
                                            # "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.109 Safari/537.3"
                                            })
        except Exception as e:
            print(e)
            time.sleep(30)
            return None

        try:
            soup = BeautifulSoup(results.text, 'html.parser', from_encoding='utf-8')
            results = soup.find_all("div", "mw-search-result-heading")

            for result in results:
                return '\t'.join([result.a['href'], result.a['title']])
        except Exception as e:
            print("label=%s error" % label)
            print(e)
        return "None"

    @staticmethod
    def _get_wikipedia_redirect(label):
        try:
            wikipedia_url = 'https://en.wikipedia.org/w/index.php?title=' + label + '&redirect=no'
            results = requests.get(url=wikipedia_url,
                                   headers={"Content-Type": "application/json;charset=UTF-8",
                                            })
        except Exception as e:
            print(e)
            time.sleep(30)
            return None

        try:
            soup = BeautifulSoup(results.text, 'html.parser', from_encoding='utf-8')
            results = soup.find_all("ul", "redirectText")

            for result in results:
                return result.a['title']
        except Exception as e:
            print("label=%s error" % label)
            print(e)
        return None


def add_entity_kgid_for_dataset():
    """
        add kb_id for each entity in document
        :return:
    """
    kgmapping = KGMapping()
    docred_triplets = []
    entity_id = {}
    relation_id = {}
    docred_eid = 1
    entity_label = {}

    data_file_names = [in_path + '/train_annotated.json', in_path + '/dev.json', in_path + '/test.json']
    suffixs = ['_train', '_dev', '_test']
    out_path = in_path
    name_prefix = "dev"
    for data_file_name, suffix in zip(data_file_names, suffixs):

        ori_data = json.load(open(data_file_name, encoding='utf-8'))
        for i in range(len(ori_data)):
            Ls = [0]
            L = 0
            for x in ori_data[i]['sents']:
                L += len(x)
                Ls.append(L)

            vertexSet = ori_data[i]['vertexSet']
            # point position added with sent start position
            for j in range(len(vertexSet)):
                for k in range(len(vertexSet[j])):
                    vertexSet[j][k]['sent_id'] = int(vertexSet[j][k]['sent_id'])

                    sent_id = vertexSet[j][k]['sent_id']
                    dl = Ls[sent_id]
                    if 'absolute_pos' in vertexSet[j][k]:
                        vertexSet[j][k]['pos'] = (vertexSet[j][k]['absolute_pos'][0], vertexSet[j][k]['absolute_pos'][1])
                    else:
                        pos1 = vertexSet[j][k]['pos'][0]
                        pos2 = vertexSet[j][k]['pos'][1]
                        vertexSet[j][k]['pos'] = (pos1 + dl, pos2 + dl)

            ori_data[i]['vertexSet'] = vertexSet
            for vertex in vertexSet:
                try:
                    temp = {}
                    for v in vertex:
                        label = v['name']
                        eid = kgmapping.getEntityId(label)
                        if "link" in v and (v['link'] is not None) and "wikidata" in v['link']:
                            eid = v['link'].split(':')[1]
                        v['wikidata_eid'] = eid
                        if eid is not None:
                            if eid not in temp:
                                temp[eid] = 1
                            else:
                                temp[eid] += 1
                    if len(temp) == 0:
                        for v in vertex:
                            v['wikidata_eid'] = "NT" + str(docred_eid)
                        find_eid = "NT" + str(docred_eid)
                        docred_eid += 1
                    else:
                        temp = list(temp.items())
                        temp.sort(key=lambda x: x[1], reverse=True)
                        find_eid = temp[0][0]

                    for v in vertex:
                        v['wikidata_eid'] = find_eid
                        assert find_eid != ""
                    entity_label[find_eid] = str(vertex[0]['name']).replace('\n', ' ').replace('.', '').replace('-',
                                                                                                                ' ').lower()
                except Exception as e:
                    print(vertexSet)
                    print(vertex)
                    exit(-1)

            if 'labels' in ori_data[i]:
                for label in ori_data[i]['labels']:
                    r = label['r']
                    h = label['h']
                    t = label['t']
                    h_eids = set()
                    t_eids = set()

                    for h_v in vertexSet[h]:
                        h_eids.add(h_v['wikidata_eid'])
                    for t_v in vertexSet[t]:
                        t_eids.add(t_v['wikidata_eid'])

                    assert len(h_eids) > 0 and len(t_eids) > 0
                    for h_eid in h_eids:
                        for t_eid in t_eids:
                            if (h_eid, r, t_eid) not in docred_triplets:
                                docred_triplets.append((h_eid, r, t_eid))

        json.dump(ori_data, open(os.path.join(out_path, name_prefix + suffix + '.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False)

    triplets_f_out = open(out_path + '/one_hop_relation_triplets.nt3', 'w', encoding='utf-8')
    triplets_ids_f_out = open(out_path + '/one_hop_relation_triplets_ids.nt3', 'w', encoding='utf-8')
    entity_f_out = open(out_path + '/one_hop_entity2id.txt', 'w', encoding='utf-8')
    relation_f_out = open(out_path + '/one_hop_relation2id.txt', 'w', encoding='utf-8')
    entity_label_out = open(out_path + '/one_hop_entity2label.txt', 'w', encoding='utf-8')
    eid = 0
    rid = 0
    for triplet in docred_triplets:
        triplets_f_out.write('\t'.join(triplet) + '\n')
        for entity in [triplet[0], triplet[2]]:
            if entity not in entity_id:
                entity_id[entity] = eid
                entity_f_out.write(entity + '\t' + str(eid) + '\n')
                eid += 1
        if triplet[1] not in relation_id:
            relation_id[triplet[1]] = rid
            relation_f_out.write(triplet[1] + '\t' + str(rid) + '\n')
            rid += 1
        triplets_ids_f_out.write(
            str(entity_id[triplet[0]]) + '\t' + str(relation_id[triplet[1]]) + '\t' + str(entity_id[triplet[2]]) + '\n')
    for k, v in entity_label.items():
        entity_label_out.write(k + '\t' + v + '\n')

    triplets_f_out.close()
    triplets_ids_f_out.close()
    entity_f_out.close()
    relation_f_out.close()
    entity_label_out.close()


def generate_kg_subgraph_for_dataset():

    docred_entities_ids = set()
    data_file_names = [in_path + '/dev_train.json', in_path + '/dev_dev.json', in_path + '/dev_test.json']
    for data_file_name in data_file_names:
        ori_data = json.load(open(data_file_name))
        for i in range(len(ori_data)):
            vertexSet = ori_data[i]['vertexSet']
            for vertex in vertexSet:
                for v in vertex:
                    docred_entities_ids.add(v['wikidata_eid'])

    def get_one_hop_triplets(entity_ids: set, outpath=None):
        if outpath is not None:
            outfile = open(outpath, 'w', encoding='utf-8')
        triplets = set()
        with open('/data1/pub/wikidata/relation_triplets.nt3', 'r', encoding='utf-8') as f:
            for line in f:
                item = line.strip().split('\t')
                if len(item) < 3:
                    continue
                if item[0] in entity_ids or item[2] in entity_ids:
                    if outpath is not None:
                        outfile.write('\t'.join(item) + '\n')
                    else:
                        triplets.add(tuple(item))
        if outpath is not None:
            outfile.close()
        return triplets

    def save_triplets(triplets: set, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            for t in triplets:
                f.write('\t'.join(t) + '\n')

    one_hop_relation_triplets = get_one_hop_triplets(docred_entities_ids)
    save_triplets(one_hop_relation_triplets, '/data1/pub/wikidata/' + dataset.lower() + '/one_hop_relation_triplets.nt3')

rel2id = {'P159': 0,
 'P17': 1,
 'P131': 2,
 'P150': 3,
 'P27': 4,
 'P569': 5,
 'P19': 6,
 'P172': 7,
 'P571': 8,
 'P576': 9,
 'P607': 10,
 'P30': 11,
 'P276': 12,
 'P1376': 13,
 'P206': 14,
 'P495': 15,
 'P551': 16,
 'P264': 17,
 'P527': 18,
 'P463': 19,
 'P175': 20,
 'P577': 21,
 'P161': 22,
 'P403': 23,
 'P20': 24,
 'P69': 25,
 'P570': 26,
 'P108': 27,
 'P166': 28,
 'P6': 29,
 'P361': 30,
 'P36': 31,
 'P26': 32,
 'P25': 33,
 'P22': 34,
 'P40': 35,
 'P37': 36,
 'P1412': 37,
 'P800': 38,
 'P178': 39,
 'P400': 40,
 'P937': 41,
 'P102': 42,
 'P585': 43,
 'P740': 44,
 'P3373': 45,
 'P1001': 46,
 'P57': 47,
 'P58': 48,
 'P272': 49,
 'P155': 50,
 'P156': 51,
 'P194': 52,
 'P241': 53,
 'P127': 54,
 'P118': 55,
 'P39': 56,
 'P674': 57,
 'P179': 58,
 'P1441': 59,
 'P170': 60,
 'P449': 61,
 'P86': 62,
 'P488': 63,
 'P1344': 64,
 'P580': 65,
 'P582': 66,
 'P676': 67,
 'P54': 68,
 'P50': 69,
 'P840': 70,
 'P136': 71,
 'P205': 72,
 'P706': 73,
 'P162': 74,
 'P710': 75,
 'P35': 76,
 'P140': 77,
 'P1336': 78,
 'P364': 79,
 'P737': 80,
 'P279': 81,
 'P31': 82,
 'P137': 83,
 'P112': 84,
 'P123': 85,
 'P176': 86,
 'P749': 87,
 'P355': 88,
 'P1198': 89,
 'P171': 90,
 'P1056': 91,
 'P1366': 92,
 'P1365': 93,
 'P807': 94,
 'P190': 95}

def generate_subgraph_data_for_kg_input(data_file_name, max_length=512, is_training=True, suffix=''):
    if is_training:
        name_prefix = "train"
    else:
        name_prefix = "dev"

    ori_data = json.load(open(data_file_name, encoding='utf-8'))
    sen_tot = len(ori_data)
    all_attr_size = sen_tot * max_attr_size

    kg_entity_attrs = np.zeros((all_attr_size, max_attr_len), dtype=np.int32)
    kg_entity_attr_lens = np.zeros((all_attr_size), dtype=np.int32)
    kg_entity_attr_nums = np.zeros((sen_tot), dtype=object)
    kg_entity_attr_indexs = np.zeros((sen_tot, max_entity_size), dtype=object)
    kg_entity_nids = np.zeros((sen_tot), dtype=object)
    kg_entity_adj = np.zeros((sen_tot), dtype=object)
    kg_adj_node_nums = np.zeros((1), dtype=object)
    kg_ent_mask = np.zeros((sen_tot, max_length), dtype=np.int32)

    t_max_attr_len = 0
    t_max_entity_size = 0
    t_max_attr_size = 0
    word2id = json.load(open('./word2id.json', 'r', encoding='utf-8'))
    word2id = {word.lower(): id for word, id in word2id.items()}


    entity2label_f = open(one_hop_entity2label_path, encoding='utf-8', mode='r')
    entity2label = {}
    for lin in entity2label_f.readlines():
        item = lin.strip().split('\t')
        if len(item) < 2:
            print(item)
            continue
        entity2label[item[0]] = item[1]

    adj_node_nums = []
    attrid = 0
    for i in tqdm(range(len(ori_data))):

        nid = 0
        wid2nid = {}
        wids = []
        vertexSet = ori_data[i]['vertexSet']
        nids = []
        wid2type = {}
        for j, vertex in enumerate(vertexSet):
            wid = vertex[0]['wikidata_eid']
            type = vertex[0]['type']
            wid2type[wid] = type
            if wid not in wid2nid:
                wid2nid[wid] = []
            wids.append(wid)
            wid2nid[wid].append(nid)
            nids.append(nid)
            attr_indexs = []
            for v in vertex:
                for index in range(v['pos'][0], v['pos'][1]):
                    kg_ent_mask[i][index] = 1
                    attr_indexs.append(index)
            kg_entity_attr_indexs[i][nid] = attr_indexs
            nid += 1
        kg_entity_nids[i] = nids

        adj = []
        one_hop_kg_relation_triplets = query_one_hop_relation_triplets([wid for wid in wids if wid2type[wid] != 'TIME'])
        for h, r, t in one_hop_kg_relation_triplets:
            if h not in wid2nid:
                wid2nid[h] = [nid]
                nid += 1
                wids.append(h)
            if t not in wid2nid:
                wid2nid[t] = [nid]
                nid += 1
                wids.append(t)
            for h_nid in wid2nid[h]:
                for t_nid in wid2nid[t]:
                    if r not in rel2id:
                        rel2id[r] = len(rel2id)
                    adj.append((h_nid, t_nid, rel2id[r]))
                    adj.append((t_nid, h_nid, rel2id[r]))  # undircted
        kg_entity_adj[i] = adj
        adj_node_nums.append(nid)

        doc_attrs = []
        attr_nums = []
        def query_entity_attr(wids):
            labels = query_entity_label(wids)
            for k,v in labels.items():
                if v is None:
                    v = entity2label[k]
                assert v is not  None
                labels[k] = v
            all_attribute_triplets = query_property_triplets(wids)
            all_attribute_triplets_rc = {}
            for wid in wids:
                label = labels[wid]
                attribute_triplets = all_attribute_triplets[wid]
                attribute_triplets_rc = {"label": [label], "description": [], "instance of": [label], "alias": []}
                for attr_triplet in attribute_triplets:
                    p = attr_triplet[1]
                    v = attr_triplet[2]
                    attribute_triplets_rc[p].append(v)
                if len(attribute_triplets_rc['description']) == 0:
                    attribute_triplets_rc['description'].append(label)
                if len(attribute_triplets_rc['alias']) == 0:
                    attribute_triplets_rc['alias'].append(label)
                all_attribute_triplets_rc[wid] = attribute_triplets_rc
            return all_attribute_triplets_rc

        attribute_triplets = query_entity_attr(wids)

        for wid in wids:
            doc_attrs.append('label # ' + ' '.join(attribute_triplets[wid]['label']))
            doc_attrs.append('description # ' + attribute_triplets[wid]['description'][0])
            doc_attrs.append('instance of # ' + ' '.join(attribute_triplets[wid]['instance of']))
            doc_attrs.append('alias # ' + ' '.join(attribute_triplets[wid]['alias']))
            attr_nums.append(4)

        kg_entity_attr_nums[i] = attr_nums
        t_max_attr_size = max(t_max_attr_size, len(doc_attrs))
        for attr in doc_attrs:
            vs = attr
            attr_ids = [word2id['unk'] if v.lower() not in word2id else word2id[v.lower()] for v in tokenizer_1(vs)]
            attr_ids = attr_ids[:max_attr_len]
            kg_entity_attrs[attrid][:len(attr_ids)] = attr_ids
            kg_entity_attr_lens[attrid] = len(attr_ids)
            t_max_attr_len = max(t_max_attr_len, len(attr_ids))
            attrid += 1

    kg_adj_node_nums[0] = adj_node_nums

    print("t_max_attr_len", t_max_attr_len) # 128
    print("t_max_entity_size", t_max_entity_size)
    print("t_max_attr_size", t_max_attr_size) # 30992
    print("attrid", attrid) # 1334872


    np.save(os.path.join(out_path, name_prefix + suffix + '_kg_subg_entity_attrs.npy'), kg_entity_attrs)
    np.save(os.path.join(out_path, name_prefix + suffix + '_kg_subg_entity_attr_lens.npy'), kg_entity_attr_lens)
    np.save(os.path.join(out_path, name_prefix + suffix + '_kg_subg_entity_attr_indexs.npy'), kg_entity_attr_indexs)
    np.save(os.path.join(out_path, name_prefix + suffix + '_kg_subg_entity_attr_nums.npy'), kg_entity_attr_nums)
    np.save(os.path.join(out_path, name_prefix + suffix + '_kg_subg_entity_adj.npy'), kg_entity_adj)
    np.save(os.path.join(out_path, name_prefix + suffix + '_kg_subg_entity_mask.npy'), kg_ent_mask)
    np.save(os.path.join(out_path, name_prefix + suffix + '_kg_subg_entity_nids.npy'), kg_entity_nids)
    np.save(os.path.join(out_path, name_prefix + suffix + '_kg_subg_entity_adj_nums.npy'), kg_adj_node_nums)


if __name__ == '__main__':
    dataset = "dwie"
    if dataset == "dwie":
        in_path = './DWIE/data'
        out_path = './DWIE/processed'
        max_length = 1800
        max_attr_len = 128
        max_attr_size = 15000
        max_entity_size = 100
        one_hop_entity2label_path = './DWIE/data/one_hop_entity2label.txt'
    else:
        in_path = './DocRED/data'
        out_path = './DocRED/processed'
        max_length = 512
        max_attr_len = 128
        max_attr_size = 6000
        max_entity_size = 42
        one_hop_entity2label_path = './DocRED/data/one_hop_entity2label.txt'

    KGMapping.get_entity_wikidataid()
    add_entity_kgid_for_dataset()
    generate_kg_subgraph_for_dataset()


    train_annotated_file_name = os.path.join(in_path, 'dev_train.json')
    dev_file_name = os.path.join(in_path, 'dev_dev.json')
    test_file_name = os.path.join(in_path, 'dev_test.json')
    generate_subgraph_data_for_kg_input(train_annotated_file_name, max_length=max_length, is_training=False,
                                                    suffix='_train')
    generate_subgraph_data_for_kg_input(dev_file_name, max_length=max_length, is_training=False, suffix='_dev')

    generate_subgraph_data_for_kg_input(test_file_name, max_length=max_length, is_training=False, suffix='_test')
    print(rel2id)
    with open(out_path + '/wikidatarel2id.txt', 'w') as f:
        for k,v in rel2id.items():
            f.write(str(k) + '\t' + str(v) + '\n')