import json
import numpy as np
from scipy import sparse
from spacy.tokenizer import Tokenizer
import spacy
import neuralcoref
import os

spacynlp = spacy.load('en_core_web_md')
spacynlp.Defaults.suffixes = ''
spacynlp.Defaults.infix = ''
spacynlp.Defaults.prefix = ''
spacynlp.tokenizer = Tokenizer(spacynlp.vocab)
neuralcoref.add_to_pipe(spacynlp, greedyness=0.5, max_dist=500)


def transform_sparse(input_dense, max_length):
    sparse_mxs_list = []
    input_shape = input_dense.shape
    for i in range(input_shape[0]):
        temp_input_dense = np.zeros((input_shape[1], max_length), np.float32)
        for j in range(input_shape[1]):
            index = input_dense[i][j]
            temp_input_dense[j][index[0]: index[1]+1] = 1.0 / (index[1]+1 - index[0])
        temp_sparse = sparse.csr_matrix(temp_input_dense)
        sparse_mxs_list.append(temp_sparse)
    return sparse_mxs_list


def sigmoid_fun(x):
    return 1/(1+np.exp(-x))


def get_corefinfo_by_spacy(document, doc_len):
    doc = spacynlp(document)
    assert doc.__len__() == doc_len, print(doc_len, doc.__len__(), '\n', doc, '\n', document)
    coref_scores = {}

    for mention1, mentions2 in doc._.coref_scores.items():
        mentions2 = sorted(mentions2.items(), key= lambda x: x[1], reverse=True)
        for mention2, score in mentions2[:8]:
            if mention1.start == mention2.start:
                continue
            score = sigmoid_fun(score)
            if mention1 not in coref_scores:
                coref_scores[mention1] = {}
            if mention2 not in coref_scores[mention1]:
                coref_scores[mention1][mention2] = score
            if mention2 not in coref_scores:
                coref_scores[mention2] = {}
            if mention1 not in coref_scores[mention2]:
                coref_scores[mention2][mention1] = score
    return coref_scores


def gen_coref(data_file_name, max_length=512, is_training=True, suffix=''):

    if is_training:
        name_prefix = "train"
    else:
        name_prefix = "dev"

    ori_data = json.load(open(data_file_name, encoding='utf-8'))
    sen_tot = len(ori_data)
    coref_h_mapping = np.zeros((sen_tot, max_pair_cnt, 2), dtype=np.int32)
    coref_t_mapping = np.zeros((sen_tot, max_pair_cnt, 2), dtype=np.int32)
    coref_lens = np.zeros((sen_tot, max_coref_mention_size), dtype=np.int32)
    coref_label = np.ones((sen_tot, max_pair_cnt, 2), dtype=np.float32)
    coref_label_mask = np.zeros((sen_tot, max_pair_cnt), dtype=np.bool)
    coref_mention_position = np.zeros((sen_tot, max_coref_mention_size, max_length), dtype=np.int32)
    max_mentions_pairs = 0
    max_mentions = 0
    for i in range(len(ori_data)):
        Ls = [0]
        L = 0
        sentences = []
        for x in ori_data[i]['sents']:
            L += len(x)
            Ls.append(L)

        for x in ori_data[i]['sents']:
            sentences.extend(x)

        coref_scores = get_corefinfo_by_spacy(' '.join(sentences), len(sentences))

        pair_id = 0
        for i_index, (mention1, mentions2) in enumerate(coref_scores.items()):
            coref_lens[i][i_index] = len(mentions2)
            coref_mention_position[i][i_index][mention1.start: mention1.end] = 1
            for mention2, score in mentions2.items():
                assert mention1.start < len(sentences) and mention1.end-1 < len(sentences)
                assert mention2.start < len(sentences) and mention2.end - 1 < len(sentences)

                coref_h_mapping[i][pair_id] = (mention1.start, mention1.end-1)
                coref_t_mapping[i][pair_id] = (mention2.start, mention2.end-1)
                coref_label[i][pair_id][1] = score  # the score for co-reference
                coref_label[i][pair_id][0] = 1 - coref_label[i][pair_id][1]
                coref_label_mask[i][pair_id] = True
                pair_id += 1

        max_mentions_pairs = max(max_mentions_pairs, pair_id)
        max_mentions = max(max_mentions, i_index)

    np.save(os.path.join(out_path, name_prefix + suffix + '_coref_h_mapping.npy'), coref_h_mapping)
    np.save(os.path.join(out_path, name_prefix + suffix + '_coref_t_mapping.npy'), coref_t_mapping)
    np.save(os.path.join(out_path, name_prefix + suffix + '_coref_lens.npy'), coref_lens)
    np.save(os.path.join(out_path, name_prefix + suffix + '_coref_label.npy'), coref_label)
    np.save(os.path.join(out_path, name_prefix + suffix + '_coref_label_mask.npy'), coref_label_mask)
    np.save(os.path.join(out_path, name_prefix + suffix + '_coref_mention_position.npy'), coref_mention_position)
    print("max_mentions_pairs", max_mentions_pairs)
    print("max_mentions", max_mentions)


def transform_to_sparse(max_length=512, is_training=True, suffix=''):
    if is_training:
        name_prefix = "train"
    else:
        name_prefix = "dev"

    coref_h_mapping = np.load(os.path.join(out_path, name_prefix + suffix + '_coref_h_mapping.npy'))
    coref_t_mapping = np.load(os.path.join(out_path, name_prefix + suffix + '_coref_t_mapping.npy'))

    coref_h_mapping = transform_sparse(coref_h_mapping, max_length)
    coref_t_mapping = transform_sparse(coref_t_mapping, max_length)

    np.save(os.path.join(out_path, name_prefix + suffix + '_coref_h_mapping_dense.npy'), coref_h_mapping)
    np.save(os.path.join(out_path, name_prefix + suffix + '_coref_t_mapping_dense.npy'), coref_t_mapping)


def get_coref_by_alias():
    priori_value = 0.9
    name_prefix = 'dev'
    in_paths = ['./data/kg_docred_dataset/dev_train.json', './data/kg_docred_dataset/dev_dev.json', './data/kg_docred_dataset/dev_test.json']
    suffixs = ['_train', '_dev', '_test']
    for data_file_name, suffix in zip(in_paths, suffixs):
        ori_data = json.load(open(data_file_name, encoding='utf-8'))
        sen_tot = len(ori_data)
        coref_h_mapping = np.zeros((sen_tot, max_pair_cnt, 2), dtype=np.int32)
        coref_t_mapping = np.zeros((sen_tot, max_pair_cnt, 2), dtype=np.int32)
        coref_lens = np.zeros((sen_tot, max_coref_mention_size), dtype=np.int32)
        coref_label = np.ones((sen_tot, max_pair_cnt, 2), dtype=np.float32)
        coref_label_mask = np.zeros((sen_tot, max_pair_cnt), dtype=np.bool)
        coref_mention_position = np.zeros((sen_tot, max_coref_mention_size, max_length), dtype=np.int32)
        max_mentions_pairs = 0
        max_mentions = 0

        for i in range(len(ori_data)):
            sentences = []
            for x in ori_data[i]['sents']:
                sentences.extend(x)

            pair_id = 0
            vertexSet = ori_data[i]['vertexSet']
            mentions_set = set()
            coref_mentions_dict = {}
            for vertex in vertexSet:
                attribute_triplets = vertex[0]['attribute_triplets']
                alias_set = set()
                for triplet in attribute_triplets:
                    if triplet[1] in ['label', 'alias']:
                        alias_set.add(triplet[2])

                def find_position(sentence, alias):
                    pos = set()
                    alias_len = len(alias)
                    for s_index in range(len(sentence)):
                        if sentence[s_index: s_index+alias_len] == alias:
                            pos.add((s_index, s_index+alias_len))
                    return pos
                coref_pos = set()
                for alias in alias_set:
                    pos = find_position(sentences, alias)
                    for _ in pos:
                        coref_pos.add(_)
                for m1 in vertex:
                    coref_pos.add((m1['pos'][0], m1['pos'][1]))

                for m1id, m1 in enumerate(coref_pos):
                    for m2id, m2 in enumerate(coref_pos):
                        if m1id != m2id:
                            if m1 not in mentions_set:
                                mentions_set.add(m1)
                                coref_mentions_dict[m1] = set()
                            if m2 not in mentions_set:
                                mentions_set.add(m2)
                                coref_mentions_dict[m2] = set()
                            coref_mentions_dict[m1].add(m2)
                            coref_mentions_dict[m2].add(m1)
            i_index = 0
            for (mention1, mentions2) in coref_mentions_dict.items():
                if len(mentions2) == 0:
                    continue
                coref_lens[i][i_index] = len(mentions2)
                coref_mention_position[i][i_index][mention1[0]: mention1[1]] = 1
                for mention2 in mentions2:
                    coref_h_mapping[i][pair_id] = (mention1[0], mention1[1] - 1)
                    coref_t_mapping[i][pair_id] = (mention2[0], mention2[1] - 1)
                    coref_label[i][pair_id][1] = priori_value  # the score for co-reference
                    coref_label[i][pair_id][0] = 1 - priori_value
                    coref_label_mask[i][pair_id] = True
                    pair_id += 1
                i_index += 1

            max_mentions_pairs = max(max_mentions_pairs, pair_id)
            max_mentions = max(max_mentions, i_index)

        coref_h_mapping = transform_sparse(coref_h_mapping, max_length)
        coref_t_mapping = transform_sparse(coref_t_mapping, max_length)

        np.save(os.path.join(out_path, name_prefix + suffix + '_alias_coref_h_mapping_dense.npy'), coref_h_mapping)
        np.save(os.path.join(out_path, name_prefix + suffix + '_alias_coref_t_mapping_dense.npy'), coref_t_mapping)
        np.save(os.path.join(out_path, name_prefix + suffix + '_alias_coref_lens.npy'), coref_lens)
        np.save(os.path.join(out_path, name_prefix + suffix + '_alias_coref_label.npy'), coref_label)
        np.save(os.path.join(out_path, name_prefix + suffix + '_alias_coref_label_mask.npy'), coref_label_mask)
        np.save(os.path.join(out_path, name_prefix + suffix + '_alias_coref_mention_position.npy'), coref_mention_position)
        print(max_mentions_pairs)
        print(max_mentions)


if __name__ == '__main__':
    dataset = "docred"
    if dataset == "dwie":
        in_path = './DWIE/data'
        out_path = 'DWIE/processed'
        max_length = 1800
        max_coref_mention_size = 800  # 相关mention个数
        max_pair_cnt = 9000
    else:
        in_path = './data'
        out_path = './kg_data'
        max_length = 512
        max_coref_mention_size = 250  # 相关mention个数
        max_pair_cnt = 3200

    train_annotated_file_name = os.path.join(in_path, 'train_annotated.json')
    dev_file_name = os.path.join(in_path, 'dev.json')
    test_file_name = os.path.join(in_path, 'test.json')  # 新的去除了空白文档的版本
    gen_coref(train_annotated_file_name, max_length=max_length, is_training=False, suffix='_train')
    gen_coref(dev_file_name, max_length=max_length, is_training=False, suffix='_dev')
    gen_coref(test_file_name, max_length=max_length, is_training=False, suffix='_test')
    transform_to_sparse(max_length=max_length, is_training=False, suffix='_train')
    transform_to_sparse(max_length=max_length, is_training=False, suffix='_dev')
    transform_to_sparse(max_length=max_length, is_training=False, suffix='_test')

    get_coref_by_alias()