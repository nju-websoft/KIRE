import math
from collections import Counter

import torch
# from pytorch_transformers.modeling_transfo_xl_utilities import LogUniformSampler
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import json
import random
import numpy as np
from torch.optim import Adam
from tqdm import tqdm
import os

"""
参考代码 https://github.com/erickrf/autoencoder/tree/master/src
参考论文 There and Back Again: Autoencoders for Textual Reconstruction
"""

class AutoEncoderModel(nn.Module):
    def __init__(self, lstm_units, embeddings, go, train_embeddings=False, bidirectional=True, gpuid=1):
        """
            Initialize the encoder/decoder and creates Tensor objects
            :param lstm_units: number of LSTM units
            :param embeddings: numpy array with initial embeddings
            :param go: index of the GO symbol in the embedding matrix
            :param train_embeddings: whether to adjust embeddings during training
            :param bidirectional: whether to create a bidirectional autoencoder
            """
        # EOS and GO share the same symbol. Only GO needs to be embedded, and
        # only EOS exists as a possible network output
        super(AutoEncoderModel, self).__init__()
        self.go = go
        self.eos = go
        self.bidirectional = bidirectional
        self.vocab_size = embeddings.shape[0]
        self.embedding_size = embeddings.shape[1]
        self.device = torch.device("cuda" if gpuid != -1 else "cpu")
        self.lstm_units = lstm_units

        self._init_encoder(embeddings, train_embeddings)

        # projection
        # self.projection = nn.Linear(2* self.lstm_units, self.vocab_size)

        # self.sampledSoftmax = SampledSoftmax(self.vocab_size, 100, 2*self.lstm_units) #, tied_weight=self.projection.weight, tied_bias=self.projection.bias)
        # self.criterion = nn.CrossEntropyLoss()

    def _init_encoder(self, embeddings, train_embeddings):
        self.embeddings = nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float(), freeze=train_embeddings)

        self.lstm_encoder = nn.LSTM(input_size=self.embedding_size, hidden_size=self.lstm_units, num_layers=1,
                                    batch_first=True, bidirectional=self.bidirectional)

    def forward(self, sentences, lens):
        """
        :param sentences: <batch, max_len>
        :param lens: <batch>

        :return [b, 1, 28, 28]:
        """
        max_len = sentences.size(1)
        batch_size = sentences.size(0)
        embedded = self.embeddings(sentences)
        # embedded = F.dropout(embedded, 0.5, self.training)

        # encoding step
        word_vec_packed = pack_padded_sequence(embedded, lens, batch_first=True, enforce_sorted=False)
        encoded_sent_outputs, (encoded_sent_hiddens, encoded_sent_cells) = self.lstm_encoder(word_vec_packed)  # batch， max_len, -1
        encoded_sent_outputs, _ = pad_packed_sequence(encoded_sent_outputs, batch_first=True)
        # print(encoded_sent_hiddens.size())  # 2,32,200


        # generate a batch of embedded GO
        # sentence_size has the batch dimension
        go_batch = self._generate_batch_go(lens)
        embedded_eos = self.embeddings(go_batch)
        embedded_eos = torch.reshape(embedded_eos, [-1, 1, self.embedding_size])
        decoder_input = torch.cat([embedded_eos, embedded], dim=1)

        # decoding step
        # We give the same inputs to the forward and backward LSTMs,
        # but each one has its own hidden state
        # their outputs are concatenated and fed to the softmax layer
        word_vec_packed1 = pack_padded_sequence(decoder_input, lens+1, batch_first=True, enforce_sorted=False)
        decoded_sent_outputs, (decoded_sent_hiddens, _) = self.lstm_encoder(word_vec_packed1, (encoded_sent_hiddens, encoded_sent_cells))
        decoded_sent_outputs, _ = pad_packed_sequence(decoded_sent_outputs, batch_first=True, total_length=max_len+1)
        decoder_outputs = decoded_sent_outputs  # <batch_size, max_len, 2*lstm_unit>

        eos_batch = self._generate_batch_go(lens).reshape([-1, 1])
        decoder_labels = torch.cat([sentences, eos_batch], -1).cuda()

        # set the importance of each time step
        # 1 if before sentence end or EOS itself; 0 otherwise
        masks = (torch.arange(max_len + 1).unsqueeze(0).repeat(batch_size, 1) < (lens+1).unsqueeze(1)).float().cuda()
        num_actual_labels = torch.sum(masks)
        # print("num_actual_labels", num_actual_labels)
        # print(torch.sum(lens))
        # reshape to have batch and time steps in the same dimension
        decoder_outputs2d = torch.reshape(decoder_outputs, [-1, decoder_outputs.size(-1)])
        labels = torch.reshape(decoder_labels, [-1])

        logits, new_targets = self.sampledSoftmax(decoder_outputs2d, labels)

        sampled_loss = self.criterion(logits, new_targets)

        masked_loss = masks.reshape([-1]) * sampled_loss
        loss = torch.mean(masked_loss) / num_actual_labels

        return loss

    def decode(self, sentences, lens):
        """
            Run the autoencoder with the given data
            :param sentences: 2-d array with the word indices
            :param lens: 1-d array with size of each sentence
            :return: a 2-d array (batch, output_length) with the answer
                    produced by the autoencoder. The output length is not
                    fixed; it stops after producing EOS for all items in the
                    batch or reaching two times the maximum number of time
                    steps in the inputs.
        """
        print(sentences)
        print(lens)
        embedded = self.embeddings(sentences)

        # encoding step
        word_vec_packed = pack_padded_sequence(embedded, lens, batch_first=True, enforce_sorted=False)
        encoded_sent_outputs, (encoded_sent_hiddens, encoded_sent_cells) = self.lstm_encoder(word_vec_packed)  # batch， max_len, -1
        encoded_sent_outputs, _ = pad_packed_sequence(encoded_sent_outputs, batch_first=True)

        time_steps = 0
        max_time_steps = 2 * len(sentences[0])
        answer = []
        input_symbol = self.go * torch.ones_like(lens, dtype=torch.long).cuda()

        # this array control which sequences have already been finished by the
        # decoder, i.e., for which ones it already produced the END symbol
        sequences_done = torch.zeros_like(lens, dtype=torch.bool).cuda()

        while True:
            input_symbol = input_symbol.long().view(1, -1)
            input_symbol = self.embeddings(input_symbol)
            word_vec_packed = pack_padded_sequence(input_symbol, [1], batch_first=True, enforce_sorted=False)
            decoded_sent_outputs, (decoded_sent_hiddens, decoded_sent_cells) = self.lstm_encoder(word_vec_packed, (encoded_sent_hiddens, encoded_sent_cells))
            decoded_sent_outputs, _ = pad_packed_sequence(decoded_sent_outputs, batch_first=True) # <batch_size, max_len, 2*lstm_unit>
            encoded_sent_hiddens = decoded_sent_hiddens
            encoded_sent_cells = decoded_sent_cells

            # now project the outputs to the vocabulary
            logits,  _= self.sampledSoftmax(decoded_sent_outputs, None)
            logits = logits.view(1, -1)  # 移除batch_size
            input_symbol = logits.argmax(1)
            answer.append(input_symbol.cpu().data)

            # use an "additive" or in order to avoid infinite loops
            sequences_done |= (input_symbol == self.eos)

            if sequences_done.all() or time_steps > max_time_steps:
                break
            else:
                time_steps += 1
        return np.hstack(answer)

    def encode(self, sentences, lens):
        """
        todo 判断一下是否需要加入 go_symbol
        :param sentences:
        :param lens:
        :return:
        """
        embedded = self.embeddings(sentences)
        word_vec_packed = pack_padded_sequence(embedded, lens, batch_first=True, enforce_sorted=False)
        encoded_sent_outputs, (encoded_sent_hiddens, encoded_sent_cells) = self.lstm_encoder(word_vec_packed)  # batch， max_len, -1
        encoded_sent_outputs, _ = pad_packed_sequence(encoded_sent_outputs, batch_first=True)
        return encoded_sent_outputs

    def _generate_batch_go(self, like):
        """
        Generate a 1-d tensor with copies of EOS as big as the batch size,
        :param like: a tensor whose shape the returned embeddings should match
        :return: a tensor with shape as `like`
        """
        ones = torch.ones_like(like).cuda()
        return ones * self.go


class SampledSoftmax(nn.Module):
    def __init__(self, ntokens, nsampled, nhid, tied_weight=None, tied_bias=None):
        super(SampledSoftmax, self).__init__()

        # Parameters
        self.ntokens = ntokens
        self.nsampled = nsampled

        self.sampler = LogUniformSampler(self.ntokens, nsampled)
        self.params = nn.Linear(nhid, ntokens)

        if tied_weight is not None:
            self.params.weight = tied_weight
            self.params.bias = tied_bias
        # else:
        #     torch.nn.utils.initialize(self.params.weight)

    def forward(self, inputs, labels):
        if self.training:
            # sample ids according to word distribution - Unique
            sample_values = self.sampler.sample(labels.data.cpu())
            return self.sampled(inputs, labels, sample_values, remove_accidental_match=True)
        else:
            return self.full(inputs, labels)

    def sampled(self, inputs, labels, sample_values, remove_accidental_match=False):
        # print("inputs", inputs.size())
        # print(labels.size())
        assert(inputs.data.get_device() == labels.data.get_device())
        device_id = labels.data.get_device()

        batch_size, d = inputs.size()
        true_log_freq, sample_log_freq, sample_ids = sample_values

        sample_ids = torch.LongTensor(sample_ids).cuda(device_id)
        true_log_freq = torch.FloatTensor(true_log_freq).cuda(device_id)
        sample_log_freq = torch.FloatTensor(sample_log_freq).cuda(device_id)
        # print("label", labels)
        # gather true labels - weights and frequencies
        true_weights = torch.index_select(self.params.weight, 0, labels)
        true_bias = torch.index_select(self.params.bias, 0, labels)

        # gather sample ids - weights and frequencies
        sample_weights = torch.index_select(self.params.weight, 0, sample_ids)
        sample_bias = torch.index_select(self.params.bias, 0, sample_ids)

        # calculate logits
        true_logits = torch.sum(torch.mul(inputs, true_weights), dim=1) + true_bias
        sample_logits = torch.matmul(inputs, torch.t(sample_weights)) + sample_bias
        # remove true labels from sample set
        # if remove_accidental_match:
        #     acc_hits = self.sampler.accidental_match(labels.data.cpu().numpy(), sample_ids.data.cpu().numpy())
        #     acc_hits = list(zip(*acc_hits))
        #     sample_logits[acc_hits] = -1e37

        # perform correction
        true_logits = true_logits.sub(true_log_freq)
        sample_logits = sample_logits.sub(sample_log_freq)

        # return logits and new_labels
        logits = torch.cat((torch.unsqueeze(true_logits, dim=1), sample_logits), dim=1)
        new_targets = torch.zeros(batch_size).long().cuda(device_id)
        return logits, new_targets

    def full(self, inputs, labels):
        return self.params(inputs), labels

from tqdm import tqdm
class Trainer():
    def __init__(self, train_data, dev_data, pred_embedding, eos_index):
        self.train_data = train_data
        self.dev_data = dev_data

        # 超参数
        self.embedding_size = 100
        self.save_dir = './ae_results'
        lstm_units = 100
        learning_rate = 0.01
        batch_size = 32
        num_epochs = 20
        dropout_rate = 0.0
        bidirectional = True
        train_embeddings = True
        max_length = 128
        self.gc = 5.0
        self.batch_size = batch_size
        self.learning_rate= learning_rate
        self.epochs = num_epochs  # 超参数，训练轮数
        self.go = eos_index
        self.eos = eos_index
        self.model = AutoEncoderModel(lstm_units, pred_embedding, eos_index)
        self.model = self.model.cuda()
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

    def run(self):
        # 开始训练
        best_loss = 10000
        accumulated_loss = 0
        batch_counter = 0
        num_sents = 0
        report_interval = 10000
        # dev_sents, dev_sizes = self.dev_data.join_all(self.go, shuffle=True)
        # print(min(dev_sizes))

        self.train_data.reset_epoch_counter()
        pbar = tqdm(total=self.epochs * len(self.train_data) // self.batch_size)
        while self.train_data.epoch_counter < self.epochs:
            # print('model train')
            pbar.update(1)
            self.model.train()
            batch_counter += 1
            train_sents, train_sizes = self.train_data.next_batch(self.batch_size)
            self.optimizer.zero_grad()
            loss = self.model(train_sents, train_sizes)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gc)  # gradient clipping
            self.optimizer.step()  # update
            # multiply by len because some batches may be smaller
            # (due to bucketing), then take the average
            accumulated_loss += float(loss.item()) * len(train_sents)
            num_sents += len(train_sents)

            if batch_counter % report_interval == 0:
                avg_loss = accumulated_loss / num_sents
                accumulated_loss = 0
                num_sents = 0

                # we can't use all the validation at once, since it would
                # take too much memory. running many small batches would
                # instead take too much time. So let's just sample it.
                # sample_indices = np.random.randint(0, len(dev_sents), 5000)
                # print('model eval')
                self.model.eval()
                val_loss = 0
                val_sum = 0
                while self.dev_data.epoch_counter < 1:
                    dev_sents, dev_sizes = self.dev_data.next_batch(self.batch_size)
                    loss = self.model(dev_sents, dev_sizes)
                # loss = self.model(torch.from_numpy(dev_sents[sample_indices]).long().cuda(), torch.from_numpy(dev_sizes[sample_indices]).long())
                    val_loss += float(loss.item()) * len(dev_sents)
                    val_sum += len(dev_sents)
                self.dev_data.reset_epoch_counter()

                msg = '%d epochs, %d batches\t' % (self.train_data.epoch_counter, batch_counter)
                msg += 'Avg batch loss: %f\t' % avg_loss
                msg += 'Validation loss: %f' % (val_loss/val_sum)
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(self.model.state_dict(), os.path.join(self.save_dir, 're.model'))
                    msg += '\t(saved model)'

                print(msg)

        pbar.close()

class Dataset(object):
    """
    Class to manage an autoencoder dataset. It contains a sentence
    matrix, an array with their sizes and functions to facilitate access.
    """
    def __init__(self, sentences, sizes):
        """
        :param sentences: either a matrix or a list of matrices
            (which could have different shapes)
        :param sizes: either an array or a list of arrays
            (which could have different shapes)
        """
        if not isinstance(sentences, list):
            sentences = [sentences]
            sizes = [sizes]

        self.sentence_matrices = [matrix for matrix in sentences if matrix.shape[0]>0]
        self.sizes = [matrix for matrix in sizes if matrix.shape[0]>0]
        self.num_items = sum(len(array) for array in sizes)
        self.next_batch_ind = 0
        self.last_matrix_ind = 0
        self.epoch_counter = 0
        self.largest_len = max(sent.shape[1] for sent in sentences)

    def __len__(self):
        return self.num_items

    def reset_epoch_counter(self):
        self.epoch_counter = 0

    def next_batch(self, batch_size):
        """
        Return the next batch (keeping track of the last, or from the beginning
        if this is the first call).
        Sentences are grouped in batches according to their sizes (similar sizes
        go together).
        :param batch_size: number of items to return
        :return: a tuple (sentences, sizes) with at most `batch_size`
            items. If there are not enough `batch_size`, return as much
            as there are
        """
        matrix = self.sentence_matrices[self.last_matrix_ind]
        if self.next_batch_ind >= len(matrix):
            self.last_matrix_ind += 1
            if self.last_matrix_ind >= len(self.sentence_matrices):
                self.epoch_counter += 1
                self.last_matrix_ind = 0

            self.next_batch_ind = 0
            matrix = self.sentence_matrices[self.last_matrix_ind]

        sizes = self.sizes[self.last_matrix_ind]
        from_ind = self.next_batch_ind
        to_ind = self.next_batch_ind + batch_size
        batch_sentences = matrix[from_ind:to_ind]
        batch_sizes = sizes[from_ind:to_ind]
        self.next_batch_ind = to_ind

        return torch.from_numpy(batch_sentences).long().cuda(), torch.from_numpy(batch_sizes).long()

    def join_all(self, eos, max_size=None, shuffle=True):
        """
        Join all sentence matrices and return them.
        :param eos: the eos index to fill smaller matrices
        :param max_size: number of columns in the resulting matrix
        :param shuffle: whether to shuffle data before returning
        :return: (sentences, sizes)
        """
        if max_size is None:
            max_size = max(matrix.shape[1]
                           for matrix in self.sentence_matrices)
        padded_matrices = []
        # for m, z in zip(self.sentence_matrices, self.sizes):
        #     print(m.shape)
        #     print(z.shape)

        for matrix in self.sentence_matrices:
            # if matrix.shape[0] == 0:
            #     continue
            if matrix.shape[1] == max_size:
                padded = matrix
            else:
                diff = max_size - matrix.shape[1]
                padded = np.pad(matrix, [(0, 0), (0, diff)],
                                'constant', constant_values=eos)
            padded_matrices.append(padded)

        sentences = np.vstack(padded_matrices)
        sizes = np.hstack(self.sizes)
        if shuffle:
            state = np.random.get_state()
            np.random.shuffle(sentences)
            np.random.set_state(state)
            np.random.shuffle(sizes)
        return sentences, sizes


def load_binary_data(path):
    """
    Load a numpy archive. It can have either a single 'sentences'
    and a single 'sizes' or many 'sentences-x' and 'sizes-x'.
    """
    data = np.load(path)
    if 'sentences' in data:
        return Dataset(data['sentences'], data['sizes'])

    sent_names = sorted(name for name in data.files
                        if name.startswith('sentences'))
    size_names = sorted(name for name in data.files
                        if name.startswith('sizes'))
    sents = []
    sizes = []

    for sent_name, size_name in zip(sent_names, size_names):
        sents.append(data[sent_name])
        sizes.append(data[size_name])

    return Dataset(sents, sizes)

def train():
    embedding_size = 100
    # save_dir = './ae_results'
    # lstm_units = 200
    # learning_rate = 0.001
    # batch_size = 32
    # num_epochs = 100
    # dropout_rate = 0.0
    # bidirectional = True
    # train_embeddings = True
    # max_length = 128

    # 加载词表和词向量
    word2id = json.load(open('../../GLRE/prepro_data/DocRED/processed/word2id_vec.json', 'r', encoding='utf-8'))
    vec = np.load('../../GLRE/prepro_data/DocRED/processed/vec.npy')
    # 添加特殊字符<s>
    eos_index = len(word2id)
    assert '</s>' not in word2id
    word2id['</s>'] = eos_index
    vec = np.concatenate([vec, np.random.rand(1, embedding_size)], axis=0)

    # 加载训练+验证集
    train_data = load_binary_data('./ae_results/train-data.npz')
    print("训练集句子数目", len(train_data))  # 18312793
    dev_data = load_binary_data('./ae_results/dev-data.npz')
    print("验证机句子数目", len(dev_data))  # 373725

    print("Creating model")
    trainer = Trainer(train_data, dev_data, vec, eos_index)
    # 开始训练
    trainer.run()

def test():
    sentence = 'label House Scanning the HIM Horizon .'
    tokens = sentence.lower().replace(':', ' ').replace(',', ' ').replace('.', ' ').split()

    # 加载词表和词向量
    word2id = json.load(open('../../GLRE/prepro_data/DocRED/processed/word2id_vec.json', 'r', encoding='utf-8'))
    vec = np.load('../../GLRE/prepro_data/DocRED/processed/vec.npy')
    # 添加特殊字符<s>
    eos_index = len(word2id)
    assert '</s>' not in word2id
    word2id['</s>'] = eos_index
    vec = np.concatenate([vec, np.random.rand(1, 100)], axis=0)
    id2word = {v: k for k, v in word2id.items()}

    tokens = [word2id[token] if token in word2id else word2id['UNK'] for token in tokens]
    lens = [len(tokens)]

    model = AutoEncoderModel(100, vec, eos_index)
    model = model.cuda()
    model.load_state_dict(torch.load('./ae_results/re.model'))
    model.eval()
    result = model.decode(torch.from_numpy(np.asarray(tokens)).view(1, -1).long().cuda(), torch.tensor(lens).long())
    print(result)
    result_str = ' '.join([id2word[id] for id in result])
    print(result_str)

def create_sentence_matrix(sents, num_sentences, min_size,
                           max_size, word_dict):
    """
    Create a sentence matrix from the file in the given path.
    :param path: path to text file
    :param min_size: minimum sentence length, inclusive
    :param max_size: maximum sentence length, inclusive
    :param num_sentences: number of sentences expected
    :param word_dict: mapping of words to indices
    :return: tuple (2-d matrix, 1-d array) with sentences and
        sizes
    """
    sentence_matrix = np.full((num_sentences, max_size), 0, np.int32)
    sizes = np.empty(num_sentences, np.int32)
    ori_sents = []
    i = 0
    for line in sents:
        tokens = line.lower().replace(':', ' ').replace(',', ' ').replace('.', ' ').split()
        sent_size = len(tokens)
        if sent_size < min_size or sent_size > max_size:
            continue
        array = np.array([word_dict[token] if token in word_dict else word_dict['UNK'] for token in tokens])
        sentence_matrix[i, :sent_size] = array
        sizes[i] = sent_size
        ori_sents.append(line)
        i += 1
        if i >= num_sentences:
            break

    return sentence_matrix, sizes, ori_sents

def prepare_data():
    """
    把各属性处理成句子
    :return:
    """
    valid_proportion = 0.02
    # 读取属性信息
    properties = set()
    with open('../../data/kg_docred_dataset/one_hop_property_triplets.nt3', 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            items = line.strip().split('\t')
            if len(items) < 3:
                print(line)
                continue
            if items[1] not in ["label", "description", "instance of", "alias"]:
                continue
            desp = items[2].strip('[]')
            if desp not in properties:
                properties.add(items[1] + ' ' + desp)
    print("属性加载完毕")

    with open('../../wikipedia_description.txt', 'r', encoding='utf-8') as f:
        lines = list(f.readlines())
        for i in range(0, len(lines), 3):
            properties.add('description ' + lines[i+1])


    # 读取句子
    docs = set()
    for file_path in ['../../data/train_annotated.json', '../../data/dev.json', '../../data/test.json']:
        datas = json.load(open(file_path, 'r', encoding='utf-8'))
        for data in tqdm(datas):
            for sent in data['sents']:
                sent = ' '.join(sent)
                if sent not in docs:
                    docs.add(sent)
    print("句子加载完毕")
    properties = list(properties)
    docs = list(docs)

    sents = properties + docs
    len_filter_cnt = 0
    new_sents = []
    word2id = json.load(open('../../GLRE/prepro_data/DocRED/processed/word2id_vec.json', 'r', encoding='utf-8'))
    for sent in sents:
        if len(sent.split()) > 128 or len(sent.split()) <= 0:
            len_filter_cnt += 1
            continue
        tokens = sent.lower().replace(':', ' ').replace(',', ' ').replace('.', ' ').split()
        flag = False
        for token in tokens:  # 把存在unknown删除
            if token not in word2id:
                flag = True
        if flag:
            continue

        new_sents.append(sent)
    print("总句子数量是%d， 长度超过128数量是%d" % (len(new_sents), len_filter_cnt))

    # 按句子长度分桶，然后划分训练集和验证集
    size_counter = Counter()
    for sent in new_sents:
        tokens = sent.lower().replace(':', ' ').replace(',', ' ').replace('.', ' ').split()
        sent_size = len(tokens)
        assert sent_size <= 128

        # keep track of different size bins, with bins for
        # 1-10, 11-20, 21-30, etc
        top_bin = int(math.ceil(sent_size / 10) * 10)
        size_counter[top_bin] += 1

    print('Converting word to indices...')
    train_data = {}
    dev_data = {}
    train_sents = []
    dev_sents = []

    for threshold in size_counter:
        min_threshold = threshold - 9
        num_sentences = size_counter[threshold]
        print('Converting %d sentences with length between %d and %d'
              % (num_sentences, min_threshold, threshold))
        sents, sizes, ori_sents = create_sentence_matrix(new_sents, num_sentences,
                                              min_threshold, threshold, word2id)

        # shuffle sentences and sizes with the sime RNG state
        state = np.random.get_state()
        np.random.shuffle(sents)
        np.random.set_state(state)
        np.random.shuffle(sizes)
        np.random.set_state(state)
        np.random.shuffle(ori_sents)

        ind = int(len(sents) * valid_proportion)
        valid_sentences = sents[:ind]
        valid_sizes = sizes[:ind]
        train_sentences = sents[ind:]
        train_sizes = sizes[ind:]
        dev_sents.extend(ori_sents[:ind])
        train_sents.extend(ori_sents[ind:])

        train_data['sentences-%d' % threshold] = train_sentences
        train_data['sizes-%d' % threshold] = train_sizes
        dev_data['sentences-%d' % threshold] = valid_sentences
        dev_data['sizes-%d' % threshold] = valid_sizes

    with open('./ae_results/train.txt', 'w', encoding='utf-8') as f:
        for line in train_sents:
            f.write(line + '\n')

    with open('./ae_results/dev.txt', 'w', encoding='utf-8') as f:
        for line in dev_sents:
            f.write(line + '\n')

    np.savez('./ae_results/train-data.npz', **train_data)
    np.savez('./ae_results/dev-data.npz', **dev_data)

# prepare_data()
# train()
test()

# 句子加载完毕
# 总句子数量是18686518， 长度超过128数量是3
# Converting word to indices...
# Converting 14491362 sentences with length between 1 and 10
# Converting 3913062 sentences with length between 11 and 20
# Converting 266825 sentences with length between 21 and 30
# Converting 12326 sentences with length between 31 and 40
# Converting 2043 sentences with length between 41 and 50
# Converting 595 sentences with length between 51 and 60
# Converting 23 sentences with length between 81 and 90
# Converting 189 sentences with length between 61 and 70
# Converting 72 sentences with length between 71 and 80
# Converting 16 sentences with length between 91 and 100
# Converting 3 sentences with length between 101 and 110
# Converting 2 sentences with length between 111 and 120
