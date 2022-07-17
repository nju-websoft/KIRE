import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from knowledge_injection_layer.coref_injection import Coref_Injection
from knowledge_injection_layer.config import Config
from knowledge_injection_layer.modules import Combineloss
from knowledge_injection_layer.kg_injection import Kg_Injection

class CNN3(nn.Module):
	def __init__(self, config):
		super(CNN3, self).__init__()
		self.config = config
		self.word_emb = nn.Embedding(config.data_word_vec.shape[0], config.data_word_vec.shape[1])
		self.word_emb.weight.data.copy_(torch.from_numpy(config.data_word_vec))
		self.word_emb.weight.requires_grad = False


		# self.char_emb = nn.Embedding(config.data_char_vec.shape[0], config.data_char_vec.shape[1])
		# self.char_emb.weight.data.copy_(torch.from_numpy(config.data_char_vec))
		# char_dim = config.data_char_vec.shape[1]
		# char_hidden = 100
		# self.char_cnn = nn.Conv1d(char_dim,  char_hidden, 5)

		self.coref_embed = nn.Embedding(config.max_length, config.coref_size, padding_idx=0)
		self.ner_emb = nn.Embedding(config.ner_size, config.entity_type_size, padding_idx=0)

		input_size = config.data_word_vec.shape[1] + config.coref_size + config.entity_type_size #+ char_hidden

		self.out_channels = 200
		self.in_channels = input_size

		self.kernel_size = 3
		self.stride = 1
		self.padding = int((self.kernel_size - 1) / 2)

		self.cnn_1 = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
		self.cnn_2 = nn.Conv1d(self.out_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
		self.cnn_3 = nn.Conv1d(self.out_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
		self.max_pooling = nn.MaxPool1d(self.kernel_size, stride=self.stride, padding=self.padding)
		self.relu = nn.ReLU()

		self.dropout = nn.Dropout(config.cnn_drop_prob)

		self.bili = torch.nn.Bilinear(self.out_channels+config.dis_size, self.out_channels+config.dis_size, config.relation_num)
		self.dis_embed = nn.Embedding(20, config.dis_size, padding_idx=10)

		# Knowledge_injection_layer
		if Config.add_kg_flag:
			self.kg_injection = Kg_Injection(torch.from_numpy(config.data_word_vec), Config.kg_freeze_words,
											 Config.ent_hidden_dim, Config.gpuid,
											 Config.gcn_layer_nums, Config.gcn_in_drop,
											 Config.gcn_out_drop, Config.hidden_dim,
											 Config.kg_intermediate_size,
											 Config.kg_num_attention_heads,
											 Config.kg_attention_probs_dropout_prob,
											 Config.adaption_type, Config.kg_align_loss, Config.gcn_type)
		print("kg_injection")
		if Config.add_coref_flag:
			if Config.coref_place == 'afterRnn':
				self.coref_injection = Coref_Injection(self.out_channels)
			elif Config.coref_place == 'afterWordvec1':
				self.coref_injection = Coref_Injection(input_size)
			else:
				self.coref_injection = Coref_Injection(Config.hidden_dim)
		print("coref_injection")
		self.combineloss = Combineloss(Config)
		if Config.add_kg_flag or Config.add_coref_flag:
			# self.kg_linear_transfer = nn.Linear(params['lstm_dim']*2, params['lstm_dim'])
			self.kg_linear_transfer = nn.Sequential(nn.Linear(self.out_channels * 2, self.out_channels),
													nn.ReLU(),
													nn.Linear(self.out_channels, self.out_channels))
		print("kg_linear_transfer")


	def forward(self, context_idxs, pos, context_ner, context_char_idxs, context_lens, h_mapping, t_mapping, relation_mask, dis_h_2_t, dis_t_2_h,
				kg_ent_attrs, kg_ent_attr_nums, kg_ent_attr_lens, kg_adj, kg_adj_edges, kg_ent_labels, kg_ent_mask,
				coref_h_mapping, coref_t_mapping, coref_lens, coref_mention_position, coref_label, coref_label_mask
				):
		# para_size, char_size, bsz = context_idxs.size(1), context_char_idxs.size(2), context_idxs.size(0)
		# context_ch = self.char_emb(context_char_idxs.contiguous().view(-1, char_size)).view(bsz * para_size, char_size, -1)
		# context_ch = self.char_cnn(context_ch.permute(0, 2, 1).contiguous()).max(dim=-1)[0].view(bsz, para_size, -1)

		sent = self.word_emb(context_idxs)
		if Config.add_kg_flag or Config.add_coref_flag:
			sent_kg = sent.detach().clone()

		# Knowledge Injection layer
		'''======加入共指信息=======(更新指代词表示)'''
		loss_coref = None
		if Config.add_coref_flag and Config.coref_place == 'afterWordvec0':
			sent_kg, loss_coref = self.coref_injection(coref_h_mapping, coref_t_mapping,
													coref_lens, sent_kg,
													coref_mention_position, coref_label,
													coref_label_mask)

		'''======加入kg信息======（更新实体mention表示）'''
		loss_kg = None
		if Config.add_kg_flag:
			context_masks = context_idxs.clone().bool()
			context_masks[context_masks > 0] = 1
			# print(context_masks.size())
			doc_rep = torch.max(sent_kg, dim=1)[0]
			sent_kg, loss_kg = self.kg_injection(kg_ent_attrs, kg_ent_attr_nums,
											  kg_ent_attr_lens,
											  kg_adj, kg_ent_labels,
											  sent_kg, doc_rep, context_masks, kg_ent_mask,
											  kg_ent_labels, kg_adj_edges)


		sent_pos = self.coref_embed(pos)
		sent_ner = self.ner_emb(context_ner)
		sent = torch.cat([sent, sent_pos, sent_ner], dim=-1)

		if Config.add_kg_flag or Config.add_coref_flag:
			sent_pos_kg = sent_pos.detach().clone()
			sent_ner_kg = sent_ner.detach().clone()
			sent_kg = torch.cat([sent_kg, sent_pos_kg, sent_ner_kg], dim=-1)

		if Config.add_coref_flag and Config.coref_place == 'afterWordvec1':
			sent_kg, loss_coref = self.coref_injection(coref_h_mapping, coref_t_mapping,
														   coref_lens, sent_kg,
														   coref_mention_position, coref_label,
														   coref_label_mask)

		sent = sent.permute(0, 2, 1)
		if Config.add_kg_flag or Config.add_coref_flag:
			sent_kg = sent_kg.permute(0, 2, 1)

		# batch * embedding_size * max_len
		x = self.cnn_1(sent)
		x = self.max_pooling(x)
		x = self.relu(x)
		x = self.dropout(x)

		x = self.cnn_2(x)
		x = self.max_pooling(x)
		x = self.relu(x)
		x = self.dropout(x)

		x = self.cnn_3(x)
		x = self.max_pooling(x)
		x = self.relu(x)
		x = self.dropout(x)

		context_output = x.permute(0, 2, 1)

		if Config.add_kg_flag or Config.add_coref_flag:
			x = self.cnn_1(sent_kg)
			x = self.max_pooling(x)
			x = self.relu(x)
			x = self.dropout(x)

			x = self.cnn_2(x)
			x = self.max_pooling(x)
			x = self.relu(x)
			x = self.dropout(x)

			x = self.cnn_3(x)
			x = self.max_pooling(x)
			x = self.relu(x)
			x = self.dropout(x)

			context_output_kg = x.permute(0, 2, 1)

		if Config.add_coref_flag and Config.coref_place == 'afterRnn':
			context_output_kg, loss_coref = self.coref_injection(coref_h_mapping, coref_t_mapping,
														   coref_lens, context_output_kg,
														   coref_mention_position, coref_label,
														   coref_label_mask)
		if Config.add_kg_flag or Config.add_coref_flag:
			# context_output = torch.mean(torch.stack([context_output, context_output_kg]), dim=0)
			context_output = self.kg_linear_transfer(torch.cat([context_output, context_output_kg], dim=-1))
		start_re_output = torch.matmul(h_mapping, context_output)
		end_re_output = torch.matmul(t_mapping, context_output)

		s_rep = torch.cat([start_re_output, self.dis_embed(dis_h_2_t)], dim=-1)
		t_rep = torch.cat([end_re_output, self.dis_embed(dis_t_2_h)], dim=-1)

		predict_re = self.bili(s_rep, t_rep)

		return predict_re, loss_kg, loss_coref
