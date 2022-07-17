import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from knowledge_injection_layer.config import Config

def process_long_input(model, input_ids, attention_mask, subword_indexs, start_tokens, end_tokens,
                       kg_injection, kg_ent_attrs, kg_ent_attr_nums, kg_ent_attr_lens, kg_adj, kg_adj_edges, kg_ent_labels, kg_ent_mask,
                       coref_injection, coref_h_mapping, coref_t_mapping, coref_dis, coref_lens, coref_mention_position, coref_label, coref_label_mask,
                       ):
    # Split the input to 2 overlapping chunks. Now BERT can encode inputs of which the length are up to 1024.
    n, c = input_ids.size()
    doc_lens = [len(x) for x in subword_indexs]

    bert_starts = torch.zeros((n,c)).to(input_ids)
    for i in range(n):
        token_start_idxs = [x + 1 for x in subword_indexs[i]] # np.cumsum([0] + subword_lengths[i][:-1])
        for x in token_start_idxs:
            if x < c:
                bert_starts[i, x] = 1
            else:
                print(x, c)

    sequence_output, attention, sequence_output_kg, loss_kg = bert_forward(
        model, kg_injection, input_ids, attention_mask, start_tokens, end_tokens,
        bert_starts, kg_ent_attrs, kg_ent_attr_nums, kg_ent_attr_lens, kg_adj, kg_adj_edges, kg_ent_labels, kg_ent_mask
    )

    loss_coref = None
    if Config.add_coref_flag:
        word_vec_kg = [layer[starts.nonzero().squeeze(1)] for layer, starts in
                       zip(sequence_output_kg, bert_starts)]
        word_vec_kg = pad_sequence(word_vec_kg, batch_first=True, padding_value=0)

        sequence_output_kg, loss_coref = coref_injection(coref_h_mapping, coref_t_mapping, coref_dis,
                                                             coref_lens, word_vec_kg,
                                                             coref_mention_position,
                                                             coref_label,
                                                             coref_label_mask)
        embedding_output_kg_result = []
        for i in range(n):
            layer = word_vec_kg[i]
            starts = bert_starts[i]
            doc_len = doc_lens[i]

            starts = torch.cumsum(starts, dim=0) - 1
            new_layer = torch.cat((layer[0].unsqueeze(0), sequence_output_kg[i][starts[1:]]), dim=0)
            new_layer[doc_len:].zero_()
            embedding_output_kg_result.append(new_layer)
        sequence_output_kg = torch.stack(embedding_output_kg_result, dim=0)

    if Config.add_kg_flag or Config.add_coref_flag:
        sequence_output = torch.max(torch.stack([sequence_output, sequence_output_kg]), dim=0)[0]

    return sequence_output, attention, loss_kg, loss_coref


def _re_cz(num_seg, seq_len, c, context_output, attention, attention_mask, len_start, len_end):
        i = 0
        re_context_output = []
        re_attention = []
        for n_seg, l_i in zip(num_seg, seq_len):
            if l_i <= 512:
                assert n_seg == 1
                if c <= 512:
                    re_context_output.append(context_output[i])
                    re_attention.append(attention[i])
                else:
                    context_output1 = F.pad(context_output[i, :512, :], (0, 0, 0, c-512))
                    re_context_output.append(context_output1)
                    attention1 = F.pad(attention[i][:, :512, :512], (0, c-512, 0, c-512))
                    re_attention.append(attention1)
            else:
                context_output1 = []
                attention1 = None
                mask1 = []
                for j in range(i, i + n_seg - 1):
                    if j == i:
                        context_output1.append(context_output[j][:512 - len_end, :])
                        attention1 = F.pad(attention[j][:, :512-len_end, :512-len_end], (0, c-(512-len_end), 0, c-(512-len_end)))
                        mask1.append(attention_mask[j][:512 - len_end])
                    else:
                        context_output1.append(context_output[j][len_start:512 - len_end, :])
                        attention2 = F.pad(attention[j][:, len_start:512-len_end, len_start:512-len_end],
                                                        (512-len_end+(j-i-1)*(512-len_end-len_start), c-(512-len_end+(j-i)*(512-len_end-len_start)),
                                                         512-len_end+(j-i-1)*(512-len_end-len_start), c-(512-len_end+(j-i)*(512-len_end-len_start))))
                        if attention1 is None:
                            attention1 = attention2
                        else:
                            attention1 = attention1 + attention2
                        mask1.append(attention_mask[j][len_start:512 - len_end])

                context_output1 = F.pad(torch.cat(context_output1, dim=0),
                                            (0, 0, 0, c - (n_seg - 1) * (512 - len_end) + (n_seg - 2) * len_start))
                att = attention1 + F.pad(attention[i + n_seg - 1][:, len_start:, len_start:],
                                         (l_i - 512 + len_start, c - l_i, l_i - 512 + len_start, c - l_i))

                context_output2 = context_output[i + n_seg - 1][len_start:]
                context_output2 = F.pad(context_output2, (0, 0, l_i - 512 + len_start, c - l_i))

                mask1 = F.pad(torch.cat(mask1, dim=0), (0, c - (n_seg - 1) * (512 - len_end) + (n_seg - 2) * len_start))
                mask2 = attention_mask[i + n_seg - 1][len_start:]
                mask2 = F.pad(mask2, (l_i - 512 + len_start, c - l_i))
                mask = mask1 + mask2 + 1e-10
                context_output1 = (context_output1 + context_output2) / mask.unsqueeze(-1)
                re_context_output.append(context_output1)
                att = att / (att.sum(-1, keepdim=True) + 1e-10)
                re_attention.append(att)

            i += n_seg
        attention = torch.stack(re_attention, dim=0)
        context_output = torch.stack(re_context_output, dim=0)
        return context_output, attention


def bert_forward(model, kg_injection, input_ids, attention_mask, start_tokens, end_tokens, bert_starts, kg_ent_attrs, kg_ent_attr_nums, kg_ent_attr_lens, kg_adj, kg_adj_edges, kg_ent_labels, kg_ent_mask):
    n, c = input_ids.size()
    start_tokens = torch.tensor(start_tokens).to(input_ids)
    end_tokens = torch.tensor(end_tokens).to(input_ids)
    len_start = start_tokens.size(0)
    len_end = end_tokens.size(0)

    embedding_output = model.embeddings.word_embeddings(input_ids)
    loss_kg = None
    if Config.add_kg_flag or Config.add_coref_flag:
        context_masks = [mask[starts.nonzero().squeeze(1)] for mask, starts in
                            zip(attention_mask, bert_starts)]
        context_masks = pad_sequence(context_masks, batch_first=True, padding_value=0)

        embedding_output_kg = embedding_output.detach().clone()

        if Config.add_kg_flag:
            word_vec_kg = [layer[starts.nonzero().squeeze(1)] for layer, starts in zip(embedding_output_kg, bert_starts)]
            word_vec_kg = pad_sequence(word_vec_kg, batch_first=True, padding_value=0)

            context_masks = context_masks.bool()
            word_vec_kg, loss_kg = kg_injection(kg_ent_attrs, kg_ent_attr_nums,
                                                kg_ent_attr_lens,
                                                kg_adj, kg_ent_labels,
                                                word_vec_kg, context_masks, kg_ent_mask,
                                                kg_ent_labels, kg_adj_edges)

            batch_size = embedding_output_kg.size(0)
            embedding_output_kg_result = []
            for i in range(batch_size):
                layer = embedding_output_kg[i]
                starts = bert_starts[i]

                starts = torch.cumsum(starts, dim=0) - 1
                new_layer = torch.cat((layer[0].unsqueeze(0), word_vec_kg[i][starts[1:]]), dim=0)
                embedding_output_kg_result.append(new_layer)

            embedding_output_kg = torch.stack(embedding_output_kg_result, dim=0)

    new_input_ids, new_attention_mask, num_seg = [], [], []
    if Config.add_kg_flag or Config.add_coref_flag:
        new_input_ids_kg = []
    seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
    for i, l_i in enumerate(seq_len):
        bert_indexs = [(bert_s, min(l_i - len_end, bert_s + 512-(len_start+len_end))) for bert_s in
                        range(len_start, l_i - len_end, 512-(len_start+len_end))]
        num_seg.append(len(bert_indexs))
        for j, (bert_s, bert_e) in enumerate(bert_indexs):
            if j == len(bert_indexs) - 1:
                if l_i <= 512:
                    new_input_ids.append(torch.cat([model.embeddings.word_embeddings(start_tokens),
                                                    embedding_output[i, len_start: min(512-len_end, c-len_end)],
                                                    model.embeddings.word_embeddings(end_tokens)],
                                                    dim=0))
                    if Config.add_kg_flag or Config.add_coref_flag:
                        new_input_ids_kg.append(
                                    torch.cat([model.embeddings.word_embeddings(start_tokens),
                                               embedding_output_kg[i, len_start: min(512-len_end, c-len_end)],
                                               model.embeddings.word_embeddings(end_tokens)], dim=0))
                    new_attention_mask.append(attention_mask[i, :512])

                else:
                    new_input_ids.append(torch.cat([model.embeddings.word_embeddings(start_tokens),
                                                    embedding_output[i, bert_e - 512 + len_start + len_end: bert_e],
                                                    model.embeddings.word_embeddings(end_tokens)],
                                                    dim=0))
                    if Config.add_kg_flag or Config.add_coref_flag:
                        new_input_ids_kg.append(
                                    torch.cat([model.embeddings.word_embeddings(start_tokens),
                                               embedding_output_kg[i, (bert_e - 512 + len_start + len_end): bert_e],
                                               model.embeddings.word_embeddings(end_tokens)], dim=0))
                    new_attention_mask.append(attention_mask[i, bert_e - 512 + len_end:bert_e + len_end])
            else:
                new_input_ids.append(torch.cat([model.embeddings.word_embeddings(start_tokens),
                                                embedding_output[i, bert_s: bert_e],
                                                model.embeddings.word_embeddings(end_tokens)],
                                                dim=0))
                if Config.add_kg_flag or Config.add_coref_flag:
                    new_input_ids_kg.append(
                                torch.cat([model.embeddings.word_embeddings(start_tokens),
                                           embedding_output_kg[i, bert_s: bert_e],
                                           model.embeddings.word_embeddings(end_tokens)], dim=0))
                new_attention_mask.append(attention_mask[i, bert_s - len_start:bert_e + len_end])

    embedding_output = torch.stack(new_input_ids, dim=0)
    attention_mask = torch.stack(new_attention_mask, dim=0)
    if Config.add_kg_flag or Config.add_coref_flag:
        embedding_output_kg = torch.stack(new_input_ids_kg)
    # print(embedding_output.size())
    # print(attention_mask.size())
    output = model(attention_mask=attention_mask, inputs_embeds=embedding_output)
    sequence_output = output[0]
    attention = output[-1][-1]

    sequence_output, attention = _re_cz(num_seg, seq_len, c, sequence_output, attention, attention_mask, len_start, len_end)

    if Config.add_kg_flag or Config.add_coref_flag:
        output = model(attention_mask=attention_mask, inputs_embeds=embedding_output_kg)
        sequence_output_kg = output[0]
        attention_kg = output[-1][-1]
        sequence_output_kg, _ = _re_cz(num_seg, seq_len, c, sequence_output_kg, attention_kg, attention_mask, len_start, len_end)
    else:
        sequence_output_kg = None


    if Config.add_kg_flag or Config.add_coref_flag:
        return sequence_output, attention, sequence_output_kg, loss_kg
    else:
        return sequence_output, attention, None, None
