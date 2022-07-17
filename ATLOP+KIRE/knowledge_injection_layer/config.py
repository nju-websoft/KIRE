

class Config(object):
        hidden_dim = 256
        kg_lr = 0.001
        other_lr = 0.00001

        ent_hidden_dim = 100
        rel_hidden_dim = 100
        gcn_in_drop = 0.0
        gcn_out_drop = 0.0
        gcn_layer_nums = 0
        gcn_head_nums = 1

        kg_num_attention_heads = 2
        kg_attention_probs_dropout_prob = 0.0
        kg_intermediate_size = 256
        kg_hidden_dropout_prob = 0.0
        kg_num_hidden_layers = 1

        add_kg_flag = True
        add_coref_flag = False

        alpha_re = 1.0
        alpha_coref = 0.1
        alpha_kg = 0.1

        re_train = False
        re_train_path = ""
        train_method = "one_step"


def load_config_file(file_name):
    pass
