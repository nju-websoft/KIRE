

class Config(object):
        hidden_dim = 256  # 输入的doc_hidden的表示 == params['lstm_dim']
        kg_lr = 0.001
        other_lr = 0.00001
        #l2_reg = 0.0001

        ## for kgs
        kg_freeze_words = True
        adaption_type = 'loss'  # gate loss
        kg_align_loss = True
        ent_hidden_dim = 100  # 输出的属性表示/实体表示的维度
        rel_hidden_dim = 100
        gcn_in_drop = 0.0
        gcn_out_drop = 0.0
        gcn_layer_nums = 0
        gcn_head_nums = 1
        gcn_type = 'GCN'

        kg_num_attention_heads = 2
        kg_attention_probs_dropout_prob = 0.0
        kg_intermediate_size = 256
        kg_hidden_dropout_prob = 0.0
        kg_num_hidden_layers = 1

        ## 模型组件控制
        attr_encode_lstm = False
        attr_encode_type = "max"  # auto
        add_kg_flag = True
        add_coref_flag = False
        gpuid = 0

        loss_combine = 'linear'
        alpha_re = 1.0
        alpha_coref = 0.1
        alpha_kg = 0.1

        coref_place = 'afterWordvec0'  # ['afterWordvec0', 'afterWordvec1', 'afterRnn']

        train_method = "two_step"  # [one_step, two_step]
        onlywc = False

        coref_kl_loss = True
        attr_module = True
        relation_module = True
        attr_label_flag = True
        attr_description_flag = True
        attr_instance_flag = True
        attr_alias_flag = True
        re_train = False
        re_train_path = ""
        kg_transfer = 'max'
        relation_type = 'all'
        coref_type = 'all'

def load_config_file(file_name):
    pass
