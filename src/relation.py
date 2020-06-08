import torch


# q - question, c - column, t - table
RELATION_LIST = [
    "[PAD]",  # 0
    "cc_identical",
    "cc_sibling",
    "cc_foreign_primary",
    "cc_primary_foreign",
    "cc_neighbor",
    "cc_cnt_col",
    "cc_col_cnt",
    "cc_etc",  # 8
    "ct_primary_child",
    "ct_child",
    "ct_etc",  # 11
    "tc_primary_child",
    "tc_child",
    "tc_etc",  # 14
    "tt_identical",
    "tt_foreign",
    "tt_reversed",
    "tt_both",
    "tt_etc",  # 19
    "qt_exact",
    "qt_partial",
    "qt_no",
    "qc_exact",
    "qc_partial",
    "qc_no",  # 25
    "tq_exact",
    "tq_partial",
    "tq_no",
    "cq_exact",
    "cq_partial",
    "cq_no",  # 31
    "qq_-2",
    "qq_-1",
    "qq_0",
    "qq_1",
    "qq_2",  # 36
    # 0: identical #1: Dependent #2: Dependent Reversed #3: ETC
    "identical",
    "depen",
    "depen_rev",
    "no_dep",  # 40
    "qc_db",
    "cq_db",  # 42
    # dependency parsing
    "cc",  # 43
    "number",
    "ccomp",
    "possessive",
    "prt",
    "num",
    "nsubjpass",
    "csubj",
    "conj",
    "dobj",
    "nn",
    "neg",
    "discourse",
    "mark",
    "auxpass",
    "infmod",
    "mwe",
    "advcl",
    "aux",
    "prep",
    "parataxis",
    "nsubj",
    "rcmod",
    "advmod",
    "punct",
    "quantmod",
    "tmod",
    "acomp",
    "pcomp",
    "poss",
    "npadvmod",
    "xcomp",
    "cop",
    "partmod",
    "dep",
    "appos",
    "det",
    "amod",
    "pobj",
    "iobj",
    "expl",
    "predet",
    "preconj",
    "root",
    # Reverse
    "cc_rev",  # 87
    "number_rev",
    "ccomp_rev",
    "possessive_rev",
    "prt_rev",
    "num_rev",
    "nsubjpass_rev",
    "csubj_rev",
    "conj_rev",
    "dobj_rev",
    "nn_rev",
    "neg_rev",
    "discourse_rev",
    "mark_rev",
    "auxpass_rev",
    "infmod_rev",
    "mwe_rev",
    "advcl_rev",
    "aux_rev",
    "prep_rev",
    "parataxis_rev",
    "nsubj_rev",
    "rcmod_rev",
    "advmod_rev",
    "punct_rev",
    "quantmod_rev",
    "tmod_rev",
    "acomp_rev",
    "pcomp_rev",
    "poss_rev",
    "npadvmod_rev",
    "xcomp_rev",
    "cop_rev",
    "partmod_rev",
    "dep_rev",
    "appos_rev",
    "det_rev",
    "amod_rev",
    "pobj_rev",
    "iobj_rev",
    "expl_rev",
    "predet_rev",
    "preconj_rev",
    "root_rev",  # 130
]

# Dictionary
RELATION_TYPE = {key: idx for idx, key in enumerate(RELATION_LIST)}
N_RELATIONS = len(RELATION_TYPE)


def create_relation(cfg, data, dbs, use_col_set=True):
    """
    * has not been considered in depths
    """

    USE_DEP = cfg.use_dep

    db = dbs[data["db_id"]]

    tokens = data["question_arg"]
    token_types = data["question_arg_type"]
    table = data["table_names"]

    assert len(tokens) == len(token_types)
    fp_relation = create_fp_relation(db)
    # construct col set and its mapping dic
    col_mapping = {}
    column = []
    for idx, (tab_id, col_name) in enumerate(db["column_names"]):
        if not use_col_set or col_name not in [item[1] for item in column]:
            col_mapping[idx] = len(column)
            column += [[[tab_id], col_name]]
        else:
            new_col_idx = [item[1] for item in column].index(col_name)
            col_mapping[idx] = new_col_idx
            column[new_col_idx][0] += [tab_id]

    # translate p/f keys
    primary_key = [col_mapping[item] for item in db["primary_keys"]]
    foreign_key = [
        [col_mapping[item[0]], col_mapping[item[1]]] for item in db["foreign_keys"]
    ]

    # Split words
    column_name = split_words([item[1] for item in column])
    table = split_words(table)

    # Sen - Sen
    question_relations = (
        data["question_relation"] if "question_relation" in data else None
    )
    qq_relation = parse_q_q_relation(USE_DEP, data["question_arg"], question_relations)

    # Sen & Col
    qc_relation = parse_match_relation(tokens, column_name, "q", "c")
    qc_relation = append_db_content_relation("q", qc_relation, column_name, token_types)
    cq_relation = parse_match_relation(column_name, tokens, "c", "q")
    cq_relation = append_db_content_relation("c", cq_relation, column_name, token_types)

    # Sen & Tab
    qt_relation = parse_match_relation(tokens, table, "q", "t")
    tq_relation = parse_match_relation(table, tokens, "t", "q")

    # Col - Col
    cc_relation = parse_c_c_relation(column, foreign_key, fp_relation)

    # Col - Tab
    ct_relation = parse_c_t_relation(table, column, primary_key)
    tc_relation = parse_t_c_relation(table, column, primary_key)

    # Tab - Tab
    tt_relation = parse_t_t_relation(fp_relation, table)

    # Append relation info to data
    relations = {
        "qq": qq_relation,
        "qc": qc_relation,
        "cq": cq_relation,
        "qt": qt_relation,
        "tq": tq_relation,
        "cc": cc_relation,
        "ct": ct_relation,
        "tc": tc_relation,
        "tt": tt_relation,
    }

    data["relation"] = relations

    return data


def create_batch(relations):
    qq = [relation["qq"] for relation in relations]
    qc = [relation["qc"] for relation in relations]
    cq = [relation["cq"] for relation in relations]
    qt = [relation["qt"] for relation in relations]
    tq = [relation["tq"] for relation in relations]
    cc = [relation["cc"] for relation in relations]
    ct = [relation["ct"] for relation in relations]
    tc = [relation["tc"] for relation in relations]
    tt = [relation["tt"] for relation in relations]

    # Max lens
    b_len = len(qq)
    max_q_len = max([len(item) for item in qq])
    max_c_len = max([len(item) for item in cc])
    max_t_len = max([len(item) for item in tt])

    # Spliting indices
    q_idx = max_q_len
    c_idx = q_idx + max_c_len
    t_idx = c_idx + max_t_len

    relation_matrix = torch.zeros(b_len, t_idx, t_idx).long()

    # Fill in the matrix
    fill_matrix(relation_matrix, 0, 0, qq)
    fill_matrix(relation_matrix, 0, q_idx, qc)
    fill_matrix(relation_matrix, q_idx, 0, cq)
    fill_matrix(relation_matrix, 0, c_idx, qt)
    fill_matrix(relation_matrix, c_idx, 0, tq)
    fill_matrix(relation_matrix, q_idx, q_idx, cc)
    fill_matrix(relation_matrix, q_idx, c_idx, ct)
    fill_matrix(relation_matrix, c_idx, q_idx, tc)
    fill_matrix(relation_matrix, c_idx, c_idx, tt)

    return relation_matrix.cuda()


def fill_matrix(matrix, x_start_idx, y_start_idx, relations):
    for idx, relation in enumerate(relations):
        x_end_idx = x_start_idx + len(relation)
        y_end_idx = y_start_idx + len(relation[0])
        matrix[idx, x_start_idx:x_end_idx, y_start_idx:y_end_idx] = torch.tensor(
            relation
        ).long()
    return matrix


def parse_c_t_relation(tables, columns, primary_keys):
    relations = []
    for col_idx, (par_tab_ids, col_name) in enumerate(columns):
        tmp = []
        for tab_id, tab_name in enumerate(tables):
            if tab_id in par_tab_ids:
                if col_idx in primary_keys:
                    key = "ct_primary_child"
                else:
                    key = "ct_child"
            else:
                key = "ct_etc"
            tmp += [RELATION_TYPE[key]]
        relations += [tmp]
    return relations


def parse_t_c_relation(tables, columns, primary_keys):
    relations = []
    for tab_id, tab_name in enumerate(tables):
        tmp = []
        for col_idx, (par_tab_ids, col_name) in enumerate(columns):
            if tab_id in par_tab_ids:
                if col_idx in primary_keys:
                    key = "tc_primary_child"
                else:
                    key = "tc_child"
            else:
                key = "tc_etc"
            tmp += [RELATION_TYPE[key]]
        relations += [tmp]
    return relations


def parse_t_t_relation(fp_relations, table_names):
    relations = []
    tab_len = len(table_names)
    for idx_1 in range(tab_len):
        tmp = []
        for idx_2 in range(tab_len):
            is_fp = [idx_1, idx_2] in fp_relations
            is_pf = [idx_2, idx_1] in fp_relations
            if idx_1 == idx_2:
                key = "tt_identical"
            elif is_fp and is_pf:
                key = "tt_both"
            elif is_fp:
                key = "tt_foreign"
            elif is_pf:
                key = "tt_reversed"
            else:
                key = "tt_etc"
            tmp += [RELATION_TYPE[key]]
        relations += [tmp]
    return relations


def parse_c_c_relation(cols, foreign_keys, fp_relations):
    relations = []
    for col_idx_1, (tab_ids_1, col_1) in enumerate(cols):
        tmp = []
        for col_idx_2, (tab_ids_2, col_2) in enumerate(cols):
            if is_partial_match(tab_ids_1, tab_ids_2):
                if col_1 == col_2:
                    key = "cc_identical"
                else:
                    key = "cc_sibling"
            elif [col_idx_1, col_idx_2] in foreign_keys:
                key = "cc_foreign_primary"
            elif [col_idx_2, col_idx_1] in foreign_keys:
                key = "cc_primary_foreign"
            elif compare_fp_relation(tab_ids_1, tab_ids_2, fp_relations):
                key = "cc_neighbor"
            elif col_1 == "*":
                key = "cc_cnt_col"
            elif col_2 == "*":
                key = "cc_col_cnt"
            else:
                key = "cc_etc"
            tmp += [RELATION_TYPE[key]]
        relations += [tmp]
    return relations


def parse_q_q_relation(USE_DEP, sentence, relation_matrix):
    USE_SELECTED_DEP = False
    SELECTED_RELATIONS = ["amod", "prep", "cc", "conj", "pobj"]
    SELECTED_RELATIONS += [item + "_rev" for item in SELECTED_RELATIONS]

    qq_relations = []
    if USE_SELECTED_DEP:
        # Default
        question_length = len(sentence)
        for idx_1 in range(question_length):
            tmp = []
            for idx_2 in range(question_length):
                key = "qq_" + str(max(min(idx_1 - idx_2, 2), -2))
                tmp += [RELATION_TYPE[key]]
            qq_relations += [tmp]

        # Change positional to dependency info
        assert relation_matrix, "Empty relation matrix"
        for idx_1, relations in enumerate(relation_matrix):
            for idx_2, relation in enumerate(relations):
                if relation in SELECTED_RELATIONS:
                    qq_relations[idx_1][idx_2] = RELATION_TYPE[relation]

    elif USE_DEP:
        assert relation_matrix, "Empty relation matrix"
        for relations in relation_matrix:
            qq_relations += [[RELATION_TYPE[relation] for relation in relations]]
    else:
        question_length = len(sentence)
        for idx_1 in range(question_length):
            tmp = []
            for idx_2 in range(question_length):
                key = "qq_" + str(max(min(idx_1 - idx_2, 2), -2))
                tmp += [RELATION_TYPE[key]]
            qq_relations += [tmp]
    return qq_relations


def append_db_content_relation(
    first_symbol, relation_matrix, column_names, token_types
):
    for q_idx, type in enumerate(token_types):
        # If token is db content
        if type[0] == "db":
            target_column_name = type[1:]
            for c_idx, column_name in enumerate(column_names):
                if column_name == target_column_name:
                    # Fix relation type
                    if first_symbol == "q":
                        relation_matrix[q_idx][c_idx] = RELATION_TYPE["qc_db"]
                    else:
                        relation_matrix[c_idx][q_idx] = RELATION_TYPE["cq_db"]
    return relation_matrix


def parse_match_relation(words_1, words_2, type_1, type_2):
    relations = []
    for idx_1, word_1 in enumerate(words_1):
        tmp = []
        for idx_2, word_2 in enumerate(words_2):
            if word_1 == ["*"]:
                word_1 = ["count", "many", "number"]
            if word_2 == ["*"]:
                word_2 = ["count", "many", "number"]
            if is_exact_match(word_1, word_2):
                key = "{}{}_exact".format(type_1, type_2)
            elif is_partial_match(word_1, word_2):
                key = "{}{}_partial".format(type_1, type_2)
            else:
                key = "{}{}_no".format(type_1, type_2)
            tmp += [RELATION_TYPE[key]]
        relations += [tmp]
    return relations


def is_exact_match(words_1, words_2):
    return words_1 == words_2


def is_partial_match(words_1, words_2):
    for word in words_1:
        if word in words_2:
            return True
    return False


def split_words(words):
    return [word.split(" ") for word in words]


def create_fp_relation(db):
    """
    in the respective of table id
    """
    fp_relations = []
    col_to_tab = {idx: item[0] for idx, item in enumerate(db["column_names"])}
    for col_id_1, col_id_2 in db["foreign_keys"]:
        fp_relations += [[col_to_tab[col_id_1], col_to_tab[col_id_2]]]
    return fp_relations


def compare_fp_relation(tab_ids_1, tab_ids_2, fp_relations):
    for tab_id_1 in tab_ids_1:
        for tab_id_2 in tab_ids_2:
            if [tab_id_1, tab_id_2] in fp_relations or [
                tab_id_2,
                tab_id_1,
            ] in fp_relations:
                return True
    return False
