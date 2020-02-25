import json


def create_qgm_action(qgm_boxes):
    # Simple query only
    qgm_box = qgm_boxes[0]

    actions = "B({}) ".format(0 if qgm_box["body"]["local_predicates"] else 1)

    # Q
    q_len = len(qgm_box["body"]["quantifiers"])
    for idx, quantifier in enumerate(qgm_box["body"]["quantifiers"]):
        actions += "Q({}) ".format(1 if idx + 1 == q_len else 0)
        actions += "T({}) ".format(quantifier)

    """
    # H
    h_len = len(qgm_box["head"])
    for idx, head in enumerate(qgm_box["head"]):
        actions += "H({}) ".format(1 if idx + 1 == h_len else 0)
        actions += "A({}) ".format(head[0])
        actions += "C({}) ".format(head[1])
    """
    h_len = len(qgm_box["head"])
    actions += "H({}) ".format(h_len-1)
    for idx, head in enumerate(qgm_box["head"]):
        actions += "A({}) ".format(head[0])
        actions += "C({}) ".format(head[1])

    # P
    p_len = len(qgm_box["body"]["local_predicates"])
    for idx, predicate in enumerate(qgm_box["body"]["local_predicates"]):
        actions += "P({}) ".format(1 if idx + 1 == p_len else 0)
        actions += "O({}) ".format(predicate[2])
        actions += "A({}) ".format(predicate[0])
        actions += "C({}) ".format(predicate[1])

    actions = actions.strip(" ")
    return actions


if __name__ == "__main__":
    #dataset_names = ["spider", "wikisql", "wikitablequestions"]
    dataset_names = ["spider"]
    data_types = ["dev", "train", "test"]

    for dataset_name in dataset_names:
        for data_type in data_types:
            if dataset_name == "spider" and data_type == "test":
                continue
            file_name = "../../data/{}/{}.json".format(dataset_name, data_type)

            print("Dealing with {}".format(file_name))

            datas = json.load(open(file_name))

            # Add qgm_actions
            for data in datas:
                qgm_action = create_qgm_action(data["qgm"])
                data["qgm_action"] = qgm_action

            # Save
            with open(file_name, "w") as f:
                json.dump(datas, f)
