import json

# dic = {"identical": 0, "dep": 1, "rev": 2, "etc": 3}
dep_type = {
    "cc",
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
}

use_dep_type = True
modes = ["train", "dev"]
for mode in modes:
    dep_file_name = "./{}_dep.txt".format(mode)
    dep_label_file_name = "./{}_label.txt".format(mode)
    data_file_name = "../../data/spider/{}.json".format(mode)
    datas = json.load(open(data_file_name))

    deps = open(dep_file_name).readlines()
    labels = open(dep_label_file_name).readlines()

    assert len(deps) == len(datas), "Num different: {} {}".format(len(deps), len(datas))

    for idx, (dep, label, sql) in enumerate(zip(deps, labels, datas)):
        question = sql["question_arg"]
        # Parse dep
        dep = dep.strip()[1:-1].split(", ")
        dep = [int(item) - 1 for item in dep]
        # Parse label
        label = label.strip()[1:-1].replace("'", "").split(", ")

        print("question: {}".format(question))
        print("dep: {}".format(dep))
        print("label: {}".format(label))
        assert len(question) == len(dep), "Idx:{} Num different: {} {}".format(
            idx, len(question), len(dep)
        )
        assert len(dep) == len(label), "Idx:{} Num different: {} {}".format(
            idx, len(dep), len(label)
        )

        matrix = []
        for idx_1 in range(len(question)):
            tmp = []
            dep_key = label[idx_1] if use_dep_type else "depen"
            for idx_2 in range(len(question)):
                if idx_1 == idx_2:
                    key = "identical"
                elif dep[idx_1] == idx_2:
                    key = dep_key
                elif dep[idx_2] == idx_1:
                    key = "{}_rev".format(dep_key)
                else:
                    key = "no_dep"
                tmp += [key]
            matrix += [tmp]
        datas[idx]["question_relation"] = matrix

    # Save
    with open(data_file_name, "w") as f:
        json.dump(datas, f)
