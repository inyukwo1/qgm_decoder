import os
import json

original_dir_path = "../data/spider/original"
new_dir_path = "../data/spider"


def parse_question(datas):
    TARGET_CHARACTERS = ["'", '"', ",", "?", ".", "@", "-", "!", "$", "%", "#"]
    for data in datas:
        question = data["question"]
        question = question.replace("\u201c", '"')
        question = question.replace("\u201d", '"')
        question = question.replace("\u2018", "'")
        question = question.replace("\u2019", "'")
        for item in TARGET_CHARACTERS:
            question = question.replace(item, " {} ".format(item))
        while "  " in question:
            question = question.replace("  ", " ")
        data["question"] = question
        data["question_toks"] = data["question"].split(" ")
    return datas


def fix_dev_gold_sql(sqls):
    ori = "SELECT T2.name FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id INTERSECT SELECT T2.name FROM Likes AS T1 JOIN Highschooler AS T2 ON T1.liked_id  =  T2.id\tnetwork_1\n"
    fixed = "SELECT T2.name FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id INTERSECT SELECT T2.name FROM Likes AS T3 JOIN Highschooler AS T2 ON T3.liked_id  =  T2.id\tnetwork_1\n"
    print("Fixing gold sql")
    for idx, sql in enumerate(sqls):
        if sql == ori:
            sqls[idx] = fixed
            print("found error at idx:{}".format(idx))
            print("ori: {}".format(ori))
            print("fixed: {}\n".format(fixed))
    return sqls


def fix_dev_data(datas):
    """
    at dev.json, db_id: car_1
    cards -> cars
        ex: What are the different models for the cards produced after 1980?",
        ex: "What is the average miles per gallon of all the cards with 4 cylinders?"
    """
    for data in datas:
        if "cards" in data["question"]:
            data["question"] = data["question"].replace("cards", "cars")
            data["question_toks"] = data["question"].split(" ")

    return datas


def fix_db_info(dbs):
    DBS_WITH_ERROR = [
        {
            "db_id": "scholar",
            "correct_table_names": [
                "venue",
                "author",
                "dataset",
                "journal",
                "key phrase",
                "paper",
                "cite",
                "paper dataset",
                "paper key phrase",
                "writes",
            ],
        },
        {
            "db_id": "store_1",
            "correct_table_names": [
                "artists",
                "sqlite sequence",
                "albums",
                "employees",
                "customers",
                "genres",
                "invoices",
                "media types",
                "tracks",
                "invoice lines",
                "playlists",
                "playlist tracks",
            ],
        },
        {
            "db_id": "formula_1",
            "correct_table_names": [
                "circuits",
                "races",
                "drivers",
                "status",
                "seasons",
                "constructors",
                "constructor standings",
                "results",
                "driver standings",
                "constructor results",
                "qualifying",
                "pitstops",
                "laptimes",
            ],
        },
    ]
    for db in dbs:
        for target_db in DBS_WITH_ERROR:
            correct_table_names = target_db["correct_table_names"]
            if db["db_id"] == target_db["db_id"]:
                print("Fixing {}...".format(db["db_id"]))
                table_names = db["table_names"]
                column_names = db["column_names"]
                # Check if no typo in my answer
                assert len(table_names) == len(correct_table_names)
                assert set([str(table_name) for table_name in table_names]) == set(
                    [str(table_name) for table_name in correct_table_names]
                )
                old = {
                    str(table_name): idx for idx, table_name in enumerate(table_names)
                }
                new = {
                    str(table_name): idx
                    for idx, table_name in enumerate(correct_table_names)
                }
                ref = {
                    new[table_name]: old[table_name]
                    for table_name in correct_table_names
                }

                # Get old list
                save = [[] for _ in range(len(table_names) + 1)]
                for item in column_names:
                    save[item[0] + 1] += [item[1]]

                # Create new list
                new_column_names = [[-1, "*"]]
                for idx in range(len(table_names)):
                    for item in save[ref[idx] + 1]:
                        new_column_names += [[idx, item]]

                # Check
                assert len(new_column_names) == len(column_names)
                for idx in range(len(column_names)):
                    assert (
                        new_column_names[idx][0] == db["column_names_original"][idx][0]
                    ), "{} : {} : {}".format(
                        idx, new_column_names[idx], db["column_names_original"][idx]
                    )

                # Change
                db["table_names"] = correct_table_names
                db["column_names"] = new_column_names

                print("Done editing!\n")
    return dbs


def parse_spider_dataset():
    ## Train file
    file_name1 = "train_spider.json"
    file_name2 = "train_others.json"
    file_path1 = os.path.join(original_dir_path, file_name1)
    file_path2 = os.path.join(original_dir_path, file_name2)
    train_data1 = json.load(open(file_path1))
    train_data2 = json.load(open(file_path2))
    # Combine train file
    train_data = train_data1 + train_data2
    # Parse data
    train_data = parse_question(train_data)
    # Save data
    new_file_name = "train.json"
    save_path = os.path.join(new_dir_path, new_file_name)
    with open(save_path, "w") as f:
        json.dump(train_data, f)

    ## Dev file
    file_name = "dev.json"
    file_path = os.path.join(original_dir_path, file_name)
    dev_data = json.load(open(file_path))
    # Pares data
    dev_data = parse_question(dev_data)
    dev_data = fix_dev_data(dev_data)
    # Save data
    save_path = os.path.join(new_dir_path, file_name)
    with open(save_path, "w") as f:
        json.dump(dev_data, f)

    ## Tables file
    file_name = "tables.json"
    file_path = os.path.join(original_dir_path, file_name)
    table_data = json.load(open(file_path))
    # Fix error
    table_data = fix_db_info(table_data)
    # Save data
    save_path = os.path.join(new_dir_path, file_name)
    with open(save_path, "w") as f:
        json.dump(table_data, f)

    ## Dev gold sql
    file_name = "dev_gold.sql"
    file_path = os.path.join(original_dir_path, file_name)
    dev_gold_sql = open(file_path).readlines()
    # Fix error
    dev_gold_sql = fix_dev_gold_sql(dev_gold_sql)
    # Save data
    save_path = os.path.join(new_dir_path, file_name)
    with open(save_path, "w") as f:
        for line in dev_gold_sql:
            f.write(line)

    ## Train gold sql
    file_name = "train_gold.sql"
    file_path = os.path.join(original_dir_path, file_name)
    train_gold_sql = open(file_path).readlines()
    save_path = os.path.join(new_dir_path, file_name)
    with open(save_path, "w") as f:
        for line in train_gold_sql:
            f.write(line)

    print("Parsing complete!")


if __name__ == "__main__":
    parse_spider_dataset()
