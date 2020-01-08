from universal_utils import download_file_from_google_drive
from gnn.end2end import End2EndGNN
import torch
import os.path


GEO_MODEL_PATH="test_models/gnn_geo.model"


def _prepare_model(path, config_id, model_id, voc_id_1, voc_id_2, voc_id_3):
    if os.path.exists(path):
        return
    os.mkdir(path)
    download_file_from_google_drive(config_id, path + "/config.json")
    download_file_from_google_drive(model_id, path + "/best.th")
    os.mkdir(path + "/vocabulary")
    download_file_from_google_drive(voc_id_1, path + "/vocabulary/non_padded_namespaces.txt")
    download_file_from_google_drive(voc_id_2, path + "/vocabulary/rule_labels.txt")
    download_file_from_google_drive(voc_id_3, path + "/vocabulary/tokens.txt")


def prepare_model(dataset):
    model_path = "test_models/gnn_{}.model".format(dataset)
    if dataset == "geo":
        _prepare_model(model_path, "16qJ73v5TzaKmj-8csAfbZEqA7FvnpAiP",
                                    "1IOzA1v9iPSMLmc7oio347U2Ugxd4a66-",
                                    "1aRmPoWXz5j7ie3CgjdiRcnlgYcSHEYbP",
                                    "1EVMAYvPqABZIPcSxSVphaUXBiTORyaxM",
                                    "1OP_d5II9Vlq5ltLCkg-IKTQ_Y7QDPfER")
    elif dataset == "imdb":
        _prepare_model(model_path, "1cqk3X6_YGmFMwLf6Ueoca7QM2v67qIER",
                                    "18AnBDEhcQUc7ChbBuaUqjaSNlJbosx8y",
                                    "1VRxV9rmrcI0FTWai3xwMjHa-gPdMv70U",
                                    "19b6NIVzpWPMOM6FtmFlU6eY71fqRECJf",
                                    "1athaqpdTpKRRjbajfhNh09jR5KU9xBGh")
    elif dataset == "mas":
        _prepare_model(model_path, "1euVD-pkKMOK_NAJhrmBQPDukgKbnmiZn",
                                    "1JrKfK9-VhybprmWlW9J6NnC1oqWHJoth",
                                    "1cBnqzoj9uuYDkDQw_TgFt3VRP14cVmg_",
                                    "1sqqxhRNaAQX2gl0THVzija0wTc_7LG_W",
                                    "1QqU4SRvAv-oBa4I-BuPVZE_YqjG7uvDr")
    elif dataset == "patients":
        _prepare_model(model_path, "1DJPg7Zx5FQlvYqagbH72o14H-mlk6b1v",
                       "15fpNJLOZcKwtITV1lW0YVY1NUkzjVN13",
                       "1A3eGMa--P-CHgixKNxoDjTsiaMrFfdAq",
                       "1ufCNnGLkkeaE6wdseRN3-bt4gGb0F0HR",
                       "1eb6ebdDn20_QWuP-IwxjOYFx6NBTKYu6")
    elif dataset == "restaurant":
        _prepare_model(model_path, "1ZpPLIc6jQuuLouFiKabdYJCgIr0uPH_n",
                       "1fPGixj8JMgE02wDqxiQgydkV9JTqQ7V4",
                       "1aX2s3393nZQIz-Gu10IdhW479XAJBQDZ",
                       "1WrBQCNSj5_T54gHU2gF4Ak6xsoWX_UKA",
                       "1I9PmGisFZ-g7cjVjG_LyRJZKiaEPUV1F")
    elif dataset == "scholar":
        _prepare_model(model_path, "1mlr7hnl6qDnaplcZEw7o06cGrmr6DRjW",
                       "17R1wm7C9uS17-4IwYduP7N4bSE4dRmR9",
                       "1XNtaONhiWKNZmPZy6UtfCxVzmBQ2KIUD",
                       "1_2ETjKZq9dZK_D-OwSTbcr2_Up5cbI4S",
                       "1vQN7_FMGQFA9tgAKFdb5WYb3zsdInvsd")
    elif dataset == "spider":
        _prepare_model(model_path, "1w1vj6t-4arb4I7D5cDdoppDVR4vQANSI",
                       "1TMebFBlFzmufZPnsieLQQRbojdGjJLJq",
                       "1Dm0tOei1DSeQyeEB3tz_D3_JJrkEe1Ez",
                       "1TAAUmmt0FLQF6lpxY31otbDuSXfZECKI",
                       "1M6RPm64ad6H66Wz1ICwynok1AOfuoEW9")
    elif dataset == "tpch":
        _prepare_model(model_path, "1uGE0M8oRaeRrDw5ZZlR0yFuiZxz0d-J5",
                       "1JFSxJsLODq_Z7OFFSY55KYLFYzYnTta9",
                       "1xi4sb09743trRfeRWZ9iLBWlcUQ1xCme",
                       "1Nk_LrO1fOKlrpNwlesMqRfEK_3r-izfz",
                       "1bpnmbMjJtUoc39YslaMD3paEqls_ZZRw")
    elif dataset == "yelp":
        _prepare_model(model_path, "10lNyTPMGd3z5O_U9d7oWxGih1Diq_93e",
                       "1XQ-deziTHtPAE-gg-9zdLz5zLLxLm6m_",
                       "1w0glYH6Om7FkJpiIAKDjc-ehPiJ1zJav",
                       "1A3Jw9aQMc7SWPM9IuSM-LXjlFH_Sm8Z6",
                       "1yGv-b7-wKYpeUYCysBMFTOyNaOfkvFfB")


def test_end2end_geo():
    end2end_gnn = End2EndGNN()
    end2end_gnn.prepare_model("geo")
    q, _, _ = end2end_gnn.run_model("geo", "what is the biggest city in arizona")
    assert q == """select city.city_name from city where city.state_name = ' value ' and city.population = ( select max ( city.population ) from city where city.state_name = ' value ' )"""


def test_end2end_imdb():
    end2end_gnn = End2EndGNN()
    end2end_gnn.prepare_model("imdb")
    q, _, _ = end2end_gnn.run_model("imdb", "What year was Ellen Page born?")
    assert q == """select actor.birth_year from actor where actor.name = ' value '"""


def test_end2end_mas():
    end2end_gnn = End2EndGNN()
    end2end_gnn.prepare_model("mas")
    q, _, _ = end2end_gnn.run_model("mas", "return me the number of researchers in database area in \"University of Michigan\"")
    assert q == "NO PREDICTION"


def test_end2end_patients():
    end2end_gnn = End2EndGNN()
    end2end_gnn.prepare_model("patients")
    q, _, _ = end2end_gnn.run_model("patients", "aggregate the ages of all patients in the database")
    assert q == """select sum ( patients.age ) from patients"""


def test_end2end_restaurant():
    end2end_gnn = End2EndGNN()
    end2end_gnn.prepare_model("restaurant")
    q, _, _ = end2end_gnn.run_model("restaurant", "give me a restaurant in the bay area ?")
    assert q == "NO PREDICTION"


def test_end2end_scholar():
    end2end_gnn = End2EndGNN()
    end2end_gnn.prepare_model("scholar")
    q, _, _ = end2end_gnn.run_model("scholar", "What papers has Sharon Goldwater written ?")
    assert q == """select distinct paper.venueid from writes , author where writes.authorid = author.authorname and author.authorname = ' value '"""


def test_end2end_spider():
    end2end_gnn = End2EndGNN()
    end2end_gnn.prepare_model("spider")
    q, _, _ = end2end_gnn.run_model("concert_singer", "How many singers do we have?")
    assert q == """select count ( * ) from singer"""


def test_end2end_tpch():
    end2end_gnn = End2EndGNN()
    end2end_gnn.prepare_model("tpch")
    q, _, _ = end2end_gnn.run_model("tpch", "Find a list of the top 100 customers who have ever placed large quantity orders. The query lists the customer name, customer key, the order key, date and total priceand the quantity for the order")
    assert q == """select orders.order_key , orders.total_price from orders order by orders.total_price desc limit 1"""


def test_end2end_yelp():
    end2end_gnn = End2EndGNN()
    end2end_gnn.prepare_model("yelp")
    q, _, _ = end2end_gnn.run_model("yelp", "Find the number of reviews on businesses located in South Summerlin neighborhood")
    assert q == """select business.business_id"""

if __name__ == "__main__":
    prepare_model("yelp")
    test_end2end_yelp()
