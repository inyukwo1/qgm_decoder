
import json
import argparse

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--in_path", type=str,  required=True)
    arg_parser.add_argument(
        "--out_path", type=str,  required=True
    )
    args = arg_parser.parse_args()

    with open(args.in_path, "r") as f:
        string = f.read()

    string = string.replace("\\u00a0", " ")
    string = string.replace("\\xa0", " ")
    # string = string.replace("\\u00b2", "2") # 제곱 기호
    # string = string.replace("\\u2212", "-")
    # string = string.replace("\\u2013", "-")
    # string = string.replace("\\u2014", "-")
    # string = string.replace("\\u00f6", "o")
    # string = string.replace("\\u20ac", "e") # 유로
    # string = string.replace("\\u2606", "") # 별
    # string = string.replace("\\u00f1", "n") #

    with open(args.out_path, "w") as f:
        f.write(string)
