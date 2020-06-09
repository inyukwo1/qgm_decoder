import os
import tqdm
import shutil
import zipfile
import requests


def download_file_from_google_drive(id, destination):
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in tqdm.tqdm(response.iter_content(CHUNK_SIZE)):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def unzip(path_to_zip_file, directory_to_extract_to):
    if not os.path.exists(directory_to_extract_to):
        os.makedirs(directory_to_extract_to)
    with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
        zip_ref.extractall(directory_to_extract_to)


def move_files(path):
    path = os.path.join(path, "spider")
    files = [f for f in os.listdir(path)]
    for file in files:
        ori_path = os.path.join(path, file)
        new_path = os.path.join(path, "..")
        new_path = os.path.join(new_path, file)
        shutil.move(ori_path, new_path)
    os.rmdir(path)


if __name__ == "__main__":
    file_id = "1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0"  # Spider 1.1(?) link
    intermediate_destination = "./spider.zip"
    final_destination = "../data/spider/original/"
    download_file_from_google_drive(file_id, intermediate_destination)
    unzip(intermediate_destination, final_destination)
    move_files(final_destination)
    os.remove(intermediate_destination)
