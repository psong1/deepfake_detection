import os
import zipfile
from dotenv import load_dotenv

def download_dataset():
    load_dotenv()

    os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
    os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')

    data_dir = './Dataset'
    if os.path.exists(data_dir):
        print(f'Dataset already exists in {data_dir}. Skipping download.')
        return
    print(f'Downloading dataset from Kaggle to {data_dir}')

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    dataset_handle = "manjilkarki/deepfake-and-real-images"
    api.dataset_download_files(dataset_handle, path=".", unzip=True)

    print(f'Dataset downloaded and extracted to {data_dir}')