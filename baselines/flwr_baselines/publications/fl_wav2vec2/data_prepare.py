import os
import ssl
import tarfile
import urllib.request

import pandas as pd


# Download the file and show a progress bar
def download_file(url, filename):
    print(f"Downloading {url}...")
    retries = 3
    while retries > 0:
        try:
            with urllib.request.urlopen(url, context=ssl._create_unverified_context()) as response, open(filename, "wb") as out_file:
                total_size = int(response.getheader("Content-Length"))
                block_size = 1024 * 8
                count = 0
                while True:
                    data = response.read(block_size)
                    if not data:
                        break
                    count += 1
                    out_file.write(data)
                    percent = int(count * block_size * 100 / total_size)
                    print(f"\rDownloading: {percent}% [{count * block_size}/{total_size}]", end="")
                print(f"\nDownload complete.")
                return
        except Exception as e:
            print(f"\nError occurred during download: {e}")
            retries -= 1
            if retries > 0:
                print(f"Retrying ({retries} retries left)...")
            else:
                print("Download failed.")
                raise e

# Extract the contents and show a progress bar
def extract_file(filename, extract_path):
    print(f"Extracting {filename}...")
    with tarfile.open(filename, "r:gz") as tar:
        members = tar.getmembers()
        total_files = len(members)
        current_file = 0
        for member in members:
            current_file += 1
            tar.extract(member, path=extract_path)
            percent = int(current_file * 100 / total_files)
            print(f"\rExtracting: {percent}% [{current_file}/{total_files}]", end="")
        print(f"\nExtraction complete.")

# Delete the downloaded file
def delete_file(filename):
    os.remove(filename)
    print(f"Deleted {filename}.")

#Change the path corespond to your actual path    
def csv_path_audio(path):
    for i in range(1983):
        df_train = pd.read_csv(f"./data/client_{i}/ted_train.csv")
        df_dev = pd.read_csv(f"./data/client_{i}/ted_dev.csv")
        df_test = pd.read_csv(f"./data/client_{i}/ted_test.csv")
        df_train["wav"] = df_train["wav"].str.replace("/local_disk/idyie/tnguyen/TEDLIUM_release-3/legacy/train/sph/",path)
        df_dev["wav"] = df_dev["wav"].str.replace("/local_disk/idyie/tnguyen/TEDLIUM_release-3/legacy/train/sph/",path)
        df_test["wav"] = df_test["wav"].str.replace("/local_disk/idyie/tnguyen/TEDLIUM_release-3/legacy/train/sph/",path)
        df_train.to_csv(f"./data/client_{i}/ted_train.csv",index = False)
        df_dev.to_csv(f"./data/client_{i}/ted_dev.csv", index = False)
        df_test.to_csv(f"./data/client_{i}/ted_test.csv", index = False)

# CHANGE THE PATH CORESPOND TO YOUR PATH
url = "https://projets-lium.univ-lemans.fr/wp-content/uploads/corpus/TED-LIUM/TEDLIUM_release-3.tgz"
filename = "data/TEDLIUM_release-3.tgz"
extract_path = "data/audio" # replace with the path to the directory where you want to extract the files


try:
    download_file(url, filename)
    extract_file(filename, extract_path)
finally:
    delete_file(filename)


csv_path_audio(extract_path)


