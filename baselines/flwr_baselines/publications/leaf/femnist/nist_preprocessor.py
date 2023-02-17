import pathlib
from typing import Union

import pandas as pd
from PIL import Image
from tqdm import tqdm

from dataset_utils import hex_decimal_to_char, calculate_series_hashes


class NISTPreprocessor:
    def __init__(self, data_dir: Union[str, pathlib.Path]) -> None:
        self._data_dir = data_dir if isinstance(data_dir, str) else pathlib.Path(data_dir)
        self._raw_data_dir = self._data_dir / "raw"
        self._processed_dir = self._data_dir / "processed"
        self._processed_images_dir = self._processed_dir / "images"
        self._by_class_nist = self._raw_data_dir / "by_class"
        self._by_writer_nist = self._raw_data_dir / "by_write"
        self._writer_df: pd.DataFrame
        self._class_df: pd.DataFrame
        self._df: pd.DataFrame

    def preprocess(self):
        self._writer_df = self._extract_writer_information()
        self._class_df = self._extract_class_information()
        self._calculate_hashes()
        self._df = self._merge_class_and_writer_information()
        self.create_dir_structure()
        original_images_information_path = self._processed_dir / "original_images_to_labels.csv"
        print(f"Saving information about raw images to {original_images_information_path} started")
        self._df.to_csv(original_images_information_path)
        print(f"Saving information about raw images to {original_images_information_path} done")
        self._new_df = self._preprocess_images()
        processed_images_information_path = self._processed_dir / "resized_images_to_labels.csv"
        print(f"Saving information about raw images to {processed_images_information_path} started")
        self._new_df.to_csv(processed_images_information_path)
        print(f"Saving information about raw images to {processed_images_information_path} done")

    def create_dir_structure(self):
        print("Directory structure creation started")
        self._processed_dir.mkdir(exist_ok=True)
        print(f"Created/already exist directory {self._processed_dir}")
        self._processed_images_dir.mkdir(exist_ok=True)
        print(f"Created/already exist directory {self._processed_images_dir}")
        print("Directory structure creation done")

    def _extract_writer_information(self):
        print("Writer information preprocessing started")
        images_paths = self._by_writer_nist.glob("*/*/*/*")
        images_paths = list(images_paths)
        writer_df = pd.DataFrame(images_paths, columns=["path_by_writer"])
        writer_df["writer_id"] = writer_df["path_by_writer"].map(lambda x: x.parent.parent.name)
        print("Writer information preprocessing done")
        return writer_df

    def _extract_class_information(self):
        print("Class information preprocessing started")
        hsf_and_train_dir = self._by_class_nist.glob("*/*")
        hsf_dirs = [path for path in hsf_and_train_dir if "train" not in str(path)]
        images_paths = []
        for hsf_path in hsf_dirs:
            images_paths.extend(list(hsf_path.glob("*")))
        class_df = pd.DataFrame(images_paths, columns=["path_by_class"])
        chars = class_df["path_by_class"].map(lambda x: x.parent.parent.name)
        class_df["character"] = chars.map(hex_decimal_to_char)
        print("Class information preprocessing done")
        return class_df

    def _merge_class_and_writer_information(self) -> pd.DataFrame:
        print("Merging of the class and writer information by hash values started")
        merged = pd.merge(self._writer_df, self._class_df, on="hash")
        print("Merging of the class and writer information by hash values done")
        return merged

    def _calculate_hashes(self):
        print("Hashes calculation started")
        # Assumes that the class_df and writer_df are created
        self._writer_df["hash"] = calculate_series_hashes(self._writer_df["path_by_writer"])
        self._class_df["hash"] = calculate_series_hashes(self._class_df["path_by_class"])
        self._df = pd.merge(self._writer_df, self._class_df, on="hash")
        print("Hashes calculation done")

    def _preprocess_images(self):
        """Preprocess images - resize to 28x28 and save them in the processed directory."""
        print("Image preprocessing started")
        resized_size = (28, 28)
        new_df = self._df.copy()  # maybe drop path_by_witer and by_class and add hsf
        new_df["path"] = pathlib.Path("")
        for index, row in tqdm(self._df.iterrows(), total=self._df.shape[0]):
            file_path = row["path_by_writer"]
            img = Image.open(file_path)
            gray = img.convert('L')
            gray.thumbnail(resized_size, Image.LANCZOS)
            image_name = "character_" + row["character"] + "_" + "author_" + row["writer_id"] + ".png"
            new_path = self._processed_images_dir / image_name
            gray.save(new_path)
            new_df.iloc[index, -1] = new_path
        print("Image preprocessing done")
        return new_df


if __name__ == "__main__":
    nist_data_path = pathlib.Path("data")
    nist_preprocessor = NISTPreprocessor(nist_data_path)
    nist_preprocessor.preprocess()
