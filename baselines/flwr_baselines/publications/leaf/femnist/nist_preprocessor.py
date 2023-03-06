import pathlib
from typing import Union

import pandas as pd
from dataset_utils import calculate_series_hashes, hex_decimal_to_char
from PIL import Image
from tqdm import tqdm


class NISTPreprocessor:
    """Preprocess files from two directories divided by class and by writer."""

    def __init__(self, data_dir: Union[str, pathlib.Path]) -> None:
        self._data_dir = (
            data_dir if isinstance(data_dir, pathlib.Path) else pathlib.Path(data_dir)
        )
        self._raw_data_dir = self._data_dir / "raw"
        self._processed_dir = self._data_dir / "processed"
        self._processed_images_dir = self._processed_dir / "images"
        self._processed_images_information_path = (
            self._processed_dir / "resized_images_to_labels.csv"
        )
        self._by_class_nist = self._raw_data_dir / "by_class"
        self._by_writer_nist = self._raw_data_dir / "by_write"
        self._writer_df: pd.DataFrame
        self._class_df: pd.DataFrame
        self._df: pd.DataFrame

    def preprocess(self, overwrite: bool = False) -> None:
        """Extracts necessary information to create data that has both writer
        and class information and preprocesses the dataset as by the authors of
        the FEMNIST paper (which is not the same as by the EMNIST paper).

        1. Extract writer_id from the directory structure.
        2. Extracts class_id from the directory structure.
        3. Calculate hashes from the images to enable merging data.
        4. Merge information based on hash values.
        5. Preprocess images (reduce the size of them, use LANCZOS resampling).
        6. Create csv file with the path.
        """
        if self._processed_images_information_path.exists() and not overwrite:
            print(
                f"The preprocessed information already exists in {self._processed_images_information_path}. "
                f"Specify 'overwrite' as True to recreate this information."
            )
            return
        self._writer_df = self._extract_writer_information()
        self._class_df = self._extract_class_information()
        self._calculate_hashes()
        self._df = self._merge_class_and_writer_information()
        self.create_dir_structure()
        original_images_information_path = (
            self._processed_dir / "original_images_to_labels.csv"
        )
        print(
            f"Saving information about raw images to {original_images_information_path} started"
        )
        self._df.to_csv(original_images_information_path)
        print(
            f"Saving information about raw images to {original_images_information_path} done"
        )
        self._new_df = self._preprocess_images()
        print(
            f"Saving information about raw images to {self._processed_images_information_path} started"
        )
        self._new_df.to_csv(self._processed_images_information_path)
        print(
            f"Saving information about raw images to {self._processed_images_information_path} done"
        )

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
        writer_df["writer_id"] = writer_df["path_by_writer"].map(
            lambda x: x.parent.parent.name
        )
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
        self._writer_df["hash"] = calculate_series_hashes(
            self._writer_df["path_by_writer"]
        )
        self._class_df["hash"] = calculate_series_hashes(
            self._class_df["path_by_class"]
        )
        self._df = pd.merge(self._writer_df, self._class_df, on="hash")
        print("Hashes calculation done")

    def _preprocess_images(self):
        """Preprocess images - resize to 28x28 and save them in the processed directory."""
        print("Image preprocessing started")
        resized_size = (28, 28)
        new_df = self._df.copy()
        new_df["path"] = pathlib.Path("")
        for index, row in tqdm(self._df.iterrows(), total=self._df.shape[0]):
            file_path = row["path_by_writer"]
            img = Image.open(file_path)
            gray = img.convert("L")
            gray.thumbnail(resized_size, Image.LANCZOS)
            image_name = (
                "character_"
                + row["character"]
                + "_"
                + "author_"
                + row["writer_id"]
                + ".png"
            )
            new_path = self._processed_images_dir / image_name
            gray.save(new_path)
            new_df.iloc[index, -1] = new_path
        print("Image preprocessing done")
        return new_df


if __name__ == "__main__":
    nist_data_path = pathlib.Path("data")
    nist_preprocessor = NISTPreprocessor(nist_data_path)
    nist_preprocessor.preprocess()
