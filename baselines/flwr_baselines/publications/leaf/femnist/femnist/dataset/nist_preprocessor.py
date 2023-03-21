import pathlib
from logging import INFO, WARN
from typing import Dict, Optional, Union

import pandas as pd
from femnist.dataset.dataset_utils import calculate_series_hashes, hex_decimal_to_char
from flwr.common.logger import log
from PIL import Image
from tqdm import tqdm


class NISTPreprocessor:
    """Preprocess NIST images from two directories divided by class and by
    writer.

    The preprocessing procedure include different step for FEMNIST and FeMNIST.

    Using two datasets (by_class and by_write, actually one dataset but divided differently) is
    required to merge information about the writer_id and label.
    """

    def __init__(
        self, data_dir: Union[str, pathlib.Path], to_dataset: str = "FeMNIST"
    ) -> None:
        self._data_dir = (
            data_dir if isinstance(data_dir, pathlib.Path) else pathlib.Path(data_dir)
        )
        self._raw_data_dir = self._data_dir / "raw"
        self._processed_dir = self._data_dir / ("processed" + "_" + to_dataset)
        self._processed_images_dir = self._processed_dir / "images"
        self._processed_images_information_path = (
            self._processed_dir / "processed_images_to_labels.csv"
        )
        self._raw_images_information_path = self._data_dir / "raw_images_to_labels.csv"
        self._by_class_nist = self._raw_data_dir / "by_class"
        self._by_writer_nist = self._raw_data_dir / "by_write"
        self._writer_df: Optional[pd.DataFrame] = None
        self._class_df: Optional[pd.DataFrame] = None
        self._df: Optional[pd.DataFrame] = None
        self._preprocessed_df: Optional[pd.DataFrame] = None

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
        # Do not preprocess if the preprocessed information already exist
        if self._processed_images_information_path.exists() and not overwrite:
            log(
                WARN,
                f"The preprocessed information already exists in {self._processed_images_information_path}. "
                f"It's assumed that the preprocessed images exist too."
                f"Specify 'overwrite' as True to preprocess the images and recreate the reference information.",
            )
            return

        # Extraction th raw images information if it does not already exist
        if not self._raw_images_information_path.exists():
            self._writer_df = self._extract_writer_information()
            self._class_df = self._extract_class_information()
            self._calculate_hashes()
            self._df = self._merge_class_and_writer_information()
            log(
                INFO,
                f"Saving information about raw images to {self._raw_images_information_path} started",
            )
            self._df.to_csv(self._raw_images_information_path)
            log(
                INFO,
                f"Saving information about raw images to {self._raw_images_information_path} done",
            )
        else:
            self._df = pd.read_csv(self._raw_images_information_path, index_col=0)
        self.create_dir_structure()
        self._preprocessed_df = self._preprocess_images()
        log(
            INFO,
            f"Saving information about raw images to {self._processed_images_information_path} started",
        )
        self._preprocessed_df.to_csv(self._processed_images_information_path)
        log(
            INFO,
            f"Saving information about raw images to {self._processed_images_information_path} done",
        )

    def create_dir_structure(self):
        log(INFO, "Directory structure creation started")
        self._processed_dir.mkdir(exist_ok=True)
        log(INFO, f"Created/already exist directory {self._processed_dir}")
        self._processed_images_dir.mkdir(exist_ok=True)
        log(INFO, f"Created/already exist directory {self._processed_images_dir}")
        log(INFO, "Directory structure creation done")

    def _extract_writer_information(self):
        log(INFO, "Writer information preprocessing started")
        images_paths = self._by_writer_nist.glob("*/*/*/*")
        images_paths = list(images_paths)
        writer_df = pd.DataFrame(images_paths, columns=["path_by_writer"])
        writer_df["writer_id"] = writer_df["path_by_writer"].map(
            lambda x: x.parent.parent.name
        )
        log(INFO, "Writer information preprocessing done")
        return writer_df

    def _extract_class_information(self):
        log(INFO, "Class information preprocessing started")
        hsf_and_train_dir = self._by_class_nist.glob("*/*")
        hsf_dirs = [path for path in hsf_and_train_dir if "train" not in str(path)]
        images_paths = []
        for hsf_path in hsf_dirs:
            images_paths.extend(list(hsf_path.glob("*")))
        class_df = pd.DataFrame(images_paths, columns=["path_by_class"])
        chars = class_df["path_by_class"].map(lambda x: x.parent.parent.name)
        class_df["character"] = chars.map(hex_decimal_to_char)
        log(INFO, "Class information preprocessing done")
        return class_df

    def _merge_class_and_writer_information(self) -> pd.DataFrame:
        log(INFO, "Merging of the class and writer information by hash values started")
        merged = pd.merge(self._writer_df, self._class_df, on="hash")
        log(INFO, "Merging of the class and writer information by hash values done")
        return merged

    def _calculate_hashes(self):
        log(INFO, "Hashes calculation started")
        # Assumes that the class_df and writer_df are created
        self._writer_df["hash"] = calculate_series_hashes(
            self._writer_df["path_by_writer"]
        )
        self._class_df["hash"] = calculate_series_hashes(
            self._class_df["path_by_class"]
        )
        self._df = pd.merge(self._writer_df, self._class_df, on="hash")
        log(INFO, "Hashes calculation done")

    def _preprocess_images(self) -> pd.DataFrame:
        """Preprocess images - resize to 28x28 and save them in the processed directory.

        Returns
            preprocessed_df_info: pd.DataFrame
            dataframe with information about the path, writer_id, character(label)
        """
        log(INFO, "Image preprocessing started")
        writer_to_character_to_count: Dict[str : Dict[str:int]] = {}
        resized_size = (28, 28)
        new_df = self._df.copy()
        new_df["path"] = pathlib.Path("")
        for index, row in tqdm(self._df.iterrows(), total=self._df.shape[0]):
            file_path = row["path_by_writer"]
            img = Image.open(file_path)
            gray = img.convert("L")
            gray.thumbnail(resized_size, Image.LANCZOS)
            writer_id = row["writer_id"]
            character = row["character"]
            if writer_id not in writer_to_character_to_count:
                writer_to_character_to_count[writer_id] = {}
            if character not in writer_to_character_to_count[writer_id]:
                writer_to_character_to_count[writer_id][character] = 0
            image_name = (
                "character_"
                + character
                + "_"
                + "nth_"
                + str(writer_to_character_to_count[writer_id][character])
                + "_"
                + "author_"
                + writer_id
                + ".png"
            )
            writer_to_character_to_count[writer_id][character] += 1
            new_path = self._processed_images_dir / image_name
            gray.save(new_path)
            new_df.iloc[index, -1] = new_path
        log(INFO, "Image preprocessing done")
        new_df = new_df.loc[:, ["path", "writer_id", "character"]]
        return new_df
