"""
DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation

https://arxiv.org/pdf/2211.11501.pdf

DS-1000 is a code generation benchmark with a thousand data science questions spanning seven Python libraries that (1) reflects diverse, realistic, and practical use cases, (2) has a reliable metric, (3) defends against memorization by perturbing questions.

Homepage: https://ds1000-code-gen.github.io/
"""

import fcntl
import functools
import io
import itertools
import pathlib
import warnings
import zipfile

import requests
import tqdm

from bigcode_eval.base import Task

_CITATION = """
@article{Lai2022DS1000,
  title={DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation},
  author={Yuhang Lai and Chengxi Li and Yiming Wang and Tianyi Zhang and Ruiqi Zhong and Luke Zettlemoyer and Scott Wen-tau Yih and Daniel Fried and Sida Wang and Tao Yu},
  journal={ArXiv},
  year={2022},
  volume={abs/2211.11501}
}
"""


def create_all_tasks():
    def create_task(key, mode):
        class DS1000(GeneralDS1000):
            def __init__(self, **kwargs):
                super().__init__(key, mode, **kwargs)

        return DS1000

    return {
        f"ds1000-{key.lower()}-{mode.lower()}": create_task(key, mode)
        for key in [
            "All",
            "Numpy",
            "Pandas",
            "Scipy",
            "Matplotlib",
            "Sklearn",
            "Tensorflow",
            "Pytorch",
        ]
        for mode in ["Completion", "Insertion"]
    }


class GeneralDS1000(Task):
    DATASET_PATH = None
    DATASET_NAME = None

    def __init__(self, key, mode):
        super().__init__(
            stop_words=["</code>", "# SOLUTION END"], requires_execution=True
        )
        self._key = key
        self._mode = mode
        if self._key == "Matplotlib" and self._mode == "Insertion":
            warnings.warn("Insertion not supported for Matplotlib. Running Completion.")
            self._mode = "Completion"
        self._dir = pathlib.Path(__file__).parent / "ds"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._src = self._dir / "ds1000.py"
        self._data = self._dir / "ds1000_data"
        self._download_source()
        self._download_dataset()

    def _download_source(self):
        url = "https://github.com/HKUNLP/DS-1000/blob/49c1c543ada8b58138181333cdc62e613204efcf/ds1000.py?raw=true"
        lock = self._src.with_suffix(".lock")
        with open(lock, "w") as f_lock:
            fcntl.flock(f_lock, fcntl.LOCK_EX)
            if not self._src.exists():
                warnings.warn(f"DS-1000 source is being saved to {self._src}.")
                print("Downloading source code...")
                r = requests.get(url, stream=True)
                with open(self._src, "wb") as f_src:
                    f_src.write(r.content)
                open(self._src.parent / "__init__.py", "w").close()
                print("Done.")
            fcntl.flock(f_lock, fcntl.LOCK_UN)

    def _download_dataset(self):
        url = "https://github.com/HKUNLP/DS-1000/blob/49c1c543ada8b58138181333cdc62e613204efcf/ds1000_data.zip?raw=true"
        lock = self._data.with_suffix(".lock")
        with open(lock, "w") as f_lock:
            fcntl.flock(f_lock, fcntl.LOCK_EX)
            if not self._data.exists():
                warnings.warn(f"DS-1000 data is being saved to {self._data}.")
                print("Downloading dataset...")
                r = requests.get(url, stream=True)
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(self._dir)
                print("Done.")
            fcntl.flock(f_lock, fcntl.LOCK_UN)

    @functools.lru_cache()
    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        from .ds.ds1000 import DS1000Dataset

        data = DS1000Dataset(self._data, mode=self._mode).data
        if self._key == "All":
            if self._mode == "Insertion":
                warnings.warn(
                    "Insertion not supported for Matplotlib. Only running others."
                )
                data = {k: v for k, v in data.items() if k != "Matplotlib"}
            dataset = list(itertools.chain(*data.values()))
        else:
            dataset = data[self._key]
        return dataset

    def get_prompt(self, doc):
        """
        Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str | dict[str: str]
        """
        if self._mode == "Completion":
            return doc["prompt"]
        elif self._mode == "Insertion":
            prefix, suffix = doc["prompt"].split("[insert]")
            prefix = f"{prefix.strip()}\n"
            suffix = f"\n{suffix.strip()}\n"
            return {"prefix": prefix, "suffix": suffix}
        else:
            raise ValueError(f"Invalid mode: {self._mode}")

    def get_reference(self, doc):
        """
        Builds the reference solution for the doc (sample from the test dataset).
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return doc["reference_code"]

    def postprocess_generation(self, generation, idx):
        """
        Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int (if needed)
            index of doc in the dataset to which the generation belongs
        :return: str
        """
        if self._mode == "Completion":
            for start in ["BEGIN SOLUTION\n<code>", "# SOLUTION START"]:
                try:
                    generation = generation.split(start, 1)[-1]
                except IndexError:
                    pass
        for stop in self.stop_words:
            generation = generation.split(stop)[0]
        return generation.strip()

    def process_results(self, generations, references):
        """
        Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations as in {"metric_name": result}.
        We encourage to directly load the metric from `evaluate` library to keep the code concise.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        :return: dict[str: float]
        """
        dataset = self.get_dataset()
        num_correct = 0
        print("Scoring generations...")
        for i, ref in tqdm.tqdm(enumerate(references), total=len(references)):
            test = [doc for doc in dataset if doc["reference_code"] == ref][0]
            for gen in generations[i]:
                is_correct = test.test(gen)
                if is_correct:
                    num_correct += 1
        accuracy = num_correct / len(references) / len(generations[0])
        return {f"mean pass@1 accuracy ({len(generations[0])} samples)": accuracy}
