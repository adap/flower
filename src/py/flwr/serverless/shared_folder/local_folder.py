from pathlib import Path
import pickle
import time
from typing import Any


class LocalFolderWithBytes:
    def __init__(
        self, directory: str = None, retry_sleep_time: int = 3, max_retry: int = 3
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.retry_sleep_time = retry_sleep_time
        self.max_retry = max_retry

    def _get_success_flag_file(self, key):
        return self.directory / ("success_" + key)

    def _delete_success_flag(self, key):
        filepath = self._get_success_flag_file(key)
        if filepath.exists():
            filepath.unlink()

    def _put_success_flag(self, key):
        filepath = self._get_success_flag_file(key)
        # create parent dir
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            f.write("")

    def get(self, key, default=None):
        success_flag_file = self._get_success_flag_file(key)
        patience = self.max_retry
        while not success_flag_file.exists():
            print(f"\nwaiting for success flag of {key}")
            time.sleep(self.retry_sleep_time)
            patience -= 1
            if patience == 0:
                return default
        filepath = self.directory / key
        if filepath.exists():
            with open(filepath, "rb") as f:
                return f.read()
        else:
            return default

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value: Any):
        assert isinstance(value, bytes), f"value must be bytes, but got {type(value)}"
        filepath = self.directory / key
        if value is None:
            raise ValueError("value must not be None")
        self._delete_success_flag(key)
        # create parent dir
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(value)
        self._put_success_flag(key)

    def __len__(self):
        # recursive
        return len(list(self.directory.glob("*")))

    def __delitem__(self, key):
        filepath = self.directory / key
        if filepath.exists():
            filepath.unlink()

    def items(self):
        for filepath in self.directory.glob("*"):
            # remove the directory name
            key = str(filepath)[len(str(self.directory)) + 1 :]
            yield key, self.get(key)


class LocalFolder:
    def __init__(
        self, directory: str = None, retry_sleep_time: int = 3, max_retry: int = 3
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.suffix = ".pkl"
        self.retry_sleep_time = retry_sleep_time
        self.max_retry = max_retry

    def get(self, key, default=None):
        success_flag_file = self._get_success_flag_file(key)
        patience = self.max_retry
        while not success_flag_file.exists():
            print(f"\nwaiting for success flag of {key}")
            time.sleep(self.retry_sleep_time)
            patience -= 1
            if patience == 0:
                return default
        filepath = self.directory / (key + self.suffix)
        if filepath.exists():
            with open(filepath, "rb") as f:
                return pickle.load(f)
        else:
            return default

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value: Any):
        filepath = self.directory / (key + self.suffix)
        if value is None:
            raise ValueError("value must not be None")
        self._delete_success_flag(key)
        with open(filepath, "wb") as f:
            pickle.dump(value, f)
        self._put_success_flag(key)

    def __delitem__(self, key):
        filepath = self.directory / (key + self.suffix)
        if filepath.exists():
            filepath.unlink()

    def _get_success_flag_file(self, key):
        return self.directory / ("success_" + key)

    def _delete_success_flag(self, key):
        filepath = self._get_success_flag_file(key)
        if filepath.exists():
            filepath.unlink()

    def _put_success_flag(self, key):
        filepath = self._get_success_flag_file(key)
        # create parent dir
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            f.write("")

    def __len__(self):
        return len(list(self.directory.glob(f"*{self.suffix}")))

    def items(self):
        for filepath in self.directory.glob(f"*{self.suffix}"):
            key_and_parameter = self.get_parameter(filepath)
            yield key_and_parameter

    def get_parameter(self, filepath):
        with open(filepath, "rb") as f:
            try:
                key = filepath.name[: -len(self.suffix)]
                parameters = self.get(key)
                return key, parameters
            except EOFError as e:
                print(f"EOFError: {e}")
                return None, None

    def get_raw_folder(self):
        """
        Creates a new LocalFolderWithBytes instance with the same directory.
        The "raw folder" is used to store raw bytes. This is different
        from the "regular" folder which stores pickled objects.
        """
        return LocalFolderWithBytes(
            directory=self.directory,
            retry_sleep_time=self.retry_sleep_time,
            max_retry=self.max_retry,
        )
