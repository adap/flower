from .base_folder import SharedFolder

class InMemoryFolder(SharedFolder):
    def __init__(self):
        self.store = {}

    def get(self, key, default=None):
        return self.store.get(key, default)

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        if key in self.store:
            del self.store[key]

    def __len__(self):
        return len(self.store)

    def items(self):
        return self.store.items()

    def get_raw_folder(self):
        return self
