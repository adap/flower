class InMemoryFolder:
    def __init__(self):
        self.model_store = {}

    def get(self, key, default=None):
        return self.model_store[key] if key in self.model_store else default

    def __getitem__(self, key):
        return self.model_store[key]

    def __setitem__(self, key, value):
        self.model_store[key] = value

    def __delitem__(self, key):
        if key in self.model_store:
            del self.model_store[key]

    def __len__(self):
        return len(self.model_store)

    def items(self):
        return self.model_store.items()

    def get_raw_folder(self):
        return self
