class DatasetTemplate:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
