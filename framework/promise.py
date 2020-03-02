class TensorPromise:
    def __init__(self, lazy_module: "LazyModule", index: int):
        self.lazy_module = lazy_module
        self.index = index

    def fetch(self) -> Any:
        return self.lazy_module.fetch(self.index)


class OneDimTensorPromise:
    def __init__(self):
        self.callbacks = []
        self.result = None

    def computed(self, result):
        self.result = result
        for callback in self.callbacks:
            callback(result)

    def after_computed(self, callback):
        self.callbacks.append(callback)


class TwoDimTensorPromise:
    pass
