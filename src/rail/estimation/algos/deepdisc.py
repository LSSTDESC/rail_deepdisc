from rail.estimation.estimator import CatEstimator, CatInformer

class DeepDiscInformer(CatInformer):
    """Placeholder for informer stage class"""

    name = "DeepDiscInformer"
    config_options = CatInformer.config_options.copy()

    def __init__(self, args, comm=None):
        CatInformer.__init__(self, args, comm=comm)

    def run(self):
        pass


class DeepDiscEstimator(CatEstimator):
    name = "DeepDiscEstimator"
    config_options = CatEstimator.config_options.copy()

    def __init__(self, args, comm=None):
        CatEstimator.__init__(self, args, comm=comm)

    def _process_chunk(self, start, end, data, first):
        pass
