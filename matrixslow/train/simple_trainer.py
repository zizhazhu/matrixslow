from .trainer import Trainer


class SimpleTrainer(Trainer):

    def __init__(self, *args, **kargs):
        super(SimpleTrainer, self).__init__(*args, **kargs)

    def _variable_weights_init(self):
        pass

    def _optimizer_update(self):
        self.optimizer.update()
