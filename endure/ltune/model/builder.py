import torch
import logging

from endure.ltune.model.classic_model import ClassicTuner


class LTuneModelBuilder:
    def __init__(self, config: dict[str, ...]):
        self._config = config
        self.log = logging.getLogger(self._config['log']['name'])
        self._models = {
            'Tier': ClassicTuner,
            'Level': ClassicTuner,
        }

    @staticmethod
    def _get_default_arch():
        return 'Level'

    def get_choices(self):
        return self._models.keys()

    def build_model(self, choice: str = None) -> torch.nn.Module:
        if choice is None:
            choice = self._config['lsm']['design']

        model = self._models.get(choice, None)
        if model is None:
            self.log.warn('Invalid model architecture. Defaulting to KLSM')
            model = self._models.get(self._get_default_arch())
        model = model(self._config)

        return model