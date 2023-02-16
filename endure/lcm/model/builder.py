import torch
import logging

from endure.lcm.model.flexible_model import FlexibleModel
from endure.lcm.model.classic_model import ClassicModel


class LearnedCostModelBuilder:
    def __init__(self, config: dict[str, ...]):
        self._config = config
        self.log = logging.getLogger(self._config['log']['name'])
        self._models = {
            'Flexible': FlexibleModel,
            'Classic': ClassicModel,
        }

    @staticmethod
    def _get_default_arch():
        return 'Classic'

    def get_choices(self):
        return self._models.keys()

    def build_model(self, choice: str = None) -> torch.nn.Module:
        if choice is None:
            choice = self._config['lcm']['arch']
        self.log.info(f'Building model: {choice}')
        model = self._models.get(choice, None)
        if model is None:
            self.log.warn('Invalid model architecture. Defaulting to KCost')
            model = self._models.get(self._get_default_arch())
        model = model(self._config)

        return model
