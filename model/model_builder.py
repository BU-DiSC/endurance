import torch
import logging

from model.kcost import KCostModel
from model.tierlevelcost import TierLevelCost


class LearnedCostModelBuilder:
    def __init__(self, config: dict[str, ...]):
        self._config = config
        self.log = logging.getLogger(self._config['log']['name'])
        self._models = {
            'QCost': KCostModel,
            'TierLevelCost': TierLevelCost,
            'KCost': KCostModel
        }

    @staticmethod
    def _get_default_arch():
        return 'KCost'

    def get_choices(self):
        return self._models.keys()

    def build_model(self, choice: str = None) -> torch.nn.Module:
        if choice is None:
            choice = self._config['model']['arch']
        self.log.info(f'Building model: {choice}')
        model = self._models.get(choice, None)
        if model is None:
            self.log.warn('Invalid model architecture. Defaulting to KCost')
            model = self._models.get(self._get_default_arch())
        model = model(self._config)

        return model
