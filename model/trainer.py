import logging
import torch
from tqdm import tqdm


class Trainer():
    def __init__(self, env_config, dataloader, model, optimizer, loss_fn,
                 validate_percentage=0.1, batch_size=32, early_term_iter=4):
        self.env_config = env_config
        self.log = logging.getLogger(env_config['log_name'])
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.validate_percentage = validate_percentage

        val_len = int(validate_percentage * len(dataloader))
        train_len = len(dataloader) - val_len
        train, val = torch.utils.random_split(dataloader, (train_len, val_len))
        self.train = torch.utils.data.DataLoader(
                train,
                batch_size=batch_size,
                shuffle=True)
        self.val = torch.utils.data.DataLoader(
                val,
                batch_size=batch_size,
                shuffle=False)

    def _train_loop(self, disp_pbar=False):
        self.model.train()
        if disp_pbar:
            pbar = tqdm(
                    self.train,
                    bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}')
        else:
            pbar = self.train

        for batch, (x, y) in enumerate(pbar):
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (batch % (10000) == 0) and (disp_pbar):
                pbar.set_description(f'loss: {loss:>7f}')

    def _validate_loop(self):
        num_batches = len(self.val)
        test_loss = 0

        self.model.eval()
        with torch.no_grad():
            for x, y in self.val:
                pred = self.model(x)
                test_loss += self.loss_fn(pred, y).item()

        test_loss /= num_batches
        self.log.info(f'validation loss: {test_loss:>8f}')

        return test_loss

    def train(self, max_epochs, disp_pbar=False):
        prev_loss = float('inf')
        for t in range(max_epochs):
            self.log.info(f"Epoch {t+1}/{max_epochs}")
            self._train_loop(disp_pbar)
            curr_loss = self._validate_loop()

            # TODO: implment better early stop
            if curr_loss > prev_loss:
                self.log.info('Loss increasing, stopping and saving model')
                torch.save(self.model.state_dict(), 'kcost.model')
                break
            self.log.info('Check pointing')
            torch.save({
                'epoch': t,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': curr_loss,
                },
                'kcost.pt')
            prev_loss = curr_loss

        return
