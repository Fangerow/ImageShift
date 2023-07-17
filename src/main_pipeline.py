import torch
from pytorch_lightning import LightningModule, Trainer
from src.models.STN_model import Net
from config import cfg


class ShiftNetModule(LightningModule):
    def __init__(self):
        super(ShiftNetModule, self).__init__()
        self.model = Net()
        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.model(data)
        loss = self.criterion(output, target)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.model(data)
        loss = self.criterion(output, target)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=cfg["lr"])
        return optimizer


def run_train(train_loader, val_loader):
    model = ShiftNetModule()
    trainer = Trainer(max_epochs=cfg['max_epoch'])
    trainer.fit(model, train_loader, val_loader)



