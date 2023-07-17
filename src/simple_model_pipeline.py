import torch.nn as nn
import torch
from pytorch_lightning import LightningModule, Trainer
from sklearn.metrics import mean_absolute_percentage_error
from config import cfg

class ShiftNet(LightningModule):
    def __init__(self):
        super(ShiftNet, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv.weight.data.fill_(0)
        self.conv.weight.data[:, :, 1, 1] = 1  # Инициализация ядра свертки так, чтобы оно представляло собой матрицу смещения
        self.loss = nn.MSELoss()

    def forward(self, x):
        return self.conv(x)

    def training_step(self, batch, batch_idx):
        src_imgs, tgt_imgs = batch
        src_imgs = src_imgs.to(cfg["device"])
        tgt_imgs = tgt_imgs.to(cfg["device"])
        output = self(src_imgs)
        loss = self.loss(output, tgt_imgs)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src_imgs, tgt_imgs = batch
        src_imgs = src_imgs.to(cfg["device"])
        tgt_imgs = tgt_imgs.to(cfg["device"])
        output = self(src_imgs)
        loss = self.loss(output, tgt_imgs)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        tgt_imgs_np = tgt_imgs.cpu().detach().numpy()
        output_np = output.cpu().detach().numpy()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=cfg["lr"])


def run_train(train_loader, val_loader):
    model = ShiftNet()
    trainer = Trainer(max_epochs=cfg['max_epoch'])
    trainer.fit(model, train_loader, val_loader)
