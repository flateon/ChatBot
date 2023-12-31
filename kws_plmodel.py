import os
import random

import wandb
import torch
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
import numpy as np

from kws_model import CTCEncoderDecoder, Wav2Vec2Finetuner
from kws_dataset import DatasetTHCHS30_raw
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

def edit_distance(src_seq, tgt_seq):
    src_len, tgt_len = len(src_seq), len(tgt_seq)
    if src_len == 0: return tgt_len
    if tgt_len == 0: return src_len

    dist = np.zeros((src_len+1, tgt_len+1))
    for i in range(1, tgt_len+1):
        dist[0, i] = dist[0, i-1] + 1
    for i in range(1, src_len+1):
        dist[i, 0] = dist[i-1, 0] + 1
    for i in range(1, src_len+1):
        for j in range(1, tgt_len+1):
            cost = 0 if src_seq[i-1] == tgt_seq[j-1] else 1
            dist[i, j] = min(
                dist[i,j-1]+1,
                dist[i-1,j]+1,
                dist[i-1,j-1]+cost,
            )
    return dist


def get_cer_per_sample(hypotheses, hypothesis_lengths, references, reference_lengths):
    assert len(hypotheses) == len(references)
    cer = []
    for i in range(len(hypotheses)):
        # print(len(hypotheses[i]))
        if len(hypotheses[i]) > 0:
            dist_i = edit_distance(hypotheses[i][:hypothesis_lengths[i]],
                                    references[i][:reference_lengths[i]])
            # CER divides the edit distance by the length of the true sequence
            # print(dist_i[-1, -1], float(reference_lengths[i]))
            cer.append((dist_i[-1, -1] / float(reference_lengths[i])))
        else:
            # print("Miss")
            cer.append(1)  # since we predicted empty 
    return np.array(cer)


class LightningCTC(pl.LightningModule):
    def __init__(self, n_mels=128, n_fft=256, win_length=256, hop_length=128,
                wav_max_length=200, transcript_max_length=200,
                learning_rate=1e-3, batch_size=256, weight_decay=1e-5,
                encoder_num_layers=2, encoder_hidden_dim=256,
                encoder_bidirectional=True):
        super().__init__()
        self.save_hyperparameters()
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.lr = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.wav_max_length = wav_max_length
        self.transcript_max_length = transcript_max_length
        self.train_dataset, self.val_dataset, self.test_dataset = self.create_datasets()
        self.encoder_num_layers = encoder_num_layers
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_bidirectional = encoder_bidirectional
        
        # Instantiate the CTC encoder/decoder.
        self.model = self.create_model()
        
        self.validation_outputs = []
        self.test_outputs = []
    
    def create_model(self):
        # model = CTCEncoderDecoder(
        #     self.train_dataset.input_dim,
        #     self.train_dataset.num_class,
        #     num_layers=self.encoder_num_layers,
        #     hidden_dim=self.encoder_hidden_dim,
        #     bidirectional=self.encoder_bidirectional)
        model = Wav2Vec2Finetuner(self.train_dataset.input_dim, self.train_dataset.num_class)
        return model
    
    def create_datasets(self):
        root = './Dataset/raw'
        train_dataset = DatasetTHCHS30_raw(root, split='train')
        # train_dataset = DatasetTHCHS30(root, split='valid')
        val_dataset = DatasetTHCHS30_raw(root, split='valid')
        # test_dataset = DatasetTHCHS30(root, split='valid')
        test_dataset = DatasetTHCHS30_raw(root, split='test')
        return train_dataset, val_dataset, test_dataset
    
    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.model.parameters(),
                                lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, 10, 0.1)
        # return [optim], [scheduler] # <-- put scheduler in here if you want to use one
        return [optim], [] # <-- put scheduler in here if you want to use one
    
    def get_loss(self, log_probs, input_lengths, labels, label_lengths):
        loss = self.model.get_loss(log_probs, labels, input_lengths, label_lengths,0)
                                    # blank=self.train_dataset.eps_index)
        return loss
    
    def forward(self, inputs, input_lengths, labels, label_lengths):
        log_probs, embedding = self.model(inputs, input_lengths)
        return log_probs, embedding
    
    def get_primary_task_loss(self, batch, split='train'):
        """Returns ASR model losses, metrics, and embeddings for a batch."""
        inputs, input_lengths = batch[0], batch[1]
        labels, label_lengths = batch[2], batch[3]
        
        if split == 'train':
            log_probs, embedding = self.forward(
                inputs, input_lengths, labels, label_lengths)
        else:
            # do not pass labels to not teacher force after training
            log_probs, embedding = self.forward(
                inputs, input_lengths, None, None)
        
        loss = self.get_loss(log_probs, input_lengths, labels, label_lengths)
        
        # Compute CER (no gradient necessary).
        with torch.no_grad():
            hypotheses, hypothesis_lengths, references, reference_lengths = \
                self.model.decode(
                    log_probs, input_lengths, labels, label_lengths)
                    # self.train_dataset.sos_index,
                    # self.train_dataset.eos_index,
                    # self.train_dataset.pad_index,
                    # self.train_dataset.eps_index)
            cer_per_sample = get_cer_per_sample(
                hypotheses, hypothesis_lengths, references, reference_lengths)
            cer = cer_per_sample.mean()
            metrics = {f'{split}_loss': loss, f'{split}_cer': cer}
        
        return loss, metrics, embedding
    
    # Overwrite TRAIN
    def training_step(self, batch, batch_idx):
        loss, metrics, _ = self.get_primary_task_loss(batch, split='train')
        self.log_dict(metrics)
        # self.log('train_loss', loss, prog_bar=True, on_step=True)
        # self.log('train_cer', metrics['train_cer'], prog_bar=True, on_step=True)
        return loss
    
    # Overwrite VALIDATION: get next minibatch
    def validation_step(self, batch, batch_idx):
        loss, metrics, _ = self.get_primary_task_loss(batch, split='val')
        self.validation_outputs.append(metrics)
        return metrics
    
    def test_step(self, batch, batch_idx):
        _, metrics, _ = self.get_primary_task_loss(batch, split='test')
        self.test_outputs.append(metrics)
        return metrics
    
    # Overwrite: e.g. accumulate stats (avg over CER and loss)
    def on_validation_epoch_end(self):
        """Called at the end of validation step to aggregate outputs."""
        # outputs is list of metrics from every validation_step (over a
        # validation epoch).
        outputs = self.validation_outputs
        metrics = {
            # important that these are torch Tensors!
            'val_loss': torch.tensor([elem['val_loss']
                                        for elem in outputs]).float().mean(),
            'val_cer': torch.tensor([elem['val_cer']
                                        for elem in outputs]).float().mean()
        }
        # self.log('val_loss', metrics['val_loss'], prog_bar=True)
        # self.log('val_cer', metrics['val_cer'], prog_bar=True)
        self.log_dict(metrics)
        self.validation_outputs.clear()
    
    def on_test_epoch_end(self):
        outputs = self.test_outputs
        metrics = {
            'test_loss': torch.tensor([elem['test_loss']
                                        for elem in outputs]).float().mean(),
            'test_cer': torch.tensor([elem['test_cer']
                                        for elem in outputs]).float().mean()
        }
        self.log_dict(metrics)
        self.test_outputs.clear()
    
    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=1,
                            shuffle=True, pin_memory=True, drop_last=True)
        return loader
    
    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=1,
                            shuffle=False, pin_memory=True)
        return loader
    
    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=1,
                            shuffle=False, pin_memory=True)
        return loader




def run(system, config, ckpt_dir, epochs=1, monitor_key='val_loss',
        use_gpu=False, seed=1337):
    
    WANDB_NAME = '2553036255'
    MODEL_PATH = './Checkpoints'
    
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    SystemClass = globals()[system]
    system = SystemClass(**config)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(MODEL_PATH, ckpt_dir),
        save_top_k=1,
        verbose=True,
        monitor=monitor_key,
        mode='min')
    
    wandb.init(project='HW3', entity=WANDB_NAME, name=ckpt_dir,
                config=config, sync_tensorboard=True)
    wandb_logger = WandbLogger()
    
    if use_gpu:
        trainer = pl.Trainer(
            max_epochs=epochs, min_epochs=epochs, enable_checkpointing=True,
            callbacks=checkpoint_callback, logger=wandb_logger, accelerator='auto')
    else:
        trainer = pl.Trainer(
            max_epochs=epochs, min_epochs=epochs, enable_checkpointing=True,
            callbacks=checkpoint_callback, logger=wandb_logger)
    
    trainer.fit(system)
    result = trainer.test()


if __name__ == '__main__':
    config = {
        'n_mels': 128,
        'n_fft': 640,
        'win_length': 640,
        'hop_length': 320,
        'wav_max_length': 782,
        'transcript_max_length': 94,
        'learning_rate': 1e-5,
        'batch_size': 1,
        'weight_decay': 0,
        'encoder_num_layers': 4,
        'encoder_hidden_dim': 256,
        'encoder_bidirectional': True,
    }
    
    run(system="LightningCTC", config=config, ckpt_dir='Finetune_test', epochs=10)

