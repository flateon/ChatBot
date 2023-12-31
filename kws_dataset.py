import os
import numpy as np
import librosa
import torch
import torchaudio


class DatasetTHCHS30(torch.utils.data.Dataset):
    def __init__(self, folder, split):
        self.folder = folder
        self.split = split
        self.VOCAB = np.load(os.path.join(self.folder, 'VOCAB.npy')).tolist()
        
        n_mels = 128
        self.input_dim = n_mels
        self.num_class = len(self.VOCAB)
        
        with open(os.path.join(self.folder, self.split, 'len.txt'), 'r') as f:
            self.len = int(f.readline())
    
    def __getitem__(self, index):
        data = np.load(os.path.join(self.folder, self.split, f'{index:>04}.npz'))
        return data['input'], data['input_length'], data['label'], data['label_length']
    
    def __len__(self):
        return self.len


class DatasetTHCHS30_raw(torch.utils.data.Dataset):
    def __init__(self, folder, split):
        self.folder = folder
        self.split = split
        self.VOCAB = np.load(os.path.join(self.folder, 'VOCAB.npy')).tolist()
        
        self.input_dim = None
        self.num_class = len(self.VOCAB)
        
        with open(os.path.join(self.folder, self.split, 'len.txt'), 'r') as f:
            self.len = int(f.readline())
    
    def __getitem__(self, index):
        data = np.load(os.path.join(self.folder, self.split, f'{index:>04}.npz'))
        return data['input'], data['input_length'], data['label'], data['label_length']
    
    def __len__(self):
        return self.len

class DatasetTHCHS30_old(torch.utils.data.Dataset):
    def __init__(self, folder, split):
        # folder = '../vakyansh-wav2vec2-experimentation/data/pretraining/data_thchs30'
        lm_phone_folder, lm_phone_file = 'lm_phone', 'lexicon.txt'
        train_folder, valid_folder, test_folder, data_folder = 'train', 'dev', 'test', 'data'
        if split == 'train':
            wav_folder = train_folder
        elif split == 'valid':
            wav_folder = valid_folder
        elif split == 'test':
            wav_folder = test_folder
        
        self.VOCAB = self.get_VOCAB(folder, lm_phone_folder, lm_phone_file)
        
        win_length, hop_length, n_mels = 640, 320, 128
        self.input_dim = n_mels
        self.num_class = len(self.VOCAB)
        self.inputs, self.labels = self.get_raw_inputs_labels(folder, wav_folder, data_folder, win_length, hop_length, n_mels)
        
        self.inputs_length = []
        max_inputs_length = max([i.shape[0] for i in self.inputs])
        for i, ipt in enumerate(self.inputs):
            self.inputs_length.append(ipt.shape[0])
            to_pad = max_inputs_length - ipt.shape[0]
            self.inputs[i] = np.pad(ipt, ((0,to_pad),(0,0)), mode='constant', constant_values=0)
        
        self.labels_length = []
        max_labels_length = max([len(l) for l in self.labels])
        for i, lbl in enumerate(self.labels):
            self.labels_length.append(lbl.shape[0])
            to_pad = max_labels_length - lbl.shape[0]
            self.labels[i] = np.pad(lbl, ((0,to_pad)), mode='constant', constant_values=0)
    
    def __getitem__(self, index):
        return self.inputs[index], self.inputs_length[index], self.labels[index], self.labels_length[index]
    
    def __len__(self):
        return len(self.labels)
    
    def get_VOCAB(self, folder, lm_phone_folder, lm_phone_file):
        file = os.path.join(folder, lm_phone_folder, lm_phone_file)
        VOCAB = []
        with open(file, 'r', encoding = 'utf-8') as f:
            lines = f.readlines()
            for line in lines:
                phone = line.split()[0]
                VOCAB.append(phone)
        return VOCAB
    
    def get_raw_inputs_labels(self, folder, wav_folder, data_folder, win_length, hop_length, n_mels):
        inputs = []
        labels = []
        files = os.listdir(os.path.join(folder, wav_folder))
        for file in files:
            # For every .wav, get name
            if file.endswith('wav'):
                name = file.split('.')[0]
                
                wav_path = os.path.join(folder, wav_folder, file)
                with open(wav_path, 'r', encoding = 'utf-8') as f:
                    # waveform: tensor (1, frames)
                    waveform, sample_rate = torchaudio.load(wav_path)
                    # mel: ndarray (1, n_wins, n_mels)
                    mel = librosa.feature.melspectrogram(y=waveform.squeeze(0).numpy(), sr=sample_rate, n_fft=win_length, n_mels=n_mels,
                                                        hop_length=hop_length, win_length=win_length)
                    mel = mel.transpose(1, 0)
                    mel = librosa.power_to_db(mel)
                    mel = librosa.util.normalize(mel)
                    inputs.append(mel)
                
                label_path = os.path.join(folder, data_folder, f"{name}.wav.trn")
                with open(label_path, 'r', encoding = 'utf-8') as f:
                    # Need <eps> explicitly or not?
                    # labels: ndarray (n_labels)
                    label_seq = np.array([self.VOCAB.index(s) for s in f.readlines()[2].strip().split()], dtype=int)
                    labels.append(label_seq)
        
        return inputs, labels


if __name__ == '__main__':
    trainDataset = DatasetTHCHS30('../vakyansh-wav2vec2-experimentation/data/pretraining/data_thchs30', 'train')
    validDataset = DatasetTHCHS30('../vakyansh-wav2vec2-experimentation/data/pretraining/data_thchs30', 'valid')
    testDataset  = DatasetTHCHS30('../vakyansh-wav2vec2-experimentation/data/pretraining/data_thchs30', 'test')
    
    print(len(trainDataset))
    print(len(validDataset))
    print(len(testDataset))
    
    input, input_length, label, label_length = trainDataset.__getitem__(114)
    print(input.shape, input_length, label.shape, label_length)
    input, input_length, label, label_length = validDataset.__getitem__(114)
    print(input.shape, input_length, label.shape, label_length)
    input, input_length, label, label_length = testDataset.__getitem__(114)
    print(input.shape, input_length, label.shape, label_length)

