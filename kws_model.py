import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np


def get_ctc_loss(
    log_probs, targets, input_lengths, target_lengths, blank=0):
    """Connectionist Temporal Classification objective function."""
    ctc_loss = None
    log_probs = log_probs.contiguous()
    targets = targets.long()
    input_lengths = input_lengths.long()
    target_lengths = target_lengths.long()
    ############################ START OF YOUR CODE ############################
    # TODO(2.1)
    # Hint:
    # - `F.ctc_loss`: https://pytorch.org/docs/stable/nn.functional.html#ctc-loss
    # - log_probs is passed in with shape (batch_size, input_length, num_classes).
    # - Notice that `F.ctc_loss` expects log_probs of shape
    #   (input_length, batch_size, num_classes)
    # - Turn on zero_infinity.
    log_probs = log_probs.permute(1, 0, 2)
    ctc_loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, zero_infinity=True)
    ############################# END OF YOUR CODE #############################
    return ctc_loss


class CTCEncoderDecoder(nn.Module):
    def __init__(self, input_dim, num_class, num_layers=2, hidden_dim=128, bidirectional=True):
        super().__init__()
        # Note: `batch_first=True` argument implies the inputs to the LSTM should
        # be of shape (batch_size x T x D) instead of (T x batch_size x D).
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        # self.decoder = nn.Linear(hidden_dim * 2, num_class)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_class),
        )
        
        self.input_dim = input_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = hidden_dim * num_layers * 2 * (2 if bidirectional else 1)
    
    def combine_h_and_c(self, h, c):
        """Combine the signals from RNN hidden and cell states."""
        batch_size = h.size(1)
        h = h.permute(1, 0, 2).contiguous()
        c = c.permute(1, 0, 2).contiguous()
        h = h.view(batch_size, -1)
        c = c.view(batch_size, -1)
        return torch.cat([h, c], dim=1)  # just concatenate
    
    def forward(self, inputs, input_lengths):
        batch_size, max_length, _ = inputs.size()
        # `torch.nn.utils.rnn.pack_padded_sequence` collapses padded sequences
        # to a contiguous chunk
        inputs = torch.nn.utils.rnn.pack_padded_sequence(
            inputs, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
        log_probs = None
        h, c = None, None
        ############################ START OF YOUR CODE ############################
        # TODO(2.1)
        # Hint:
        # - Refer to https://pytorch.org/docs/stable/nn.html
        # - Use `self.encoder` to get the encodings output which is of shape
        #   (batch_size, max_length, num_directions*hidden_dim) and the
        #   hidden states and cell states which are both of shape
        #   (batch_size, num_layers*num_directions, hidden_dim)
        # - Pad outputs with `0.` using `torch.nn.utils.rnn.pad_packed_sequence`
        #   (turn on batch_first and set total_length as max_length).
        # - Apply 50% dropout.
        # - Use `self.decoder` to take the embeddings sequence and return
        #   probabilities for each character.
        # - Make sure to then convert to log probabilities.
        x, (h, c) = self.encoder(inputs)
        # print("hc type shape", type(h), h.shape, type(c), c.shape)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=max_length)
        # x = torch.nn.utils.rnn.pack_padded_sequence(
        #     x, (max_length*torch.ones_like(input_lengths)).cpu(), batch_first=True, enforce_sorted=True)
        x = nn.Dropout()(x)
        x = self.decoder(x)
        x = nn.Softmax(2)(x)
        log_probs = torch.log(x)
        # print("primary log", log_probs)
        ############################# END OF YOUR CODE #############################
        
        # The extracted embedding is not used for the ASR task but will be
        # needed for other auxiliary tasks.
        embedding = self.combine_h_and_c(h, c)
        # print(h.shape, c.shape, embedding.shape)
        return log_probs, embedding
    
    def get_loss(
        self, log_probs, targets, input_lengths, target_lengths, blank=0):
        return get_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank)
    
    def decode(self, log_probs, input_lengths, labels, label_lengths):
        # Use greedy decoding.
        decoded = torch.argmax(log_probs, dim=2)
        batch_size = decoded.size(0)
        # Collapse each decoded sequence using CTC rules.
        hypotheses = []
        for i in range(batch_size):
            hypotheses_i = self.ctc_collapse(decoded[i], input_lengths[i].item())
                                            # blank_index=eps_index)
            hypotheses.append(hypotheses_i)
        
        hypothesis_lengths = input_lengths.cpu().numpy().tolist()
        if labels is None: # Run at inference time.
            references, reference_lengths = None, None
        else:
            references = labels.cpu().numpy().tolist()
            reference_lengths = label_lengths.cpu().numpy().tolist()
        
        return hypotheses, hypothesis_lengths, references, reference_lengths
    
    def ctc_collapse(self, seq, seq_len, blank_index=0):
        result = []
        for i, tok in enumerate(seq[:seq_len]):
            if tok.item() != blank_index:  # remove blanks
                if i != 0 and tok.item() == seq[i-1].item():  # remove dups
                    pass
                else:
                    result.append(tok.item())
        return result

class Wav2Vec2Finetuner(nn.Module):
    def __init__(self, input_dim, num_class):
        super().__init__()
        # Note: `batch_first=True` argument implies the inputs to the LSTM should
        # be of shape (batch_size x T x D) instead of (T x batch_size x D).
        # self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.encoder = bundle.get_model()
        self.decoder = nn.Linear(768, num_class)
        
        self.input_dim = input_dim
        self.num_class = num_class
    
    def forward(self, inputs, input_lengths):
        features, _ = self.encoder(inputs)
        logits = self.decoder(features)
        logprobs = nn.LogSoftmax(2)(logits)
        return logprobs, None
    
    def get_loss(
        self, log_probs, targets, input_lengths, target_lengths, blank=0):
        # print(log_probs.shape, targets.shape, input_lengths, target_lengths)
        input_lengths = torch.tensor([x.shape[0] for x in log_probs], device=input_lengths.device, dtype=input_lengths.dtype)
        # print(log_probs.shape, targets.shape, input_lengths, target_lengths)
        return get_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank)
    
    def decode(self, log_probs, input_lengths, labels, label_lengths):
        # Use greedy decoding.
        decoded = torch.argmax(log_probs, dim=2)
        batch_size = decoded.size(0)
        # Collapse each decoded sequence using CTC rules.
        hypotheses = []
        for i in range(batch_size):
            hypotheses_i = self.ctc_collapse(decoded[i], decoded[i].shape[0])
            # hypotheses_i = self.ctc_collapse(decoded[i], input_lengths[i].item())
                                            # blank_index=eps_index)
            hypotheses.append(hypotheses_i)
        
        hypothesis_lengths = [decoded[i].shape[0]]
        # hypothesis_lengths = input_lengths.cpu().numpy().tolist()
        if labels is None: # Run at inference time.
            references, reference_lengths = None, None
        else:
            references = labels.cpu().numpy().tolist()
            reference_lengths = label_lengths.cpu().numpy().tolist()
        
        return hypotheses, hypothesis_lengths, references, reference_lengths
    
    def ctc_collapse(self, seq, seq_len, blank_index=0):
        result = []
        for i, tok in enumerate(seq[:seq_len]):
            if tok.item() != blank_index:  # remove blanks
                if i != 0 and tok.item() == seq[i-1].item():  # remove dups
                    pass
                else:
                    result.append(tok.item())
        return result
