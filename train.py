import torch 
from torch import nn
from torch import optim
from torch.utils import data as Data
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from itertools import chain

# Hyperparameter
d_model = 512 # embedding size 
max_len = 1024 # max length of sequence
d_ff = 2048 # feedforward nerual network  dimension
d_k = d_v = 64 # dimension of k(same as q) and v
n_layers = 6 # number of encoder and decoder layers
n_heads = 8 # number of heads in multihead attention
p_drop = 0.1 # propability of dropout

def get_attn_pad_mask(seq_q, seq_k, dec=False):
    if dec: pad = dec_pad 
    else: pad = enc_pad 
    '''
    Padding, because of unequal in source_len and target_len.

    parameters:
    seq_q: [batch, seq_len]
    seq_k: [batch, seq_len]

    return:
    mask: [batch, len_q, len_k]

    '''
    batch, len_q = seq_q.size()
    batch, len_k = seq_k.size()
    # search pad according to pad id.
    pad_attn_mask = seq_k.data.eq(pad).unsqueeze(1) # [batch, 1, len_k]
    return pad_attn_mask.expand(batch, len_q, len_k) # [batch, len_q, len_k]

def get_attn_subsequent_mask(seq):
    '''
    Build attention mask matrix for decoder when it autoregressing.

    parameters:
    seq: [batch, target_len]

    return:
    subsequent_mask: [batch, target_len, target_len] 
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)] # [batch, target_len, target_len]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1) # [batch, target_len, target_len] 
    subsequent_mask = torch.from_numpy(subsequent_mask).to(device=device)

    return subsequent_mask # [batch, target_len, target_len] 


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=p_drop)

        positional_encoding = torch.zeros(max_len, d_model) # [max_len, d_model]
        position = torch.arange(0, max_len).float().unsqueeze(1) # [max_len, 1]

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.Tensor([10000])) / d_model)) # [max_len / 2]

        positional_encoding[:, 0::2] = torch.sin(position * div_term) # even
        positional_encoding[:, 1::2] = torch.cos(position * div_term) # odd

        # [max_len, d_model] -> [1, max_len, d_model] -> [max_len, 1, d_model]
        positional_encoding = positional_encoding.unsqueeze(0).transpose(0, 1)

        # register pe to buffer and require no grads
        self.register_buffer('pe', positional_encoding)

    def forward(self, x):
        # x: [seq_len, batch, d_model]
        # we can add positional encoding to x directly, and ignore other dimension
        x = x + self.pe[:x.size(0), ...]
        return self.dropout(x)

class FeedForwardNetwork(nn.Module):
  '''
  Using nn.Conv1d replace nn.Linear to implements FFN.
  '''
  def __init__(self):
    super(FeedForwardNetwork, self).__init__()
    # self.ff1 = nn.Linear(d_model, d_ff)
    # self.ff2 = nn.Linear(d_ff, d_model)
    self.ff1 = nn.Conv1d(d_model, d_ff, 1)
    self.ff2 = nn.Conv1d(d_ff, d_model, 1)
    self.relu = nn.ReLU()

    self.dropout = nn.Dropout(p=p_drop)
    self.layer_norm = nn.LayerNorm(d_model)

  def forward(self, x):
    # x: [batch, seq_len, d_model]
    residual = x
    x = x.transpose(1, 2) # [batch, d_model, seq_len]
    x = self.ff1(x)
    x = self.relu(x)
    x = self.ff2(x)
    x = x.transpose(1, 2) # [batch, seq_len, d_model]

    # sAdd & Norm
    return self.layer_norm(residual + x)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        # do not use more instance to implement multihead attention
        # it can be complete in one matrix
        self.n_heads = n_heads

        # we can't use bias because there is no bias term in formular
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.W_O = nn.Linear(d_v * n_heads, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        To make sure multihead attention can be used both in encoder and decoder, 
        we use Q, K, V respectively.
        input_Q: [batch, len_q, d_model]
        input_K: [batch, len_k, d_model]
        input_V: [batch, len_v, d_model]
        '''
        residual, batch = input_Q, input_Q.size(0)

        # [batch, len_q, d_model] -- matmul W_Q --> [batch, len_q, d_q * n_heads] -- view --> 
        # [batch, len_q, n_heads, d_k,] -- transpose --> [batch, n_heads, len_q, d_k]

        Q = self.W_Q(input_Q).view(batch, -1, n_heads, d_k).transpose(1, 2) # [batch, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch, -1, n_heads, d_k).transpose(1, 2) # [batch, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch, -1, n_heads, d_v).transpose(1, 2) # [batch, n_heads, len_v, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # [batch, n_heads, seq_len, seq_len]

        # prob: [batch, n_heads, len_q, d_v] attn: [batch, n_heads, len_q, len_k]
        prob, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)

        # concat
        prob = prob.transpose(1, 2).contiguous() # [batch, len_q, n_heads, d_v]
        prob = prob.view(batch, -1, n_heads * d_v).contiguous() # [batch, len_q, n_heads * d_v]

        output = self.W_O(prob) # [batch, len_q, d_model]

        # Add & Norm
        return self.layer_norm(residual + output), attn

class ScaledDotProductAttention(nn.Module):
  def __init__(self):
    super(ScaledDotProductAttention, self).__init__()

  def forward(self, Q, K, V, attn_mask):
    '''
    Q: [batch, n_heads, len_q, d_k]
    K: [batch, n_heads, len_k, d_k]
    V: [batch, n_heads, len_v, d_v]
    attn_mask: [batch, n_heads, seq_len, seq_len]
    '''
    scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # [batch, n_heads, len_q, len_k]
    scores.masked_fill_(attn_mask, -1e9)

    attn = nn.Softmax(dim=-1)(scores) # [batch, n_heads, len_q, len_k]
    prob = torch.matmul(attn, V) # [batch, n_heads, len_q, d_v]
    return prob, attn

class EncoderLayer(nn.Module):

    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.encoder_self_attn = MultiHeadAttention()
        self.ffn = FeedForwardNetwork()

    def forward(self, encoder_input, encoder_pad_mask):
        '''
        encoder_input: [batch, source_len, d_model]
        encoder_pad_mask: [batch, n_heads, source_len, source_len]

        encoder_output: [batch, source_len, d_model]
        attn: [batch, n_heads, source_len, source_len]
        '''
        encoder_output, attn = self.encoder_self_attn(encoder_input, encoder_input, encoder_input, encoder_pad_mask)
        encoder_output = self.ffn(encoder_output) # [batch, source_len, d_model]

        return encoder_output, attn
    
class Encoder(nn.Module):

  def __init__(self):
    super(Encoder, self).__init__()
    self.source_embedding = nn.Embedding(source_vocab_size, d_model)
    self.positional_embedding = PositionalEncoding(d_model)
    self.encoders = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

  def forward(self, encoder_input):
    # encoder_input: [batch, source_len]
    encoder_output = self.source_embedding(encoder_input) # [batch, source_len, d_model]
    encoder_output = self.positional_embedding(encoder_output.transpose(0, 1)).transpose(0, 1) # [batch, source_len, d_model]
    # 对于encoder, pad需要mask
    encoder_self_attn_mask = get_attn_pad_mask(encoder_input, encoder_input) # [batch, source_len, source_len]
    encoder_self_attns = list()
    for encoder in self.encoders:
      # encoder_output: [batch, source_len, d_model]
      # encoder_self_attn: [batch, n_heads, source_len, source_len]
      encoder_output, encoder_self_attn = encoder(encoder_output, encoder_self_attn_mask)
      encoder_self_attns.append(encoder_self_attn)

    return encoder_output, encoder_self_attns

class DecoderLayer(nn.Module):

  def __init__(self):
    super(DecoderLayer, self).__init__()
    self.decoder_self_attn = MultiHeadAttention()
    self.encoder_decoder_attn = MultiHeadAttention()
    self.ffn = FeedForwardNetwork()

  def forward(self, decoder_input, encoder_output, decoder_self_mask, decoder_encoder_mask):
    '''
    decoder_input: [batch, target_len, d_mdoel]
    encoder_output: [batch, source_len, d_model]
    decoder_self_mask: [batch, target_len, target_len]
    decoder_encoder_mask: [batch, target_len, source_len]
    '''
    # masked mutlihead attention
    # Q, K, V all from decoder it self
    # decoder_output: [batch, target_len, d_model]
    # decoder_self_attn: [batch, n_heads, target_len, target_len]
    decoder_output, decoder_self_attn = self.decoder_self_attn(decoder_input, decoder_input, decoder_input, decoder_self_mask)

    # Q from decoder, K, V from encoder
    # decoder_output: [batch, target_len, d_model]
    # decoder_encoder_attn: [batch, n_heads, target_len, source_len]
    decoder_output, decoder_encoder_attn = self.encoder_decoder_attn(decoder_output, encoder_output, encoder_output, decoder_encoder_mask)
    decoder_output = self.ffn(decoder_output) # [batch, target_len, d_model]

    return decoder_output, decoder_self_attn, decoder_encoder_attn

class Decoder(nn.Module):

  def __init__(self):
    super(Decoder, self).__init__()
    self.target_embedding = nn.Embedding(target_vocab_size, d_model)
    self.positional_embedding = PositionalEncoding(d_model)
    self.decoders = nn.ModuleList([DecoderLayer() for layer in range(n_layers)])

  def forward(self, decoder_input, encoder_input, encoder_output):
    '''
    decoder_input: [batch, target_len]
    encoder_input: [batch, source_len]
    encoder_output: [batch, source_len, d_model]
    '''
    decoder_output = self.target_embedding(decoder_input) # [batch, target_len, d_model]
    decoder_output = self.positional_embedding(decoder_output.transpose(0, 1)).transpose(0, 1) # [batch, target_len, d_model]
    # 1. 将PAD掩住
    decoder_self_attn_mask = get_attn_pad_mask(decoder_input, decoder_input, dec=True) # [batch, target_len, target_len]
    # 2. 将下一个值掩住，上三角
    decoder_subsequent_mask = get_attn_subsequent_mask(decoder_input) # [batch, target_len, target_len]
    decoder_encoder_attn_mask = get_attn_pad_mask(decoder_input, encoder_input, dec=True) # [batch, target_len, source_len]

    decoder_self_mask = torch.gt(decoder_self_attn_mask + decoder_subsequent_mask, 0)
    decoder_self_attns, decoder_encoder_attns = [], []

    for decoder in self.decoders:
      # decoder_output: [batch, target_len, d_model]
      # decoder_self_attn: [batch, n_heads, target_len, target_len]
      # decoder_encoder_attn: [batch, n_heads, target_len, source_len]
      decoder_output, decoder_self_attn, decoder_encoder_attn = decoder(decoder_output, encoder_output, decoder_self_mask, decoder_encoder_attn_mask)
      decoder_self_attns.append(decoder_self_attn)
      decoder_encoder_attns.append(decoder_encoder_attn)

    return decoder_output, decoder_self_attns, decoder_encoder_attns
  
class Transformer(nn.Module):

  def __init__(self):
    super(Transformer, self).__init__()

    self.encoder = Encoder()
    self.decoder = Decoder()
    self.lm_head = nn.Linear(d_model, target_vocab_size, bias=False)
    print("是否训练：", self.training)

  def forward(self, encoder_input, decoder_input):
    '''
    encoder_input: [batch, source_len]
    decoder_input: [batch, target_len]
    '''
    # encoder_output: [batch, source_len, d_model]
    # encoder_attns: [n_layers, batch, n_heads, source_len, source_len]
    encoder_output, encoder_attns = self.encoder(encoder_input)
    # decoder_output: [batch, target_len, d_model]
    # decoder_self_attns: [n_layers, batch, n_heads, target_len, target_len]
    # decoder_encoder_attns: [n_layers, batch, n_heads, target_len, source_len]
    decoder_output, decoder_self_attns, decoder_encoder_attns = self.decoder(decoder_input, encoder_input, encoder_output)
    decoder_logits = self.lm_head(decoder_output) # [batch, target_len, target_vocab_size]

    # decoder_logits: [batch * target_len, target_vocab_size]
    return decoder_logits.view(-1, decoder_logits.size(-1)), encoder_attns, decoder_self_attns, decoder_encoder_attns
  
def tokenizer(sentences):
    enc_inputs = [sentence[0] for sentence in sentences]
    enc_words = list(chain(*[sentence.split() for sentence in enc_inputs]))
    enc_vocab = {word: idx for idx, word in enumerate(set(enc_words))}
    enc_vocab["pad"] = len(enc_vocab) 

    dec_inputs = [sentence[1] for sentence in sentences]
    dec_words = list(chain(*[sentence.split() for sentence in dec_inputs]))
    dec_vocab = {word: idx for idx, word in enumerate(set(dec_words))}
    dec_vocab['sos'] = len(dec_vocab) 
    dec_vocab['eos'] = len(dec_vocab)


    encoder_inputs, decoder_inputs, decoder_outputs = [], [], []
    # 1. convert sentences to ids according the vocabulary.
    for i in range(len(sentences)):
        encoder_input = [enc_vocab[word] for word in sentences[i][0].split()]
        target = [dec_vocab[word] for word in sentences[i][1].split()]
        decoder_input =  [dec_vocab['sos']] + target
        decoder_output = target + [dec_vocab['eos']] 
        encoder_inputs.append(encoder_input)
        decoder_inputs.append(decoder_input)
        decoder_outputs.append(decoder_output)

    # 2. pad zero to all sentence to the length of longest sentence
    encoder_inputs = [torch.tensor(ids) for ids in encoder_inputs]
    encoder_inputs = pad_sequence(encoder_inputs, batch_first=True, padding_value=enc_vocab["pad"])

    decoder_inputs = [torch.tensor(ids) for ids in decoder_inputs]
    decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=dec_vocab['eos'])

    decoder_outputs = [torch.tensor(ids) for ids in decoder_outputs]
    decoder_outputs = pad_sequence(decoder_outputs, batch_first=True, padding_value=dec_vocab['eos'])

    # 3. return ids with padding
    return torch.LongTensor(encoder_inputs), torch.LongTensor(decoder_inputs), torch.LongTensor(decoder_outputs), len(enc_vocab), len(dec_vocab), enc_vocab["pad"], dec_vocab["eos"]

class Seq2SeqDataset(Data.Dataset):
    def __init__(self, encoder_input, decoder_input, decoder_output):
        super(Seq2SeqDataset, self).__init__()
        self.encoder_input = encoder_input # [num_sentences, num_tokens]
        self.decoder_input = decoder_input
        self.decoder_output = decoder_output

    def __len__(self):
        return self.encoder_input.shape[0]

    def __getitem__(self, idx):
        return self.encoder_input[idx], self.decoder_input[idx], self.decoder_output[idx]

sentences = [
    # enc_input            dec_output         
    ['ich mochte ein bier .', 'i want a beer . '],
    ['ich mochte ein gut cola .', 'i want a good coke .'],
    ['ich möchte ein buch .', 'i want a book .'],
    ['werden sie morgen im büro sein ?', 'will you be in the office tomorrow ?'],
    ['ich liebe dich', 'i love you'],
]

encoder_inputs, decoder_inputs, decoder_outputs, source_vocab_size, target_vocab_size, enc_pad, dec_pad  = tokenizer(sentences)
dataset = Seq2SeqDataset(encoder_inputs, decoder_inputs, decoder_outputs)
data_loader = Data.DataLoader(dataset, batch_size=2, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Transformer().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(1000):
  '''
  encoder_input: [batch, source_len]
  decoder_input: [batch, target_len]
  decoder_ouput: [batch, target_len]
  '''
  for encoder_input, decoder_input, decoder_output in data_loader:
    encoder_input = encoder_input.to(device)
    decoder_input = decoder_input.to(device)
    decoder_output = decoder_output.to(device)
    # [batch * target_len, target_vocab_size]
    output, encoder_attns, decoder_attns, decoder_encoder_attns = model(encoder_input, decoder_input)
    loss = criterion(output, decoder_output.view(-1))
    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model, "/homes/bohaochu/project/transformer-trm-pt/model/model.pth")
