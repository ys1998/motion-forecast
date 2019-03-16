import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torchdiffeq import odeint

# class MnistModel(BaseModel):
#     def __init__(self, num_classes=10):
#         super(MnistModel, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, num_classes)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

class LatentODEFunc(nn.Module):
    def __init__(self, k, input_size, hidden_size):
        super(LatentODEFunc, self).__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ELU(inplace=True),
                nn.Linear(hidden_size, hidden_size),
                nn.ELU(inplace=True),
				nn.Linear(hidden_size, hidden_size),
                nn.ELU(inplace=True),
                nn.Linear(hidden_size, input_size)
            )
        for _ in range(k)])
        self.k = k
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nfe = 0
    
    def forward(self, t, x):
        assert x.size(1) == self.k * self.input_size, "Invalid input size"
        result = torch.Tensor([]).to(x.device)
        for i in range(self.k):
            out = self.layers[i](x[:,i*self.input_size:(i+1)*self.input_size])
            result = torch.cat([result, out], dim=1)
        self.nfe += 1
        return result

class EncoderRNN(nn.Module):
    def __init__(self, latent_size, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, latent_size * 2)

    def forward(self, x, h):
        combined = torch.cat([x, h], dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

class Decoder(nn.Module):
    def __init__(self, latent_size, output_size, hidden_size):
        super(Decoder, self).__init__()
		self.layers = nn.Sequential(
			nn.Linear(latent_size, hidden_size)
			nn.ELU(inplace=True)
			nn.Linear(hidden_size, hidden_size)
			nn.ELU(inplace=True)
			nn.Linear(hidden_size, hidden_size)
			nn.ELU(inplace=True)
			nn.Linear(hidden_size, output_size)
		)

    def forward(self, z):
        return self.layers(z)

class Model(BaseModel):
    def __init__(self, k, input_size, hidden_size, latent_size, scale):
        super(Model, self).__init__()
        # self.encoders = nn.ModuleList([
        #     EncoderRNN(latent_size, input_size, hidden_size)
        # for _ in range(k+1)])
		self.encoders = nn.ModuleList([
			nn.GRU(input_size, hidden_size, 4)
		for _ in range(k+1)])
		self.projections = nn.ModuleList([
			nn.Linear()
		])
		
		
        self.decoders = nn.ModuleList([
            Decoder(latent_size, 2*input_size, hidden_size)
        for _ in range(k+1)])
        self.func = LatentODEFunc(k+1, latent_size, hidden_size)
        self.k = k
        self.latent_size = latent_size
        self.scale = scale

    def forward(self, x, states):
        # x : (batch, time, (k+1)*input_size)
        assert x.size(2) % (self.k+1) == 0, "Invalid input size"
        splits = torch.chunk(x, self.k+1, dim=2)
        out, h = [None]*(self.k+1), [s.to(x.device) for s in states]
        for t in reversed(range(x.size(1))):
            for i in range(self.k+1):
                out[i], h[i] = self.encoders[i](splits[i][:,t,:], h[i])
        z0 = []
        mean = []
        logvar = []
        for i in range(self.k+1):
            qz0_mean, qz0_logvar = out[i][:, :self.latent_size], out[i][:, self.latent_size:]
            mean.append(qz0_mean)
            logvar.append(qz0_logvar)
            epsilon = torch.randn(qz0_mean.size()).to(x.device)
            z0.append(epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean)
        concat_z0 = torch.cat(z0, dim=1)
        pred_z = odeint(self.func, concat_z0, torch.arange(x.size(1)).float().to(x.device)/self.scale, atol=1e-7, rtol=1e-5)
        pred_z_splits = torch.chunk(pred_z, self.k+1, dim=2)
        pred_x_splits = [self.decoders[i](pred_z_splits[i]).permute(1,0,2) for i in range(self.k+1)] # convert to (batch, time, :)
        
        # compute mean and logvar of distribution of sequence, which is modelled by the random variable
        # equal to the sum of random variables modelling each motion band
        input_size = pred_x_splits[0].size(2) // 2
        pred_mean = sum([x[..., :input_size] for x in pred_x_splits])
        pred_logvar = torch.log(sum([torch.exp(x[..., input_size:]) for x in pred_x_splits]))
        output = {
            'z0_means' : mean,
            'z0_logvars': logvar,
            'pred_mean' : pred_mean,
            'pred_logvar' : pred_logvar
        }
        return output

    def init_hidden(self, batch_size):
        return [self.encoders[i].init_hidden(batch_size) for i in range(self.k+1)]
        