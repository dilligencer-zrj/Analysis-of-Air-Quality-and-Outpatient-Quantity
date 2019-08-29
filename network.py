import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class LSTM(nn.Module):

    def __init__(self,INPUT_SIZE,HIDDEN_SIZE):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,     # rnn hidden unit
            num_layers=1,       # number of rnn layer

        )
        self.out = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        x, _ = self.rnn(x)
        t, b, f = x.shape
        x = x.view ( t * b, f )
        x = self.out ( x )
        x = x.view ( t, b, -1 )
        return x

class RNN(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=1
        )
        self.out = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        x, _ = self.rnn(x)
        t, b, f = x.shape
        x = x.view(t * b, f)
        x = self.out(x)
        x = x.view(t, b, -1)
        return x

class GRU(nn.Module):
    def __init__(self, args):
        super(GRU, self).__init__()
        self.window = args['window']
        self.variables = args['input_size']
        self.hw=args['highway_window']
        self.activate1=F.relu
        self.hidR=args['hidden_size']
        self.rnn1=nn.GRU(self.variables,self.hidR,num_layers=args['rnn_layers'])
        #self.linear1 = nn.Linear(self.hidR, self.variables)
        self.linear1 = nn.Linear(self.hidR, 1)
        # self.linear1=nn.Linear(1280,100)
        # self.out=nn.Linear(100,self.variables)
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)

        self.dropout = nn.Dropout(p=args['dropout'])
        self.output = None
        if (args['output_fun'] == 'sigmoid'):
            self.output = F.sigmoid
        if (args['output_fun'] == 'tanh'):
            self.output = F.tanh

    def forward(self, x):
        r= x.permute(1,0,2).contiguous()
        _,r=self.rnn1(r)
        r=self.dropout(torch.squeeze(r[-1:,:,:], 0))
        out = self.linear1(r)


        if (self.hw > 0):

            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.variables)
            out = out + z
        if self.output is not None:
            out=self.output(out)
        return out

class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.window = args['window']
        #self.variables = data.m
        self.variables = args['input_size']
        self.hw=args['highway_window']
        self.conv1 = nn.Conv1d(self.variables, 32, kernel_size=1)
        self.activate1=F.relu
        self.conv2=nn.Conv1d(32,32,kernel_size=1)
        self.maxpool1=nn.MaxPool1d(kernel_size=2)
        self.conv3=nn.Conv1d(32,16,kernel_size=1)
        self.maxpool2=nn.MaxPool1d(kernel_size=2)
        self.linear1=nn.Linear(1280,100)
        self.out=nn.Linear(100,self.variables)
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)

        self.dropout = nn.Dropout(p=args['dropout'])
        self.output = None
        if (args['output_fun'] == 'sigmoid'):
            self.output = F.sigmoid
        if (args['output_fun'] == 'tanh'):
            self.output = F.tanh

    def forward(self, x):
        c = x.permute(0,2,1).contiguous()
        c=self.conv1(c)
        c=self.activate1(c)
        c=self.conv2(c)
        c=self.activate1(c)
        c=self.maxpool1(c)
        c=self.conv3(c)
        c=self.activate1(c)
        c=c.view(c.size(0),c.size(1)*c.size(2))
        c=self.dropout(c)
        c=self.linear1(c)
        c=self.dropout(c)
        out=self.out(c)

        if (self.hw > 0):

            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.variables)
            out = out + z
        if self.output is not None:
            out=self.output(out)
        return out


class ScaledDotProductAttention(nn.Module):

    # Scaled Dot-Product Attention

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.2))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class LSTNet(nn.Module):
    def __init__(self, args, data):
        super(LSTNet, self).__init__()
        self.P = args.window
        self.m = data.m
        self.hidR = args.hidRNN
        self.hidC = args.hidCNN
        self.hidS = args.hidSkip
        self.Ck = args.CNN_kernel
        self.skip = args.skip

        self.hw = args.highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=args.dropout)
        if (self.skip > 0):
            self.pt = (self.P - self.Ck) / self.skip
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m)
        else:
            self.linear1 = nn.Linear(self.hidR, self.m)
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = F.tanh

    def forward(self, x):
        batch_size = x.size(0)

        # CNN
        c = x.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)

        r = self.dropout(torch.squeeze(r, 0))



        # skip-rnn

        if (self.skip > 0):
            self.pt=int(self.pt)
            s = c[:, :, int(-self.pt * self.skip):].contiguous()

            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)

        res = self.linear1(r)

        # highway
        if (self.hw > 0):

            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.m)
            res = res + z

        if (self.output):
            res = self.output(res)
        return res

class MHA(nn.Module):
    def __init__(self, args):
        super(MHA, self).__init__()
        self.window = args['window']
        self.variables = args['input_size']
        #self.hidC = args.hidCNN
        self.hidR = args['hidden_size']
        self.hw=args['highway_window']

        self.d_v=args['d_v']
        self.d_k=args['d_k']
        self.Ck = args['CNN_kernel']
        self.GRU = nn.GRU(self.variables, self.hidR, num_layers=args['rnn_layers'])
        # self.Conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.variables))

        self.slf_attn = MultiHeadAttention(args['n_head'], self.variables, self.d_k,self.d_v , dropout=args['dropout'])

        self.dropout = nn.Dropout(p=args['dropout'])
        #self.linear_out=nn.Linear(self.hidR,self.variables)
        self.linear_out=nn.Linear(self.hidR,1)

        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)
        self.output = None
        if (args['output_fun'] == 'sigmoid'):
            self.output = F.sigmoid
        if (args['output_fun'] == 'tanh'):
            self.output = F.tanh


    def forward(self, x):
        # r = x.permute(1, 0, 2).contiguous()
        # out, _ = self.GRU1(r)
        # c = out.permute(1, 0, 2).contiguous()
        # c = x.view(-1,1, self.window, self.variables)
        # c = F.relu(self.Conv1(c))
        # c = self.dropout(c)
        # c = torch.squeeze(c, 3)
        # c=c.permute(0,2,1).contiguous()

        attn_output, slf_attn=self.slf_attn(x,x,x,mask=None)

        r=attn_output.permute(1,0,2).contiguous()
        _,r=self.GRU(r)
        r = self.dropout(torch.squeeze(r[-1:, :, :], 0))
        out = self.linear_out(r)

        if (self.hw > 0):

            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.variables)
            out = out + z
        if self.output is not None:
            out=self.output(out)
        return out