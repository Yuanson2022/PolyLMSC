import torch
import torch.nn as nn
from collections import OrderedDict
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


class Stress_LMSC(nn.Module):
    def __init__(self, args):
        super(Stress_LMSC, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        self.noise = args.add_noise
        self.lmsc = LMSC_cell(args.input_dim, args.output_dim, units = args.hidden_dim,\
                               width = args.width, depth = args.depth)
        self.lamda = 2e-4
        # self.tex_fc = nn.Linear([])


    def forward(self, x, init_F):
        h0 = torch.zeros(x.size(0), self.hidden_dim).requires_grad_().to(device)
        hidden_out = torch.zeros(x.size(0), x.size(1), self.hidden_dim).requires_grad_().to(device)
        alpha_out = torch.zeros(x.size(0), x.size(1), self.hidden_dim).requires_grad_().to(device)
        out_state_list =  torch.zeros(x.size(0), x.size(1), self.output_dim).requires_grad_().to(device)
        
        # use mf to initialze h0
        h0 = init_F[:,2:2 + self.hidden_dim].to(device).type(torch.cuda.FloatTensor)
        h_last = h0.clone()
        for i in range(x.size(1)):
            x_input = x[:,i,:].unsqueeze(1).clone()
    
            lmsc_out, h_last, alpha = self.lmsc(x_input, h_last)

            if self.noise == True:
                h_last = h_last + (torch.rand_like(h_last)-0.5)*self.lamda*2
                
            out_state_list[:,i,:] = lmsc_out.squeeze(1)
            hidden_out[:,i,:] = h_last.squeeze(1)
            alpha_out[:,i,:] = alpha.squeeze(1)

        return out_state_list, alpha_out


class Texture_LMSC(nn.Module):
    def __init__(self, args):
        super(Texture_LMSC, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        self.grain_num = args.grain_num

        self.lmsc = LMSC_cell2(args.input_dim, args.output_dim, units = args.hidden_dim,\
                               width = args.width, depth = args.depth)
        self.nn_out = DNN(args.layers, nn.Tanh)
        self.noise = args.add_noise
        self.lamda = 2e-4

    def forward(self, x, init_ori):
        h0 = torch.zeros(x.size(0), self.grain_num, self.hidden_dim).requires_grad_().to(device)
        out_state_list =  torch.zeros(x.size(0), self.grain_num, x.size(1),\
                                       self.output_dim).requires_grad_().to(device)
        
        h0[:,:,0:3] = init_ori.to(device)
        
        # size = sys.getsizeof(out_state_list)
        
        h_last = h0
        for i in range(x.size(1)):
            x_input = x[:,i,:].unsqueeze(1).clone()    
            h_last = self.lmsc(x_input, h_last)

            if self.noise == True:
                h_last = h_last + (torch.rand_like(h_last)-0.5)*self.lamda*2
                          
            out_state_list[:,:,i,:] = self.nn_out(h_last)
        return out_state_list


class LMSC_cell(nn.Module):
    def __init__(self, input_dim, output_dim, width=32, depth=3, units=16, output_depth=0):
        super(LMSC_cell, self).__init__()

        self.output_dim = output_dim
        self.depth = depth
        self.output_depth = output_depth
        self.units = units

        start_dim = units + input_dim

        self.qb =  Quadratic_block(start_dim, width, depth)

        if self.depth>0:
            inside_dim = width
        else:
            inside_dim = start_dim

        self.fc_alpha = nn.Linear(inside_dim,units)
        self.fc_beta = DNN([inside_dim,units],nn.Tanh())

        self.fc_out = nn.Linear(units, output_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

        self.fc_out.bias.requires_grad = False
        self.fc_out.bias.data.zero_()

    def forward(self, x, hidden_state):
        h_t = hidden_state.unsqueeze(1)
        
        strain_norm = torch.norm(x[:,:,:],dim=2)
        x_input = x.clone()
        x_input[:,:,:] = (x_input[:,:,:].squeeze(1)/(strain_norm + 1e-8)).unsqueeze(1)

        cat_input = torch.cat([x_input,h_t], dim = 2)

        cat_input = torch.tanh(self.qb(cat_input))

        alpha = torch.exp(self.fc_alpha(cat_input))
        beta  = torch.tanh(self.fc_beta(cat_input))

        exp_f =  torch.exp(- alpha * strain_norm.unsqueeze(2).repeat(1,1,self.units)) 
        h = exp_f * (h_t  - beta) + beta
        x_out = self.fc_out(h)

        return x_out, h.squeeze(1), alpha.squeeze(1)
        
        
class LMSC_cell2(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, units:int,
                    width=128, depth=4,):
            super(LMSC_cell2, self).__init__()

            self.in_channels = in_channels
            self.out_channels = out_channels
            self.units = units
            self.depth = depth

            start_dim = units + in_channels
            inside_dim = start_dim
            self.qb =  Quadratic_block(inside_dim, width, depth)

            if self.depth>0:
                inside_dim = width
            else:
                inside_dim = in_channels

            self.fc_alpha = nn.Linear(inside_dim,units)
            self.fc_beta = DNN([inside_dim,units],nn.Tanh())

            for m in self.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    m.bias.data.fill_(0)

    def forward(self, x_input, h_t):
        # X = [batch, seq, feature_dim]
        # H = [batch, node_number, feature_dim]
        strain_norm = torch.norm(x_input[:,:,:],dim=2)
        x_input[:,:,:] = (x_input[:,:,:].squeeze(1)/(strain_norm + 1e-15)).unsqueeze(1)

        x_input = torch.cat([x_input.repeat(1,h_t.shape[1],1),h_t], dim = 2)

        x_input = torch.tanh(self.qb(x_input))

        alpha = torch.exp(self.fc_alpha(x_input))

        # model previous to 1213 not used tanh for activation
        beta  = torch.tanh(self.fc_beta(x_input))

        # del x_input

        alpha =  torch.exp(- alpha * strain_norm.unsqueeze(2).repeat(1,1,self.units)) 
        h = alpha * (h_t  - beta) + beta

        return h.squeeze(1)


class Quadratic_block(nn.Module):
    def __init__(self, input_dim, width, depth):
        super(Quadratic_block, self,).__init__()
        self.modlist1 = nn.ModuleList()
        self.modlist2 = nn.ModuleList()
        self.depth = depth
        for i in range(depth):
            if i == 0:
                self.modlist1.append(torch.nn.Linear(input_dim, width))
                self.modlist2.append(torch.nn.Linear(input_dim, width))
            else:
                self.modlist1.append(torch.nn.Linear(width, width))
                self.modlist2.append(torch.nn.Linear(width, width))


    def forward(self, x):
        i = 0
        for m in self.modlist1:
            x1 = m(x)
            m2 = self.modlist2[i]
            x2 = m2(x)
            i += 1
            if i < self.depth:
                x1 = torch.tanh(x1)
                x2 = torch.tanh(x2)
            x = x1 * x2
        return x


class DNN(torch.nn.Module):
    def __init__(self, layers, activation):
        super(DNN, self).__init__()
        
        # parameters
        self.depth = len(layers) - 1
        
        # set up layer order dict
        self.activation = activation
        
        layer_list = list()
        for i in range(self.depth - 1): 
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))
            
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        
        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)
        
    def forward(self, x):
        out = self.layers(x)
        return out 