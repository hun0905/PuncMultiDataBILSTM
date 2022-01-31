from nltk.util import pr
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import torch.nn.functional as F
class bi_GRU(nn.Module):
    def __init__(self,input_size,embedding_dim,hidden_size,num_layers,output_size,pretrained = True):#4979 , 256 , 512 , 2 , 4  batchc size 128
        super(bi_GRU,self).__init__() #繼承父類nn.Module 的特性
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.pretrained = pretrained
        if pretrained:
            #載入pretrained的embedding weight
            self.embedding = nn.Embedding(input_size,embedding_dim)
            weight = torch.load('/home/yunghuan/Desktop/PuncTBRNN/ChineseFastText.pth')
            self.embedding.load_state_dict({'weight':weight})
            self.embedding.weight.requires_grad = True
        else:
            self.embedding = nn.Embedding(input_size,embedding_dim)
        self.GRU = nn.GRU(self.embedding_dim,self.hidden_size,self.num_layers,bidirectional = True,batch_first = True)
        self.linaer = nn.Linear(hidden_size*2,self.output_size) #hidden_size*2 因為雙向 , punc_num 是標點的種類加1 ,因為要含' '也就是無標點
    
    def forward(self,input,input_length):
        #unk = input.cpu().numpy()
        input = self.embedding(input) #input : (batch_size , seq_len , embedding_dim)
        #print(np.sum(unk==1)/np.size(unk))
        input_length = input_length.view(-1) 
        pack_input = pack_padded_sequence(input,input_length.cpu(),batch_first=True) 
        pack_out,(h_n) = self.GRU(pack_input) #pack_out : (seq_len , batch_size , hidden_size*2)
        length = input.size(1)
        output,_ = pad_packed_sequence(pack_out,batch_first=True,total_length=length) #output : (batch_size , seq_len , hidden_size*2 )
        output = self.linaer(output) #output : (batch_size , seq_len , output_size)
        return output
    @classmethod
    def load_model(cls, path):
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)########
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls(package['input_size'], package['embedding_dim'],
                    package['hidden_size'], package['num_layers'],
                    package['output_size'], package['pretrained'])
        model.load_state_dict(package['state_dict'])
        return model
        
    def serialize(self,model, optimizer,scheduler, epoch,train_loss,val_loss):
        package = {
            # hyper-parameter
            'input_size': model.input_size,
            'embedding_dim': model.embedding_dim,
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers,
            'output_size': model.output_size,
            'pretrained':model.pretrained,
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss':val_loss
        }
        return package