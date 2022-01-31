import math
from nltk.util import pr
from numpy.core.fromnumeric import argmax
import torch
import random
from torch import nn
from torch._C import device
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules import dropout
import numpy as np
from torch.nn.modules.sparse import Embedding
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    @staticmethod
    def weight_init(m):
        #model的weight和bias初始化，weight 用 Glorot initialization  bias則初使為0
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m,nn.GRU) or isinstance(m,nn.LSTM):
            for name, param in m.named_parameters():
	            if name.startswith("weight"):
		            nn.init.xavier_uniform_(param)
	            else:
	            	nn.init.zeros_(param)   
    def __init__(self,input_size,embedding_size,hidden_size,output_size,num_layers,p,pre_trained):
        super(Encoder,self).__init__()
        self.pre_trained = pre_trained
        self.p = p
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if pre_trained:#使否用word pretrained vector
            self.embedding = nn.Embedding(input_size,embedding_size)
            #載入pretrain embedding ,這裡採用的是以FastText訓練wiki corpus所得到的pretrained embedding
            weight = torch.load('/home/yunghuan/Desktop/PuncTBRNN/ChineseFastText.pth')
            self.embedding.load_state_dict({'weight':weight})
            #設定載入的weight可以進行梯度更新
            self.embedding.weight.requires_grad = True
        else:
            self.embedding = nn.Embedding(input_size,embedding_size)
        self.rnn = nn.GRU(embedding_size,hidden_size,num_layers,bidirectional = True)
        self.fc_hidden = nn.Linear(self.hidden_size*2,self.hidden_size*2) #because it is bidirection, so *2.
        self.output_size = output_size
        #self.predict = nn.Linear(self.embedding_size*2,self.output_size)
        self.apply(self.weight_init)
    def forward(self,x,length):
        #print(x.size())
        
        length = length.view(-1)#句子長度

        embedding = self.dropout( self.embedding(x) ) #單字的idx經過embedding後輸出word vector然後在經過dropout,但此處dropout預設是0,所以無效
        seq_len = embedding.size()[1]
        batch_size = embedding.size(0) 
        #print(torch.flip(embedding,[0]))
        #pack_padded_sequence() 是用來壓縮序列的，而 pad_packed_sequence() 則是用來展開序列成原本形狀的
        #pack_padded_sequence : 將一個batch中不同長度的padded的句子包裝在一起，這裡會輸出維度（seqlen(max) , batch_size , *）
        pack_input= pack_padded_sequence(embedding,length.cpu(),batch_first=True)
        
       
        #將packed的句子放入rnn，h_t是每個time step的輸出在這裡也就是我們句子中每個單字的最後一層輸出,維度是（seq_len,batch_size,hidden_size*2）
        # *2是因為是bidirectional所以會把正向和逆向的最後一層的hidden_size拼在一起
        #print(hidden_f.size())
        length = x.size(1)
        
        output,_ = self.rnn(pack_input)
        
            
        hidden,_= pad_packed_sequence(output,total_length=length,batch_first=True)
       
        #print(hidden.size())
        hidden = self.fc_hidden(hidden)
        
        return hidden
    def serialize(self,model, optimizer, epoch,train_loss,test_loss):
        package = {
            # hyper-parameter
            'input_size':model.input_size,
            'embedding_size':model.embedding_size,
            'encoder_hidden_size':model.hidden_size,
            'encoder_output_size':model.output_size,
            'encoder_num_layers':model.num_layers,
            'encoder_dropout':model.p,
            'pre_trained':model.pre_trained,
            # state
            'encoder_state':model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss,
            'test_loss':test_loss
        }
        return package
    @classmethod
    def load_model(cls, path): #使用load_model可以載入已經訓練好的model
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)########
        model = cls.load_encoder_from_package(package)
        return model

    @classmethod
    def load_encoder_from_package(cls, package):
        encoder = cls(package['input_size'], package['embedding_size'],package['encoder_hidden_size'],package['encoder_output_size'],
                    package['encoder_num_layers'],package['encoder_dropout'],package['pre_trained'] )
        encoder.load_state_dict(package['encoder_state'])
        return encoder

class Seq2Seq(nn.Module):
    def __init__(self,encoder,hidden_size,num_layers,heads):
        super(Seq2Seq,self).__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.heads = heads
        self.num_layers = num_layers
        self.predict = nn.Linear(hidden_size*2,4)
        self.rnn = nn.GRU(self.hidden_size*2,self.hidden_size*2,batch_first = True)
        self.multi_head_attns = nn.MultiheadAttention(self.hidden_size*2,self.heads,batch_first=True)#,dropout=0.2)
        nn.init.xavier_uniform_(self.predict.weight)
        nn.init.zeros_(self.predict.bias)
    def forward(self,source,length):
        #my source and target : batch_size , seq_len 
        source = source
    
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        context = self.encoder(source,length) # context : (num_layers,batch_size,seq_len,hidden_size*2)
         #= context.permute(0,2,1,3) # context : (num_layers,seq_len,batch_size,hidden_size*2)
        attn_output = torch.zeros_like(context)
        rnn_output,_ = self.rnn(context) # output : (seq_len , batch_size , hidden_size)
        
        #attn_output,_ = self.multi_head_attns(rnn_output,context,context)
        attn_output,_ = self.multi_head_attns(rnn_output,context,context)
        #print(attn_output.size())
        outputs = self.predict(attn_output)
        #print(outputs)
        #outputs = outputs.transpose(0,1)
        return outputs
    @classmethod
    def load_model(cls, path): #載入model
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)########
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):

        model = cls(package['encoder'],package['hidden_size'],package['num_layers'],package['heads'])
        
        model.encoder = Encoder.load_encoder_from_package(package)
        model.load_state_dict(package['state_dict'])
        
        return model
    
    def serialize(self,model, optimizer,scheduler, epoch,train_loss,val_loss): #儲存model所有需要的參數和state
        package = {
            # hyper-parameter
            'encoder': model.encoder,
            'input_size':model.encoder.input_size,
            'embedding_size':model.encoder.embedding_size,
            'encoder_hidden_size':model.encoder.hidden_size,
            'encoder_output_size':model.encoder.output_size,
            'pre_trained':model.encoder.pre_trained,
            'encoder_num_layers':model.encoder.num_layers,
            'encoder_dropout':model.encoder.p,
            'pre_trained':model.encoder.pre_trained,
            'hidden_size':model.hidden_size,
            'num_layers':model.num_layers,
            'heads':model.heads,
            # state
            'encoder_state':model.encoder.state_dict(),
            'state_dict': model.state_dict(),
            'scheduler':scheduler.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss':val_loss
        }
        return package