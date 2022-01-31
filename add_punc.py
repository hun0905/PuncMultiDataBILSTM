import torch
from torch._C import _llvm_enabled
from torch.utils.data import dataset
from model2 import bi_GRU
from model1 import Encoder,Seq2Seq
from dataset import NoPuncDataset
from torch.utils.data import Dataset, DataLoader
vocab_path = '/home/yunghuan/NLP_Dataset/Chinese/data_Ch/vocab.txt'
with open(vocab_path, encoding='utf-8',errors='ignore') as file:
    vocab = { word.strip(): i + 4 for i, word in enumerate(file) } 
def punc_vocab(punc_vocab_path):
    with open(punc_vocab_path, encoding='utf-8') as file:
        punc_vocab = { i + 1 : word.strip()for i, word in enumerate(file) } 
    punc_vocab[0] = " " #沒有標點
    return punc_vocab

def add_punc(seq, predict, class2punc):    
    txt_with_punc = ""
    #print(len(seq.split()))
    #print(len(predict))
    #print(seq.split())
    #print(predict)
    for i, word in enumerate(seq.split()):
        punc = class2punc[predict[i]][0]
        #txt_with_punc += word + " " if punc == " " else punc + " " + word + " "
        txt_with_punc += word + " " if punc == " " else word+' '+punc
    #punc = class2punc[predict[i + 1]][0]
    #txt_with_punc += punc
    #print(txt_with_punc)
    return txt_with_punc

def add_punctuation(use_cuda = True):
    attn=False
    if attn:
        model = Seq2Seq.load_model('/home/yunghuan/Desktop/Punc_new/model/model_GRU.pth.tar7')
    else:
        model = bi_GRU.load_model('/home/yunghuan/Desktop/Punc_new/model/model_GRU_wiki.pth.tar5')
    model.eval()
    if use_cuda:
        model = model.cuda()
    demo_result = open('demo_result.txt','w')
    dataset = NoPuncDataset('/home/yunghuan/Desktop/Punc_new/demo_no_punc.csv',\
                            '/home/yunghuan/NLP_Dataset/Chinese/data_Ch/vocab.txt',\
                            '/home/yunghuan/NLP_Dataset/Chinese/data_Ch/punc.txt')
    data_loader = DataLoader(dataset, batch_size=1)
    class2punc = punc_vocab('/home/yunghuan/NLP_Dataset/Chinese/data_Ch/punc.txt')
    
    with torch.no_grad():
        for i,(seq_id,seq) in enumerate(data_loader):
            lengths = torch.LongTensor([ seq_id.size()[1] ])
            if use_cuda:
                seq_id = seq_id.cuda()
            output = model(seq_id,lengths)
            #print(output.size())
            output = torch.argmax( output.squeeze(0) , 1)
            output = output.data.cpu().numpy().tolist()
            out_text = add_punc(seq[0],output[2:],class2punc)
            demo_result.write(out_text)
            demo_result.write('\n')
        
def main():
    add_punctuation()
if __name__ == '__main__':
    main()
