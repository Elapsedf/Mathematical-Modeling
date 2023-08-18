import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot
import pandas as pd
torch.manual_seed(0)
np.random.seed(0)

# This concept is also called teacher forceing. 
# The flag decides if the loss will be calculted over all 
# or just the predicted values.
calculate_loss_over_all_values = True  # BUG

# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number

#src = torch.rand((10, 32, 512)) # (S,N,E) 
#tgt = torch.rand((20, 32, 512)) # (T,N,E)
#out = transformer_model(src, tgt)
#
#print(out)

input_window = 100
output_window = 1
batch_size = 10 # batch size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

       

class TransAm(nn.Module):
    def __init__(self,feature_size=250,num_layers=1,dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)#, self.src_mask)
        output = self.decoder(output)
        # DenNet(out)
        # softmax()
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask



# if window is 100 and prediction step is 1
# in -> [0..99]
# target -> [1..100]
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = np.append(input_data[i:i+tw][:-output_window] , output_window * [0])
        train_label = input_data[i:i+tw]
        train_label = input_data[i+output_window:i+tw+output_window]
        inout_seq.append((train_seq ,train_label))
    return torch.FloatTensor(inout_seq)

def dataset():
    df = pd.read_csv(r'/home/huangjiehui/Project/Small_Project/exam/data2.csv',encoding='gb18030')
    data_numpy = np.array(df)
    df_temperature = df
    # 第一问(a)
    del df_temperature['Country']
    del df_temperature['Latitude']
    del df_temperature['Longitude']
    data_temperature = np.array(df_temperature)

    data_temperature_list = data_temperature.tolist()
    all_temp = set() # 所有的温度
    for i in range(0,len(data_temperature_list)):
        # data_temperature_list[i][3]
        all_temp.add(data_temperature_list[i][0])
    all_temp = list(all_temp)
    all_temp.sort()

    temp_dict=dict()
    for i in range(0,len(all_temp)):
        temp_dict[all_temp[i]] = []

    for i in range(0,len(data_temperature_list)):
        temp_dict[data_temperature_list[i][0]].append(data_temperature_list[i][1])

    # 对每个日期的温度求均值
    f = open(r'/home/huangjiehui/Project/Small_Project/exam/mean_tmp.txt','w')
    train_data = []
    k = 0 # 记录当前是第几个数值
    f.write(" ======== train ========"+'\n')
    for i in temp_dict:
        train_data.append(np.mean(temp_dict[i]))
        msg = f'{i}    mean_tmp:  {np.mean(temp_dict[i])}'
        if k == 3139:
            f.write(" ======== test ========"  + '\n')
        k+=1
        f.write(msg+'\n')
    
    return np.array(train_data)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()



def get_data():
    time        = np.arange(0, 400, 0.1)
    # amplitude   = np.sin(time) + np.sin(time*0.05) +np.sin(time*0.12) *np.random.normal(-0.2, 0.2, len(time))
    
    #from pandas import read_csv
    # series = pd.read_csv('test2.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
    
    # from sklearn.preprocessing import MinMaxScaler
    # scaler = MinMaxScaler(feature_range=(-1, 1)) 
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler() 
    global scaler
    train_data_org = dataset()
    # amplitude = scaler.fit_transform(train_data_org.to_numpy().reshape(-1, 1)).reshape(-1)
    amplitude = scaler.fit_transform(train_data_org.reshape(-1, 1)).reshape(-1)
    # amplitude = train_data_org
    
    # 
    sampels = 3039
    train_data = amplitude[:sampels]
    test_data = amplitude[sampels:]

    # convert our train data into a pytorch train tensor
    #train_tensor = torch.FloatTensor(train_data).view(-1)
    # todo: add comment.. 
    train_sequence = create_inout_sequences(train_data,input_window)
    train_sequence = train_sequence[:-output_window] #todo: fix hack?

    #test_data = torch.FloatTensor(test_data).view(-1) 
    test_data = create_inout_sequences(test_data,input_window)
    test_data = test_data[:-output_window] #todo: fix hack?

    return train_sequence.to(device),test_data.to(device)

def get_batch(source, i,batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i+seq_len]    
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window,1)) # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window,1))
    return input, target

cnt = 0
def train(train_data,cur_epoch):
    floss = open("loss.txt",'w')
    # cur_epoch当前的epoch -- 如果是最后一个epoch -- 输出预测结果
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    if(cur_epoch == 10):
        f = open(r'/home/huangjiehui/Project/Small_Project/exam/train_output.txt','w')
    print('',end='')
    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        global cnt
        cnt+=1
        data, targets = get_batch(train_data, i, batch_size)
        optimizer.zero_grad()
        output = model(data)
        if(cur_epoch == 10):
            for i in range(0,len(targets[0])):
                msg = f'targets: {targets[0][i].tolist()} output:  {output[0][i].tolist()}'
                f.write(msg + '\n')      
        
        if calculate_loss_over_all_values:
            loss = criterion(output, targets)
        else:
            loss = criterion(output[-output_window:], targets[-output_window:])
    
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(
                    cur_epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0], # BUG epoch改成cur_epoch
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            floss.write(f'{epoch},{cur_loss}\n')
            total_loss = 0
            start_time = time.time()

def plot_and_loss(eval_model, data_source,epoch):
    eval_model.eval() 
    total_loss = 0.
    cur_loss = 0.
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i,1)
            # look like the model returns static values for the output window
            output = eval_model(data)    
            if calculate_loss_over_all_values:                                
                cur_loss = criterion(output, target).item()
                total_loss += cur_loss
            else:
                cur_loss += criterion(output[-output_window:], target[-output_window:]).item()
                total_loss += cur_loss
            
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0) #todo: check this. -> looks good to me
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)
        print()
            
    #test_result = test_result.cpu().numpy()
    len(test_result)

    # pyplot.plot(test_result,color="red")
    # pyplot.plot(truth[:500],color="blue")
    # pyplot.plot(test_result-truth,color="green")
    # pyplot.grid(True, which='both')
    # pyplot.axhline(y=0, color='k')
    # pyplot.savefig('graph/transformer-epoch%d.png'%epoch)
    # pyplot.close()
    
    return total_loss / i


def predict_future(eval_model, data_source,steps):
    df_data_org = pd.read_csv()
    eval_model.eval() 
    total_loss = 0.
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)
    _ , data = get_batch(data_source, 0,1)
    with torch.no_grad():
        for i in range(0, steps,1):
            input = torch.clone(data[-input_window:])
            input[-output_window:] = 0     
            output = eval_model(data[-input_window:])                     
            data = torch.cat((data, output[-1:]))
    # 此时把结果记录一下就行了
    # from sklearn.preprocessing import StandardScaler
    # data = scaler.fit_transform(data.reshape(-1, 1).cpu().numpy().reshape(-1, 1)).reshape(-1)
    # data = data.reshape(-1,1)
    # data = data.reshape(-1,1).cpu()
    # data = data.reshape(steps+100).cpu()
    # scaler = StandardScaler().fit(data)
    # data = data.cpu().numpy().reshape(steps+100)
    # data = scaler.transform(data)
    # data = data.reshape(steps+100)
    
    global scaler
    data = scaler.inverse_transform(data.cpu().reshape(-1,1))
    f = open('/home/huangjiehui/Project/Small_Project/exam/predict_output.txt','w')
    for i in range(0,len(data)):
        # 把每一行都弄进去就行
        msg = f'{i},{data[i].tolist()}'
        f.write(msg+'\n')
    # data = data.cpu().view(-1)
    
        
# entweder ist hier ein fehler im loss oder in der train methode, aber die ergebnisse sind unterschiedlich 
# auch zu denen der predict_future
def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i,eval_batch_size)
            output = eval_model(data)            
            if calculate_loss_over_all_values:
                total_loss += len(data[0])* criterion(output, targets).cpu().item()
            else:                                
                total_loss += len(data[0])* criterion(output[-output_window:], targets[-output_window:]).cpu().item()            
    return total_loss / len(data_source)

train_data, val_data = get_data()
model = TransAm().to(device)
# from torchsummary import summary
# summary(model, train_data.shape, batch_size , device)
# summary(model, train_data.shape)


criterion = nn.MSELoss()

lr = 0.005  
#optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)

best_val_loss = float("inf")
epochs = 3 # The number of epochs
best_model = None
# BUG 记得处理best_model
# ================================================== train ==================================================
path = r'/home/huangjiehui/Project/Small_Project/exam/model.pth'
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data,epoch)
    print()
    if (epoch == 3):
        val_loss = evaluate(model, val_data)
        predict_future(model, val_data, 1500) # 设置预测步数
    else:
        val_loss = evaluate(model, val_data)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
    scheduler.step() 
    # torch.save(best_model,path)
    # arr = list(best_model.decoder.named_parameters())
    # print(arr.__len__())
    # array = arr[0].to_numpy()
    # import seaborn as sns
    # df = torch.tensor(array)
    # sns.pairplot()
    # print
print()