import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 44
torch.manual_seed(seed)

seed = 44
beta   = 1e-3
z_dim  = 256
epochs = 30
learning_rate = 1e-4
decay_rate = 0.97

def renyi_cross_entropy(prop, label, alpha=2):
    
    assert alpha > 1, "Require alpha > 1"    
    assert prop.min()>=0 or prop.max()<=1, \
        f"Probability out of range [0,1] but found probaility: {prop}"
    
    label = F.one_hot(label, num_classes=-1) 
    #print(f"prop shape {prop.shape} label shape {label.shape}")
    output = torch.sum(prop.pow(alpha-1)*label, dim=1)
    
    if output.isnan().any():
            print(f"renyi_cross entropy is nan {output} ")
                    
    output= output.log().mean()/(1-alpha)
    return output

def renyi_divergence(mu, log_sigma, alpha=2):
    #https://mast.queensu.ca/~communications/Papers/GiAlLi13.pdf

    sigma = log_sigma.exp()
    var_star =  sigma**2*alpha+1-alpha
        
    if sigma.min() <=0:
        print('sigma',sigma,sigma.min())
        
    #if var_star.min()<=0:
        #print('sigma_star', var_star, var_star.min())
   
    output = alpha/(alpha-1)*log_sigma \
             +0.5*alpha*mu**2/var_star #dropping alpha
    if output.isnan().any():
            print(f"renyi_divergence is nan {output} ")
                         
        
    return output.mean()
    
    
def renyi_loss(y_pred,y,mu,std, alpha = 2, beta=beta):
    if std.min()<0:
        print('std',std,std.min())
    return renyi_cross_entropy(y_pred,y,alpha)\
        +beta*renyi_divergence(mu,std,alpha)


def shannon_loss(y_pred,y,mu,std,beta=beta):
    """    
    y_pred : [batch_size,10]
    y : [batch_size,10]   wrong 
    mu : [batch_size,z_dim]  
    std: [batch_size,z_dim] 
    """   
    CE = F.cross_entropy(y_pred, y, reduction='sum')
    
    KL = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2*std.log() - 1)
    
    return (beta*KL + CE) / y.size(0)
    
# Initialize Deep VIB
class DeepVIB(nn.Module):
    def __init__(self, input_dim, output_dim, K):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.K = K       
        #encoder
        self.encoder = nn.Sequential(nn.Linear(input_dim, 1024),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(1024, 1024),
                                     nn.ReLU(inplace=True),
                                     )  
        self.encoder2 = nn.Sequential(nn.Linear(input_dim+1, 1024),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(1024, 1024),
                                     nn.ReLU(inplace=True),
                                     )   
        self.mu = nn.Linear(1024, self.K)  
        self.std = nn.Linear(1024, self.K)     
        #decoder 
        self.decoder = nn.Linear(self.K, output_dim)
    
    def encode(self, x):
        #assert x.shape == (,784), "check dimension"
        x = self.encoder(x)
        return self.mu(x), F.softplus(self.std(x), beta=1)
    
    def decode(self, z):
        return self.decoder(z)
    
    def reparameterize(self, mu, std):
        eps = torch.rand_like(std)
        return mu + std*eps
    
    def forward(self, x):
        mu, std = self.encode(x)
        z = self.reparameterize(mu, std)
        y_pred = nn.Softmax(dim=1)(self.decode(z))
        if y_pred.min() < 0:
            print('y_pred',y_pred,y_pred.min())
        return y_pred, mu, std
       
    def train_(self, dataloader):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        self.train()
        batch_loss = 0
        batch_accuracy = 0
        for _, (X,y) in enumerate(dataloader):
            X = X.view(X.size(0),-1).to(device)        
            y = y.long().to(device)

            # Zero accumulated gradients
            self.zero_grad()
            # forward pass through Deep VIB
            y_pred, mu, std = self.forward(X)
            loss = renyi_loss(y_pred, y, mu, std, alpha =2)
            loss.backward()
            optimizer.step()    
            batch_loss += loss.item()*X.size(0) 
            y_predication = torch.argmax(y_pred,dim=1)
            batch_accuracy += int(torch.sum(y == y_predication))  
            
            
            
            if loss.isnan().any():
                print(f"loss nan at batch idx {_} ")
                raise SystemExit    
        return batch_loss, batch_accuracy

    def validate_(self, dataloader):
        self.eval()
        batch_loss = 0
        batch_accuracy = 0
        for _, (X,y) in enumerate(dataloader):
            X = X.view(X.size(0),-1).to(device)        
            y = y.long().to(device)

            # Zero accumulated gradients
            self.zero_grad()
            # forward pass through Deep VIB
            y_pred, mu, std = self(X)
            loss = renyi_loss(y_pred, y, mu, std, alpha =2)
            #loss.backward()
            #optimizer.step()    
            batch_loss += loss.item()*X.size(0) 
            
            y_predication = torch.argmax(y_pred,dim=1)
            batch_accuracy += int(torch.sum(y == y_predication))  
            
            if loss.isnan().any():
                print(f"loss nan at batch idx {_} ")
                raise SystemExit    
        return batch_loss, batch_accuracy  

    def fit(self, train_dataloader, val_dataloader,epoch=10): 
        
        from collections import defaultdict
        measures = defaultdict(list)  
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)     
        for epoch in range(epochs):
            epoch_start_time = time.time()  
            
            # exponential decay of learning rate every 2 epochs
            if epoch % 2 == 0 and epoch > 0:
                #scheduler.step()    
                pass
            

            train_loss,train_acc,s_acc = self.train_(train_dataloader)
            val_loss,val_acc= self.validate_(val_dataloader)
            #print(len(train_dataloader))
            # Save losses per epoch
            measures['train_loss'].append(train_loss / len(train_dataloader.dataset))        
            # Save accuracy per epoch
            measures['train_acc'].append(train_acc / len(train_dataloader.dataset)) 
            measures['val_loss'].append(val_loss / len(val_dataloader.dataset))        
            # Save accuracy per epoch
            measures['val_acc'].append(val_acc / len(val_dataloader.dataset))               
            measures['s_acc'].append(s_acc / len(train_dataloader.dataset))
            print("Epoch: {}/{}...".format(epoch+1, epochs),
                "Train Loss: {:.4f}...".format(measures['train_loss'][-1]),
                "Train Accuracy: {:.4f}...".format(measures['train_acc'][-1]),
                "val Loss: {:.4f}...".format(measures['val_loss'][-1]),
                "val Accuracy: {:.4f}...".format(measures['val_acc'][-1]),
                "s Accuracy: {:.4f}...".format(measures['s_acc'][-1]),
                
                
                "Time Taken: {:,.4f} seconds".format(time.time()-epoch_start_time))
            
            
        #print("Total Time Taken: {:,.4f} seconds".format(time.time()-start_time))
def fair_loss(y_pred,y,mu,std,mu2,std2, alpha = 2, beta=beta,beta2 =1e-3):
    if std.min()<0:
        print('std',std,std.min())
    return renyi_cross_entropy(y_pred,y,alpha)\
        +beta*renyi_divergence(mu,std,alpha)\
            +beta2*renyi_divergence(mu2,std2,alpha)

class FairVIB(DeepVIB):  
    def encode2(self, x):
        x = self.encoder2(x)
        return self.mu(x), F.softplus(self.std(x), beta=1)
        
    def forward(self, x, s):
        mu, std = self.encode(x)
        mu2,std2 =self.encode2(torch.cat((x,s),dim = 1))
        z = self.reparameterize(mu, std)
        z2 = self.reparameterize(mu2, std2)
        y_pred = nn.Softmax(dim=1)(self.decode(z))
        s_pred = nn.Softmax(dim=1)(self.decode(z2))
        #s_pred = nn.Softmax(dim=1)(self.decode(z))
        if y_pred.min() < 0:
            print('y_pred',y_pred,y_pred.min())
        return y_pred, s_pred, mu, std, mu2,std2
    def train_(self, dataloader):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        self.train()
        batch_loss = 0
        batch_accuracy = 0
        s_accuracy = 0
        for _, (X,y) in enumerate(dataloader):
            
            X = X.view(X.size(0),-1).to(device)        
            y = y.view(y.size(0),-1)
            s_label = y[:,1].to(device)
            s = y[:,1].view(X.size(0),1).to(device)
            y = y[:,0].to(device)
            
            #print(f"idx {_}...xshape{X.shape}...s_shape,{s.shape}")
           # print(f'concata {torch.cat((X,s),dim = 1)}')
            # Zero accumulated gradients
            self.zero_grad()
            # forward pass through Deep VIB
            y_pred, s_pred, mu, std, mu2, std2 = self.forward(X, s)
            loss = fair_loss(y_pred, y, mu, std, mu2,std2, alpha =2) 
            loss.backward()
            optimizer.step()    
            batch_loss += loss.item()*X.size(0) 
            
            y_predication = torch.argmax(y_pred,dim=1)
            s_predication = torch.argmax(s_pred,dim=1)
            batch_accuracy += int(torch.sum(y == y_predication))  
            s_accuracy += int(torch.sum(s_label == s_predication))
            if loss.isnan().any():
                print(f"loss nan at batch idx {_} ")
                raise SystemExit    
        return batch_loss, batch_accuracy, s_accuracy
    def validate_(self, dataloader):
        self.eval()
        batch_loss = 0
        batch_accuracy = 0
        for _, (X,y) in enumerate(dataloader):
            X = X.view(X.size(0),-1).to(device)        
            y = y.view(y.size(0),-1)
           
            s = y[:,1].view(X.size(0),1).to(device)
            y = y[:,0].to(device)

            # Zero accumulated gradients
            self.zero_grad()
            # forward pass through Deep VIB
            y_pred, s_pred, mu, std, mu2, std2 = self.forward(X, s)
            loss = fair_loss(y_pred, y, mu, std, mu2,std2, alpha =2)
            #loss.backward()
            #optimizer.step()    
            batch_loss += loss.item()*X.size(0) 
            
            y_predication = torch.argmax(y_pred,dim=1)
            batch_accuracy += int(torch.sum(y == y_predication))  
            
            if loss.isnan().any():
                print(f"loss nan at batch idx {_} ")
                raise SystemExit    
        return batch_loss, batch_accuracy  
    