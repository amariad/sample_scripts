import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pylab as plt
import numpy as np
import time
from numpy import random
import sys, os
from cube_augmentation import * 


### TRAINING OR TESTING

training = True
testing  = False

########################################
########################################
### ARCHITECTURE

neurons = 100

#3-D constructor
class CNN1(nn.Module):
    
    def __init__(self, out_1=1, out_2=1, out_3=2, out_4=4, out_5=4, out_6=8, params=5):
        super(CNN1, self).__init__()

        self.cnn1 = nn.Conv3d(in_channels=1,     out_channels=out_1, kernel_size=5, 
                              stride=1, padding=0) 
        self.cnn2 = nn.Conv3d(in_channels=out_1, out_channels=out_2, kernel_size=5, 
                              stride=3, padding=0)
        self.cnn3 = nn.Conv3d(in_channels=out_2, out_channels=out_3, kernel_size=5, 
                              stride=3, padding=0)
        self.cnn4 = nn.Conv3d(in_channels=out_3, out_channels=out_4, kernel_size=3, 
                              stride=3, padding=0)


        self.BN1 = nn.BatchNorm3d(num_features=out_1)
        self.BN2 = nn.BatchNorm3d(num_features=out_2)
        self.BN3 = nn.BatchNorm3d(num_features=out_3)
        self.BN4 = nn.BatchNorm3d(num_features=out_4)
        
        self.AvgPool1 = nn.AvgPool3d(kernel_size=2)

        self.fc1 = nn.Linear((out_4 * 4 * 4 * 4) +len(kmask), neurons) 
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, neurons)
        self.fc4 = nn.Linear(neurons, params)
	
        self.dropout   = nn.Dropout(p=0.5)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU()
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
       
    
    # Prediction
    def forward(self, x, pk):
        out = self.cnn1(periodic_padding(x,2)) #first convolutional layer
        out = self.BN1(out)
        out = self.LeakyReLU(out)
        out = self.AvgPool1(out)
        out = self.cnn2(periodic_padding(out,2)) #second convolutional layer
        out = self.BN2(out)
        out = self.LeakyReLU(out)
        out = self.cnn3(periodic_padding(out,2)) #third convolutional layer
        out = self.BN3(out)
        out = self.LeakyReLU(out)
        out = self.cnn4(periodic_padding(out,2)) #fourth conv. layer
        out = self.BN4(out)
        out = self.LeakyReLU(out)

        x1 = out.view(out.size(0),-1) 
        x2 = pk.view(out.size(0),-1)
        out = torch.cat((x1, x2), dim=1) ### CONCANTANATE PK
        out = self.fc1(out) 			 # first fully connected layer
        out = self.LeakyReLU(out)
        out = self.dropout(out)
        out = self.fc2(out) 			# second fully connected layer
        out = self.LeakyReLU(out)
        out = self.dropout(out)
        out = self.fc3(out) 			# third fully connected layer
        out = self.LeakyReLU(out)
        out = self.dropout(out)
        out = self.fc4(out) 			# fourth fully connected layer
        
        
        return out
        

#DEFINE A DEVICE
device = torch.device('cpu')

if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')



########################################
########################################
### LOAD DATA

path = 'type_path_to_data_here/'

parameters   = [0,1,2,3,4]  #[0,1,2,3,4] ---> [Om, Ob, h, ns, s8] 
length       = 2000 		#dataset size
offset       = 0
total_choice = 24 			#number of available rotaions
num_rot      = 1            #number of rotations to make

### The Pk
load = np.loadtxt(path+'Pk_m_z=0.txt')  #this loads all values of k
k    = 0.4 #value of k
mask = load[:,0]<=k 					 #get power spectra with limit of k
kmask=load[mask] 					    #use in dimension of pk zero array

file_name = f'hybrid_4cnn_4fcl_{neurons}n_{k}k' # for saving/loading

### SPLIT THE DATA INTO TRAINING, VALIDATION, AND TEST SETS

set_names = 'data/3d_128_dfpk'  #remember to rename as necessary
'''
###using a data splitter
datasetsize=length
lengths = [int(datasetsize*0.8), int(datasetsize*0.1), int(datasetsize*0.1)]
torch.manual_seed(datasetsize)
print('splitting the dataset...')
           
train, validation, test = torch.utils.data.random_split(Data_set(offset = offset, length=length, parameters=parameters), lengths)

###manually
print('splitting the dataset...')

train = Data_set(0, 1600, parameters)
print('train set')
validation = Data_set(1600, 200, parameters)
print('validation set')
test = Data_set(1800, 200, parameters)
print('test set')


print('saving the datasets...')
torch.save(train, path+f'{set_names}_trainset.pt') 	        #remember to rename as necessary
torch.save(validation, 'path+f'{set_names}_validset.pt') 	#remember to rename as necessary
torch.save(test, 'path+f'{set_names}_testset.pt')  	        #remember to rename as necessary
'''
###LOAD PREVIOUSLY SAVED data sets if applicable
print('loading datasets...')
train = torch.load(path+f'{set_names}_trainset.pt')
print("train")
validation = torch.load(path+f'{set_names}__validset.pt')
print("valid")
test = torch.load(path+f'{set_names}_testset.pt')
print("test")



###DEFINE DATA SET CLASS
class Data_set(Dataset):
    def __init__(self, offset, length, parameters):
        
        ###read file with value of the cosmological parameters
        cp = np.loadtxt(path+'latin_hypercube_params.txt')
        cp = cp[:,parameters]
        mean, std = np.mean(cp, axis=0), np.std(cp,  axis=0)
        self.mean, self.std = mean, std
        cp = (cp - mean)/std #normalize the labels
        #cp = cp[offset:offset+length]
        self.cosmo_params = np.zeros((length, cp.shape[1]), dtype=np.float32)
        count = 0
        for i in range(length):
            self.cosmo_params[count] = cp[offset+i]
            count += 1
        self.cosmo_params = torch.tensor(self.cosmo_params, dtype=torch.float)


        ### read all 3D cubes
        self.cubes = np.zeros((length, 128, 128, 128), dtype=np.float32)
        count = 0 
        for i in range(length):
            f_df = path+f'{offset+i}/df_m_128_z=0.npy'
            df = np.load(f_df)
            df = np.log(1+df)	#normalize
            self.cubes[count] = df
            count += 1
        min_df = np.min(self.cubes)
        max_df = np.max(self.cubes)
        self.cubes = (self.cubes - min_df)/(max_df - min_df) #normalize the cubes
        self.cubes = torch.tensor(self.cubes)
        self.len = length
        
        ###read all Pk
        self.pk =  np.zeros((length, len(kmask)), dtype=np.float32)
        
        count = 0
        for i in range(length):
        	pk_data = np.loadtxt(path+f'{i}/Pk_m_z=0.txt')
        	mask = pk_data[:,0] <=k
        	pk_data = pk_data[mask][:,1] #get spectra with limit of k
        	pk_data = np.log(1+pk_data) #normalize
        	self.pk[count] = pk_data
        	count += 1
        min_pk = np.min(self.pk)
        max_pk = np.max(self.pk)
        self.pk = (self.pk - min_pk)/(max_pk - min_pk) #normalize the spectra
        
        self.pk = torch.tensor(self.pk)
        self.len = length
        
    def __getitem__(self, index):
        return self.cubes[index].unsqueeze(0), self.cosmo_params[index], self.pk[index]
    
    def __len__(self):
        return self.len



########################################
########################################
### TRAIN THE MODEL
if training == True:
    print('preparing to train...')
    
    batch_size   = 16		#size of batches
    epochs       = 1500   	#number of epochs
    lr           = 1e-4		#learning rate
    weight_decay = 1e-6   	#weight decay
    
    f_losses  = path+f'losses/losses_{file_name}.txt'    # losses
    f_model   = path+f'best_models/model_{file_name}.pt' # best model
    num_workers = 1
    
    cosmo_dict = {0:'dOm', 1:'dOb', 2:'dh', 3:'dns', 4:'ds8'}	#dictionary to hold parameter names
    
    print('preparing dataset loaders')
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,shuffle=True, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(validation, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    num_train_batches = len(train_loader)
    num_valid_batches = len(valid_loader)
    
    ### If output file exists, delete it
    if os.path.exists(f_losses):  os.system('rm %s'%f_losses)
    #if os.path.exists(f_model):  os.system('rm %s'%f_model)  #DELETE
    
    
    ### define model, loss and optimizer 
    print('Initializing...')
    model = CNN1(params=len(parameters)) #model
    model.to(device=device)
    
    if os.path.exists(f_model):  
        print("Loading model : Are we sure we want to load now ?")
        model.load_state_dict(torch.load(f_model))
        model.to(device=device)
        #print(model.state_dict())
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), 			
    						weight_decay=weight_decay)	#optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
    				factor=0.5, patience=25, verbose=True, eps=1e-6) #reduce learning rate
    
    criterion = nn.MSELoss() #criterion
    #criterion = nn.L1Loss()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('total number of parameters in the model =',pytorch_total_params)
    
    
    #get validation loss
    print('Computing initial validation loss')
    model.eval()
    min_valid_loss = 0.0
    for x_val, y_val, pk_val in valid_loader:
        with torch.no_grad():
            x_val = x_val.to(device=device)
            y_val = y_val.to(device=device)
            pk_val= pk_val.to(device=device)
            y_pred2 = model(x_val, pk_val)
            min_valid_loss += criterion(y_pred2, y_val).item()
    
    min_valid_loss /= num_valid_batches
    print('Initial valid loss = %.3e'%min_valid_loss)
    
    #train model
    start = time.time()
    print('Starting training...')
    for epoch in range(epochs):
        train_loss, valid_loss = 0.0, 0.0
        
        if epoch%500==0:
        	optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), 			
    						weight_decay=weight_decay)
            
        #do training
        batch_num = 0
        model.train()
        for x_train, y_train, pk_train in train_loader:
            optimizer.zero_grad()
            for i in range(batch_size):
            	flip = np.random.choice(2, 1)
            	if flip ==0:
            		x_flip = np.fliplr(x_train[i,0,:,:])
            		#print('flipped', x_flip)
            	else:
            		x_flip = x_train[i,0,:,:]
            		#print('not flipped', x_flip)
            	choice = np.random.choice(total_choice, num_rot)
            	rot = np.ascontiguousarray(random_rotation(x_flip,choice))
            	x_train[i,0,:,:] = torch.from_numpy(rot)
            x_train=x_train.to(device)
            y_train=y_train.to(device)
            pk_train=pk_train.to(device)
            #print('choice', choice)
            #print('augmented size', x_train.shape)
    
    
            y_pred = model(x_train, pk_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            #if batch_num==50:  break
            batch_num += 1
        train_loss /= batch_num
    
        # do validation
        model.eval()
        error = torch.zeros(len(parameters))
        for x_val, y_val, pk_val in valid_loader:
            with torch.no_grad():
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                pk_val = pk_val.to(device)
                y_pred2 = model(x_val, pk_val)
                valid_loss += criterion(y_pred2, y_val).item()
                error += torch.sum((y_pred2.cpu() - y_val.cpu())**2, dim=0)
                
        valid_loss /= num_valid_batches
        error /= len(validation)
    
        scheduler.step(valid_loss)
    
        ### save model if it is better
        if valid_loss<min_valid_loss:
            print('SAVING MODEL...')
            torch.save(model.state_dict(), f_model)
            min_valid_loss = valid_loss
    
        ### print some information
        print('%03d ---> train loss = %.3e : valid loss = %.3e'\
              %(epoch, train_loss, valid_loss))
        for i,j in enumerate(parameters):
            print('%03s = %.3f'%(cosmo_dict[j], error[i]))
        
        ### save losses to file
        f = open(f_losses, 'a')
        f.write('%d %.5e %.5e\n'%(epoch, train_loss, valid_loss))
        f.close()
        
    stop = time.time()
    print('Time (m):', "{:.4f}".format((stop-start)/60.0))



########################################
########################################
###TEST THE MODEL
if testing == True:

    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, 
                                              shuffle=True, num_workers=num_workers)
    
    test_results = path+f'predictions/results_{file_name}.pt'
    
    mean, std = [0.3, 0.05, 0.7, 1.0, 0.8], [0.11547004, 0.011547, 0.11547004, 0.11547004, 0.11547004] #computed beforhenad
    
    
    #If output file exists, delete it
    if os.path.exists(test_results):  os.system('rm %s'%test_results)
    
    num_params = len(parameters)
    
    #get the pretrained model
    model.load_state_dict(torch.load(f_model))
    
    # get parameters from trained network
    print('Testing results...')
    model.eval()
    results = np.zeros((len(test), 2*num_params), dtype=np.float32)
    offset = 0
    test_error = torch.zeros(len(parameters))
    for x_test, y_test, pk_test in test_loader:
        with torch.no_grad():
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            pk_test = pk_test.to(device)
            y_pred3 = model(x_test, pk_test)
            length = len(x_test)
            results[offset:offset+length, 0:num_params]            = mean + y_test.cpu().numpy()*std
            results[offset:offset+length, num_params:2*num_params] = mean + y_pred3.cpu().numpy()*std
            offset += length
            test_error += torch.sum((y_pred3.cpu() - y_test.cpu())**2, dim=0)
    
    test_error /= len(test)   
    np.savetxt(test_results, results) #save prediction results
    
    for i,j in enumerate(parameters):
        print('%s = %.3f'%(cosmo_dict[j], test_error[i])) #print test error on paramaters
        
    print('results saved as', test_results)
    
