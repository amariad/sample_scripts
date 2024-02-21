
import numpy as np

### RANDOM CUBE ROTATION
# this function chooses a random rotation of a cube 
#for data augmentation during training of a cnn

def random_rotation(dataset, rotation, flip):
    
    if rotation ==0:
        choice = np.rot90(dataset, 1, (0,1)) #this represents the x-axis rotations
    elif rotation ==1:
        choice = np.rot90(dataset, 1, (0,2)) #this represents the y-axis rotations
    elif rotation ==2:
        choice = np.rot90(dataset, 1, (1,2)) #this represents the z-axis rotations
    # let's get the combined rotations:
    elif rotation ==3:
        choice = np.rot90(dataset, 2, (0,1)) #this represents the x-axis only rotations
    elif rotation ==4:
        choice = np.rot90(dataset, 3, (0,1))
    #combine rotations from x-axis rotations:
    elif rotation ==5:
        choice = np.rot90(np.rot90(dataset, 2, (0,1)), 1, (0,2)) # additional rotation along y-axis
    elif rotation ==6:
        choice= np.rot90(np.rot90(dataset, 3, (0,1)), 1, (0,2)) # additional rotation along y-axis
    elif rotation ==7:
        choice = np.rot90(np.rot90(dataset, 2, (0,1)), 1, (1,2)) # additional rotation along z-axis
    elif rotation ==8:
        choice = np.rot90(np.rot90(dataset, 3, (0,1)), 1, (1,2)) # additional rotation along z-axis
    #now we go back and do the y-axis only rotations:
    elif rotation ==9:
         choice = np.rot90(dataset, 2, (0,2)) #this represents the y-axis only rotations
    elif rotation ==10:
         choice= np.rot90(dataset, 3, (0,2))
    #combine rotations from y-axis rotations:
    elif rotation ==11:
         choice= np.rot90(np.rot90(dataset, 2, (0,2)), 1, (0,1)) # additional rotation along x-axis
    elif rotation ==12:
        choice = np.rot90(np.rot90(dataset, 3, (0,2)), 1, (0,1)) # additional rotation along x-axis
    elif rotation ==13:
        choice = np.rot90(np.rot90(dataset, 2, (0,2)), 1, (1,2)) # additional rotation along z-axis
    elif rotation ==14:
        choice = np.rot90(np.rot90(dataset, 3, (0,2)), 1, (1,2)) # additional rotation along z-axis
    # now we go back and do z-axis only rotations:
    elif rotation ==15:
         choice = np.rot90(dataset, 2, (1,2)) #this represents the z-axis rotations
    elif rotation ==16:
        choice = np.rot90(dataset, 3, (1,2))
    #combine rotations from z-axis rotations:
    elif rotation ==17:
        choice = np.rot90(np.rot90(dataset, 2, (1,2)), 1, (0,1)) # additional rotation along x-axis
    elif rotation ==18:
        choice = np.rot90(np.rot90(dataset, 3, (1,2)), 1, (0,1)) 
    elif rotation ==19:
        choice = np.rot90(np.rot90(dataset, 2, (1,2)), 1, (0,2))# additional rotation along y-axis
    elif rotation ==20:
        choice = np.rot90(np.rot90(dataset, 3, (1,2)), 1, (0,2))
    #we have a single 1x1 rotation
    elif rotation ==21:
        choice = np.rot90(np.rot90(dataset, 1, (0,1)), 1, (1,2))# 1 rotation along x and 1 along z 
    #we have a single 3x3 rotation
    elif rotation ==22:
        choice = np.rot90(np.rot90(dataset, 3, (0,2)), 3, (1,2))# 3 rotation along x and 3 along y 
    #let's include the identity
    elif rotation ==23:
        choice = dataset #no rotations

    if flip == 1:
    	choice = np.fliplr(choice)
    	#print('flipped', x_flip)
    	
    	
    return choice



###PERIODIC PADDING

# This function implements PERIODIC BOUNDARY conditions for convolutional neural nets
# df ----------> density field
# padding -----> number of elements to pad

def periodic_padding(df, padding):
    
    #y       z
    #y     z
    #y   z
    #y z
    #0 x x x x

    right = df[:,:,-padding:, :, :]
    left  = df[:,:,:padding, :, :]
    df = torch.cat((df,    left), dim=2)
    df = torch.cat((right, df),   dim=2)

    top    = df[:,:,:, -padding:, :]
    bottom = df[:,:,:, :padding, :]
    df = torch.cat((df,  bottom), dim=3)
    df = torch.cat((top, df),     dim=3)

    front = df[:,:,:,:,-padding:]
    back  = df[:,:,:,:,:padding]
    df = torch.cat((df,    back),  dim=4)
    df = torch.cat((front, df),    dim=4)

    return df

