import torch.nn as nn
import torch.nn.functional as F
import torch.tensor as tensor


class Autoencoder(nn.Module):
    '''
    criterion is the criterion used for the score method
    '''
    def __init__(self, criterion): 
        super(Autoencoder, self).__init__()
        self.criterion = criterion
        # encoder
        self.enc1 = nn.Linear(in_features=122, out_features=90)
        self.enc2 = nn.Linear(in_features=90, out_features=80)
        self.enc3 = nn.Linear(in_features=80, out_features=70)
        self.enc4 = nn.Linear(in_features=70, out_features=60)
 
        # decoder
        self.dec1 = nn.Linear(in_features=60, out_features=70)
        self.dec2 = nn.Linear(in_features=70, out_features=80)
        self.dec3 = nn.Linear(in_features=80, out_features=90)
        self.dec4 = nn.Linear(in_features=90, out_features=122)

    def forward(self, x):
        x = F.leaky_relu(self.enc1(x))
        x = F.leaky_relu(self.enc2(x))
        x = F.leaky_relu(self.enc3(x))
        x = F.leaky_relu(self.enc4(x))
 
        x = F.leaky_relu(self.dec1(x))
        x = F.leaky_relu(self.dec2(x))
        x = F.leaky_relu(self.dec3(x))
        x = F.leaky_relu(self.dec4(x))
        return x
    
    '''
    Returns the anomaly score of x
    '''
    def score(self, x):
        x2 = self(x)
        anomaly_scores = []
        for i in range(len(x)):
            l = self.criterion(x[i], x2[i])
            anomaly_scores.append(l)
        return tensor(anomaly_scores)

class Generator(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(Generator, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, output_dim),
        )
        
    def forward(self,x):
        x = self.layer(x)
        return x

    
class Discriminator(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(Discriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, output_dim),
        )

    def forward(self,x):
        return self.layer(x) 
