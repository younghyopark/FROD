import torch.nn as nn
import torch


class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super().__init__()
        self.enc = base_encoder(pretrained=False)  # load model from torchvision.models without pretrained weights.
        self.feature_dim = self.enc.fc.in_features

        # Customize for CIFAR10. Replace conv 7x7 with conv 3x3, and remove first max pooling.
        # See Section B.9 of SimCLR paper.
        self.enc.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.enc.maxpool = nn.Identity()
        self.enc.fc = nn.Identity()  # remove final fully connected layer.

        # Add MLP projection.
        self.projection_dim = projection_dim
        self.projector = nn.Sequential(nn.Linear(self.feature_dim, 2048),
                                       nn.ReLU(),
                                       nn.Linear(2048, projection_dim))
        
    def forward(self, x):
        feature = self.enc(x)
        projection = self.projector(feature)
        return feature, projection

    

class SimCLR_FROD(nn.Module):
    def __init__(self, base_encoder, AE, projection_dim=128):
        super().__init__()
        self.enc = base_encoder(pretrained=False)  # load model from torchvision.models without pretrained weights.
        self.feature_dim = self.enc.fc.in_features

        # Customize for CIFAR10. Replace conv 7x7 with conv 3x3, and remove first max pooling.
        # See Section B.9 of SimCLR paper.
        self.enc.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.enc.maxpool = nn.Identity()
        self.enc.fc = nn.Identity()  # remove final fully connected layer.

        # Add MLP projection.
        self.projection_dim = projection_dim
        self.projector = nn.Sequential(nn.Linear(self.feature_dim, 2048),
                                       nn.ReLU(),
                                       nn.Linear(2048, projection_dim))
        
        self.enc_midlayers=nn.Sequential(nn.Sequential(self.enc.conv1,self.enc.bn1,self.enc.relu,self.enc.maxpool),self.enc.layer1[0],self.enc.layer1[1],self.enc.layer2[0],self.enc.layer2[1],self.enc.layer3[0],self.enc.layer3[1],self.enc.layer4[0],self.enc.layer4[1])
        
        self.midlayers_num=len(self.enc_midlayers)
        self.AE = nn.Sequential(AE(64, 32, 16, 8,4,0,0),
                               AE(64, 32, 16, 8,4,0,0),
                               AE(64, 32, 16, 8,4,0,0),
                               AE(128, 64, 32, 16,8,4,0),
                                AE(128, 64, 32, 16,8,4,0),
                                AE(256, 128, 64, 32, 16, 8,4),
                                AE(256, 128, 64, 32, 16, 8,4),
                                AE(512,256,128,64,32,8,4),
                                AE(512,256,128,64,32,8,4),
                                
                               )
#     def recon_error(self, x):
#         z = self.encoder(x)
#         x_recon = self.decoder(z)
#         return torch.norm((x_recon - x), dim=1)
    
#     def forward(self, x):
#         z = self.encoder(x)
#         return self.decoder(z)

    def forward(self, x):
        feature = self.enc(x)
        projection = self.projector(feature)
        return feature, projection
    
    def intermediate_features(self,x,index):
        out_features=self.enc_midlayers[:index+1](x)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)

        return out_features
    
    def recon_error(self,x,index):
        return self.AE[index].recon_error(self.intermediate_features(x,index))