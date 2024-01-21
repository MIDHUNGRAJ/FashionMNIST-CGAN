import torch
import torch.nn as nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

## Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.label_conditioned = nn.Sequential(
            nn.Embedding(10,100),
            nn.Linear(100, 2*2)
        )

        self.latent = nn.Sequential(
            nn.Linear(100, 256*2*2)
        )

        self.model = nn.Sequential(
            nn.ConvTranspose2d(257, 64*8,kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64*8),
            nn.ReLU(),
            nn.ConvTranspose2d(64*8, 64*4,kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64*4),
            nn.ReLU(),
            nn.ConvTranspose2d(64*4, 64*2,kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64*2),
            nn.ReLU(),
            nn.ConvTranspose2d(64*2, 64*1,kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64*1),
            nn.ReLU(),
            nn.ConvTranspose2d(64*1, 1,kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        
        )
    
    def forward(self, input):
        noise, label = input
        label_output = self.label_conditioned(label)
        label_output = label_output.view(-1,1,2,2)

        latent_output = self.latent(noise)
        latent_output = latent_output.view(-1,256,2,2)

        concat = torch.cat((label_output, latent_output), 1) 
        
        return self.model(concat)
    

## Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.label_condition = nn.Sequential(
            nn.Embedding(10, 100),
            nn.Linear(100, 1*64*64)
        )
             
        self.model = nn.Sequential(nn.Conv2d(2, 32, 4, 2, 1, bias=False),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.Conv2d(32, 64, 4, 3, 2, bias=False),
                      nn.BatchNorm2d(64, momentum=0.1,  eps=0.8),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.Conv2d(64, 64*2, 4, 3, 2, bias=False),
                      nn.BatchNorm2d(64*2, momentum=0.1,  eps=0.8),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.Conv2d(64*2, 64*4, 4, 3, 2, bias=False),
                      nn.BatchNorm2d(64*4, momentum=0.1,  eps=0.8),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.Conv2d(64*4, 64*8, 4, 3, 2, bias=False),
                      nn.BatchNorm2d(64*8, momentum=0.1, eps=0.8),
                      nn.LeakyReLU(0.2, inplace=True), 
                      nn.Flatten(),
                      nn.Dropout(0.4),
                      nn.Linear(512, 1),
                      nn.Sigmoid()
                     )
    
    def forward(self, input):
        img, label = input
        label_output = self.label_condition(label)

        label_output = label_output.view(-1,1,64,64)
        concat = torch.cat((img, label_output), dim=1)
        return self.model(concat)


# plot images
def plot_images(image, title=None, nrow=5):
    img_show_fake = make_grid(image[:40], normalize=True, nrow=nrow)

    img_show_fake = img_show_fake.detach().cpu().permute(1, 2, 0).numpy()

    plt.imshow(img_show_fake)
    plt.title(title)
    plt.axis("off")
    plt.show()


