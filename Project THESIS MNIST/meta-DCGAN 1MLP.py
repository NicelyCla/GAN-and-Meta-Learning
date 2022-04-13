import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
import numpy as np
import matplotlib
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from matplotlib import pyplot as plt
from tqdm import tqdm
from torchvision.utils import save_image
import torchvision.utils as vutils
import matplotlib.animation as animation
from IPython.display import HTML
from torchsummary import summary
from copy import deepcopy
from random import randrange

from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm
import os
import ftplib

matplotlib.style.use('ggplot')
plot = True

#ATTENZIONE:
#slow_mode = True e kShot_bool = False può essere estremamente lento!

slow_mode = False #attiva o disattiva la modalità slow_mode per l'antidominio

kShot_bool = False #attiva o disattiva la modalità kShot
kShot = 1

NUM_LABELS = 10

# learning parameters / configurations according to paper
batch_size = 64

image_size = 32
#nc = 1
nz = 100
lr = 0.0002
beta1 = 0.5
ngpu = 1
sample_size = 1

#reptile
innerepochs = 20 # number of epochs of each inner ADAM, 80 is better

outerstepsize0 = 0.01 # stepsize of outer optimization, i.e., meta-optimization
niterations = 40000 # number of outer updates; each iteration we sample one task and update on it


# set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#split del dataset:
######################################################################################

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()#,
    #transforms.Normalize((0.1307,), (0.3081,))
    ])

# load mnist
train_dataset = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=transform
)

# questo train loader con batch size 1 serve solo all'inizio,
# poi viene sovrascritto *
train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=1
)

# questo train loader con batch size 1 serve solo all'inizio,
# poi viene sovrascritto *
test_loader = DataLoader(
    test_dataset,
    shuffle=True,
    batch_size=1
)


# crea 10 loader diversi (uno per ogni numero)
loaders_list = [[], [], [], [], [], [], [], [], [], []]
for img, label in train_loader:
    if (not(kShot_bool)):
        loaders_list[label.item()].append((img[0], label))
    else:
        if len(loaders_list[label.item()])<kShot:
            loaders_list[label.item()].append((img[0], label))
            
    #print(len(loaders_list[label.item()]))
    #print(label[0])
    #break

loader_single_class = []
for data_list in loaders_list:
    x = torch.stack(list(zip(*data_list))[0])
    y = torch.stack(list(zip(*data_list))[1])

    dataset = TensorDataset(x, y)
    # ogni volta che si itera il dataset, viene fatto a random
    loader_single_class.append(DataLoader(
            dataset,
            sampler=RandomSampler(dataset),
            batch_size=batch_size
        )
    )



# crea 10 loader diversi (uno per ogni numero)
loaders_list = [[], [], [], [], [], [], [], [], [], []]
for img, label in test_loader:
    if (not(kShot_bool)):
        loaders_list[label.item()].append((img[0], label))
    else:
        if len(loaders_list[label.item()])<kShot:
            loaders_list[label.item()].append((img[0], label))
            
loader_single_class_test = []
for data_list in loaders_list:
    x = torch.stack(list(zip(*data_list))[0])
    y = torch.stack(list(zip(*data_list))[1])
    dataset = TensorDataset(x, y)
    # ogni volta che si itera il dataset, viene fatto a random
    loader_single_class_test.append(DataLoader(
            dataset,
            sampler=RandomSampler(dataset),
            batch_size=batch_size
        )
    )

# creo l'antidominio utile per creare dataset da liste
######################################################################
def createAntidomain(domain, train):
    loaders_list=[[]]
    label_cat=[]
    img_cat=[]
    #scorro le mie 10 classi e creo una lista con tutti gli elementi per ogni classe
    if(train):
        loader = loader_single_class
    else:
        loader = loader_single_class_test
        
    for i,data_single_class in enumerate(loader):
        for j,(img,label) in enumerate(data_single_class):
            #print(label.view(-1))
            if label[0].item() != domain:
                label_cat.append(label)
                img_cat.append(img)
            
    loaders_list[0].append((img_cat, label_cat))
    #i tensori dentro la lista vengono tutti concatenati
    label_cat = torch.cat(label_cat, dim=0)
    img_cat = torch.cat(img_cat, dim=0)

    #create antidomain
    antiDomain = []
    for data_list in loaders_list:
        x = torch.stack(list(img_cat))
        y = torch.stack(list(label_cat))

        dataset = TensorDataset(x, y)
        # ogni volta che si itera il dataset, viene fatto a random
        #b_size è il batch_size per la modalità lenta anche qualora non ci sia il kshot attivato
        b_size=0
        if slow_mode == True and kShot_bool == False:
            b_size = batch_size
        else:
            b_size = kShot
        antiDomain.append(DataLoader(
                dataset,
                sampler=RandomSampler(dataset),
                batch_size=b_size
            )
        )
    return antiDomain[0]
#######################################################################

# train loader con batch size corretto
train_loader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=batch_size
)
test_loader = DataLoader(
    test_dataset,
    sampler=RandomSampler(test_dataset),
    batch_size=batch_size
)

######################################################################################
#MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(nz, nz),
            nn.ReLU(),
            nn.Linear(nz, nz)
        )

    def forward(self, input):
        return self.main(input)

# generator
class Generator(nn.Module):
    def __init__(self, nz = 100, ngf = 64, nc = 1):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()

        )


    def forward(self, input):
        return self.main(input)

# discriminator
class Discriminator(nn.Module):
    def __init__(self, nc = 1, ndf = 64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(

            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

        )

        self.conv1 = nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False)
        self.sigmoid1 = nn.Sigmoid()

        self.conv2 = nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False)
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, input):
        x = self.main(input)

        out_adv = self.conv1(x)
        out_adv = self.sigmoid1(out_adv)
        
        out_cls = self.conv2(x)
        out_cls = self.sigmoid2(out_cls)

        return out_adv, out_cls

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train_G(model_G, label, optimizer_G):
    
        # (2) Update G network: maximize log(D(G(z)))

    model_G.zero_grad()
    b_size = label.size(0)#prendo il batch size dal label
    noise = torch.randn(b_size, nz, 1, 1, device=device)
    out_mlp = model_MLP(noise.view(noise.size(0), -1))
    out_mlp = out_mlp.reshape(b_size,nz,1,1)#rimodello il tensore
    # Generate fake image batch with G
    fake = model_G(out_mlp.detach())
    
    label.fill_(real_label)  # fake labels are real for generator cost
    # Since we just updated D, perform another forward pass of all-fake batch through D
    output, out_cls = model_D(fake)
    output = output.view(-1)
    out_cls = out_cls.view(-1)
    # Calculate G's loss based on this output
    errG = criterion(output, label)
    err_out_cls = criterion(out_cls,label)
    #Classify out_cls as real
    errG = errG + err_out_cls
    # Calculate gradients for G
    errG.backward()
    D_G_z2 = output.mean().item()
    # Update G
    optimizer_G.step()

    return errG


def train_GAN_on_task(model_G, model_D, optimizer_G, optimizer_D, single_loader, train_loader, train):
    for i in range(innerepochs):
        data = next(iter(single_loader))
        dominio_corrente = data[1][0].item()

        #distinguo due diverde modalità
        if (not(kShot_bool) and slow_mode == False):
            #print("fast_mode")
            data_full = next(iter(train_loader))
            #rendo invisibile le immagini del dominio corrente tramite una "censura"
            ######################################################################################
            for j in range(len(data_full[1])):
                if data_full[1][j].item() == dominio_corrente:
                    #censuro con del rumore
                    '''
                    data_full[1][j] = torch.ones(1)*(-1)
                    data_full[0][j] = torch.randn([1,32,32]) #censuro l'immagine
                    '''
                    #censuro con una ridondanza di immagini
                    while data_full[1][j].item() == dominio_corrente:
                        rnd = randrange(len(data_full[1]))
                        data_full[1][j] = data_full[1][rnd]
                        data_full[0][j] = data_full[0][rnd]
                        
        else:
            #print("slow mode or kshot")
            if i == 0:
                antiDomain = createAntidomain(dominio_corrente, train)
            data_full = next(iter(antiDomain))
        '''
        print(dominio_corrente)
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(data_full[0], padding=2).cpu(),(1,2,0)))
        plt.show()
        '''
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        
        ## Train with all-real batch
        model_D.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        real_full = data_full[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output,_ = model_D(real_cpu)
        _,out_cls = model_D(real_cpu)
        
        output = output.view(-1)
        out_cls = out_cls.view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        err_out_cls = criterion(out_cls,label)
        #Classify out_cls as real
        errD_real = errD_real + err_out_cls
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        out_mlp = model_MLP(noise.view(noise.size(0), -1))
        out_mlp = out_mlp.reshape(b_size,100,1,1)#rimodello il tensore
                
        # Generate fake image batch with G
        fake = model_G(out_mlp.detach())
        label.fill_(fake_label)
        # Classify all fake batch with D
        output, _ = model_D(fake.detach())
        output = output.view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        #errD_fake.backward()
        #D_G_z1 = output.mean().item()
        # classify out_cls as wrong domain
        _, out_cls = model_D(real_full)
        out_cls=out_cls.view(-1)
        # calculate criterion
        err_out_cls = criterion(out_cls,label)
        # Add the gradients from the all-real and all-fake domain
        errD_fake = errD_fake + err_out_cls
        #backpropagate
        errD_fake.backward()
        # Update D
        optimizer_D.step()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake


        model_MLP.zero_grad()
        '''
        out_mlp = model_MLP(noise.view(noise.size(0), -1))
        out_mlp = out_mlp.reshape(b_size,100,1,1)#rimodello il tensore
        '''
        out_gen = model_G(out_mlp)
        out_real, out_dom = model_D(out_gen)

        label.fill_(real_label)
        out_real=out_real.view(-1)
        out_dom=out_dom.view(-1)
        errD_real = criterion(out_real, label) #il label reale
        err_out_cls = criterion(out_dom,label)
        
        err_mlp = errD_real+err_out_cls
        err_mlp.backward()
        optimizer_MLP.step()
        
    # Output training stats

    return errD, err_mlp, label

model_D = Discriminator().to(device)#.cuda()
model_D.apply(weights_init)

model_G = Generator().to(device)#.cuda()
model_G.apply(weights_init)

model_MLP = MLP().to(device)#.cuda()
model_MLP.apply(weights_init)


fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Initialize BCELoss function
criterion = nn.BCELoss()
#criterion_class = nn.BCELoss()

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizer_D = optim.Adam(model_D.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_G = optim.Adam(model_G.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_MLP = optim.Adam(model_MLP.parameters(), lr=lr, betas=(beta1, 0.999))

loss_task_G = [[], [], [], [], [], [], [], [], [], []]
loss_task_D = [[], [], [], [], [], [], [], [], [], []]
loss_task_MLP = [[], [], [], [], [], [], [], [], [], []]

PATH='./generate/'

# Reptile training loop
for iteration in tqdm(range(niterations)):
    random = randrange(NUM_LABELS)
    
    #weights_before_G = deepcopy(model_G.state_dict())
    weights_before_D = deepcopy(model_D.state_dict())
    weights_before_MLP = deepcopy(model_MLP.state_dict())

    # Generate task
    single_loader = loader_single_class[random] # random from 0 to 9
    
    
    # train the GAN model with task images
    err_D, err_MLP, label = train_GAN_on_task(model_G, model_D, optimizer_G, optimizer_D, single_loader, train_loader, train=True)
    err_G = train_G(model_G, label, optimizer_G)

    loss_task_G[random].append(err_G)
    loss_task_D[random].append(err_D)
    loss_task_MLP[random].append(err_MLP)

    
    # Interpolate between current weights and trained weights from this task
    # I.e. (weights_before - weights_after) is the meta-gradient
    outerstepsize = outerstepsize0 * (1 - iteration / niterations) # linear schedule
    '''
    weights_after_G = model_G.state_dict()
    model_G.load_state_dict({name : 
        weights_before_G[name] + (weights_after_G[name] - weights_before_G[name]) * outerstepsize 
        for name in weights_before_G})
    '''
    
    weights_after_D = model_D.state_dict()
    model_D.load_state_dict({name : 
        weights_before_D[name] + (weights_after_D[name] - weights_before_D[name]) * outerstepsize 
        for name in weights_before_D})

    weights_after_MLP = model_MLP.state_dict()
    model_MLP.load_state_dict({name : 
        weights_before_MLP[name] + (weights_after_MLP[name] - weights_before_MLP[name]) * outerstepsize 
        for name in weights_before_MLP})
    
    # Periodically plot the results on a particular task and minibatch
    #TEST
    if plot and (iteration==0 or iteration%1000 == 0 or iteration+1==niterations):

        for n in range(NUM_LABELS):
            weights_before_D = deepcopy(model_D.state_dict()) # save snapshot before evaluation
            weights_before_G = deepcopy(model_G.state_dict()) # save snapshot before evaluation
            weights_before_MLP = deepcopy(model_MLP.state_dict()) # save snapshot before evaluation

            single_loader_test = loader_single_class_test[n] #selezioni il loader della classe n

            # aggiorno la rete su una singola classe
            err_D, err_MLP, label = train_GAN_on_task(model_G, model_D, optimizer_G, optimizer_D, single_loader_test, test_loader, train=False)
            err_G = train_G(model_G, label, optimizer_G)

            with torch.no_grad():
                
                out_mlp = model_MLP(fixed_noise.view(fixed_noise.size(0), -1))
                out_mlp = out_mlp.reshape(fixed_noise.size(0),nz,1,1)#rimodello il tensore
                out_imgs_fake = model_G(out_mlp)
                
            # salvi le immagini generate
            save_image(out_imgs_fake,f"{PATH}{iteration}_{n}.png") #aggiungi percorso: "path/iterazione_classe.png" es "pippo/20000_3.png"
                
            # ripristini il modello prima dell'aggiornamento su una singola classe
            model_D.load_state_dict(weights_before_D) # restore from snapshot   
            model_G.load_state_dict(weights_before_G)
            model_MLP.load_state_dict(weights_before_MLP)



    torch.cuda.empty_cache()

#mostra il grafico dei Loss per ogni task
for n in range(NUM_LABELS):
    plt.figure(figsize=(10,5))
    plt.title(f"D, MLP and G Loss During Training On Task: {n}")
    plt.plot(loss_task_D[n],label="D")
    plt.plot(loss_task_MLP[n],label="MLP")
    plt.plot(loss_task_G[n],label="G")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    #plt.show()
    plt.savefig(f'{PATH}D, MLP and G Loss During Training On Task: {n}.png')
    plt.close('all')

torch.save(model_D.state_dict(),f'{PATH}discriminator.pth')
torch.save(model_G.state_dict(), f'{PATH}generator.pth')
torch.save(model_MLP.state_dict(), f'{PATH}mlp.pth')
