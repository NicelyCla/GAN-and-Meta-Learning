import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from copy import deepcopy
from random import randrange
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch.autograd import Variable, grad
from matplotlib import pyplot as plt
from os import path
from IPython.display import clear_output# Hyperparameters etc.

device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
batch_size = 64
IMAGE_SIZE = 32
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 20
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10
NUM_LABELS = 10
niterations = 20000
outerstepsize0 = 0.01

mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)

PATH = './generate/'
PATH_MODEL = './modelli/'
#split del dataset:
######################################################################################

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
    ])

# load mnist
train_dataset = datasets.CIFAR10(
    root='data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
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
    loaders_list[label.item()].append((img[0], label))
    
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
        b_size = batch_size

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


class Discriminator(nn.Module):
    def __init__(self, nc = 3, ndf = 64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 32x32
            nn.Conv2d(nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x4
        )
        self.conv1 = nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False)
        self.conv2 = nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False)

    def forward(self, input):
        x = self.main(input)
        out_adv = self.conv1(x)
        out_cls = self.conv2(x)
        
        return out_adv, out_cls

#MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100)
        )

    def forward(self, input):
        return self.main(input)

class Generator(nn.Module):
    def __init__(self, ngf = 64, nc = 3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( 100, ngf * 4, 4, 1, 0, bias=False),
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


def initialize_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores,_ = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def gradient_penalty_domain(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    _,mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty




def train_generator():
    # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
    gen.zero_grad()
    noise = torch.randn(batch_size, Z_DIM, 1, 1).to(device)
    out_mlp = mlp(noise.view(noise.size(0), -1))
    z_mlp = out_mlp.reshape(batch_size,Z_DIM,1,1)#rimodello il tensore
    
    out_gen = gen(z_mlp)
    gen_fake, gen_fake_domain = critic(out_gen)
    gen_fake = gen_fake.reshape(-1)
    gen_fake_domain = gen_fake_domain.reshape(-1)
    loss_gen = -(torch.mean(gen_fake) + torch.mean(gen_fake_domain))
    loss_gen.backward()
    opt_gen.step()
    
    return loss_gen


def train_critic_on_task(single_loader, train_loader):
    for epoch in range(NUM_EPOCHS):
        # Target labels not needed! <3 unsupervised
        real, lab = next(iter(single_loader))
        real = real.to(device)
        dominio_corrente = lab[0].item()
        cur_batch_size = real.shape[0]

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
        #######################################################################################
        real_full = data_full[0].to(device)
        real_full = Variable(real_full, requires_grad = True)
        
        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        for _ in range(CRITIC_ITERATIONS):
            critic.zero_grad()

            #train with real
            critic_real, critic_real_domain = critic(real)
            critic_real = critic_real.reshape(-1)
            critic_real_domain = critic_real_domain.reshape(-1)
            #train with fake
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            out_mlp = mlp(noise.view(noise.size(0), -1))
            z_mlp = out_mlp.reshape(cur_batch_size,Z_DIM,1,1)#rimodello il tensore    
            fake = gen(z_mlp)
            critic_fake, _ = critic(fake)
            _ , critic_fake_domain = critic(real_full)
            critic_fake = critic_fake.reshape(-1)
            critic_fake_domain = critic_fake_domain.reshape(-1)
            
            #train with gp
            gp = gradient_penalty(critic, real, fake, device=device)

            #train with gp domain
            gp_domain = gradient_penalty_domain(critic, real, real_full, device=device)
            
            #total loss of discr
            '''
            loss_critic = (
                -((torch.mean(critic_real)+torch.mean(critic_real_domain)) - (torch.mean(critic_fake)+torch.mean(critic_fake_domain))) + LAMBDA_GP * gp
            )
            '''
            loss_critic = (
                (-(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp) +
                (-(torch.mean(critic_real_domain) - torch.mean(critic_fake_domain)) + LAMBDA_GP * gp_domain)
            )
            
            #backward
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        #train MLP
        mlp.zero_grad()
        out_gen = gen(z_mlp)
        gen_fake, gen_fake_domain = critic(out_gen)
        gen_fake = gen_fake.reshape(-1)
        gen_fake_domain = gen_fake_domain.reshape(-1)
        loss_mlp = -(torch.mean(gen_fake) + torch.mean(gen_fake_domain))
        loss_mlp.backward(retain_graph=True)
        opt_mlp.step()
        
    return loss_critic, loss_mlp

        # Print losses occasionally and print to tensorboard
    '''
    print(
        f"Epoch [{epoch}/{NUM_EPOCHS}] \ Loss D: {loss_critic:.4f}, Loss MLP: {loss_mlp:.4f}"
    )
    '''



# initialize gen and disc, note: discriminator should be called critic,
# according to WGAN paper (since it no longer outputs between [0, 1])
gen = Generator().to(device)
critic = Discriminator().to(device)
mlp = MLP().to(device)

initialize_weights(gen)
initialize_weights(critic)
initialize_weights(mlp)

if path.exists(f"{PATH_MODEL}discriminator.pth") and path.exists(f"{PATH_MODEL}generator.pth") and path.exists(f"{PATH_MODEL}mlp.pth"):
    critic.load_state_dict(torch.load(f"{PATH_MODEL}discriminator.pth"))
    gen.load_state_dict(torch.load(f"{PATH_MODEL}generator.pth"))
    mlp.load_state_dict(torch.load(f"{PATH_MODEL}mlp.pth"))
    print("Tutti i modelli sono stati trovati e caricati correttamente!")
else:
    print("Non sono stati trovati modelli preaddestrati, procedo all'addestramento inizializzandoli con weights_init")


if path.exists(f"{PATH_MODEL}fixed_noise.pt"):
    fixed_noise = torch.load(f'{PATH_MODEL}fixed_noise.pt')
    print("Il rumore è stato trovato e caricato correttamente!")

else:
    fixed_noise = torch.randn(batch_size, Z_DIM, 1, 1).to(device)
    torch.save(fixed_noise, f'{PATH_MODEL}fixed_noise.pt')
    print("Non è stato trovato nessun rumore salvato, procedo alla creazione e al salvataggio di un rumore casuale")



# initializate optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_mlp = optim.Adam(mlp.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

# for tensorboard plotting
fixed_noise = torch.randn(batch_size, Z_DIM, 1, 1).to(device)

gen.train()
critic.train()
mlp.train()

loss_task_G = [[], [], [], [], [], [], [], [], [], []]
loss_task_D = [[], [], [], [], [], [], [], [], [], []]
loss_task_MLP = [[], [], [], [], [], [], [], [], [], []]

for iteration in tqdm(range(niterations)):
    random = randrange(NUM_LABELS)

    weights_before_D = deepcopy(critic.state_dict())
    weights_before_MLP = deepcopy(mlp.state_dict())

    # Generate task
    single_loader = loader_single_class[random] # random from 0 to 9

    # train the GAN model with task images    
    errD, err_mlp = train_critic_on_task(single_loader, train_loader)
    errG = train_generator()

    loss_task_G[random].append(errG.item())
    loss_task_D[random].append(errD.item())
    loss_task_MLP[random].append(err_mlp.item())

    # Interpolate between current weights and trained weights from this task
    # I.e. (weights_before - weights_after) is the meta-gradient  
    outerstepsize = outerstepsize0 * (1 - iteration / niterations) # linear schedule

    weights_after_D = critic.state_dict()
    critic.load_state_dict({name : 
        weights_before_D[name] + (weights_after_D[name] - weights_before_D[name]) * outerstepsize 
        for name in weights_before_D})
    
    weights_after_MLP = mlp.state_dict()
    mlp.load_state_dict({name : 
        weights_before_MLP[name] + (weights_after_MLP[name] - weights_before_MLP[name]) * outerstepsize 
        for name in weights_before_MLP})

    # Periodically plot the results on a particular task and minibatch
    #TEST    
    if (iteration%100 == 0 or iteration+1==niterations) and iteration != 0:
        for n in range(NUM_LABELS):
            weights_before_D = deepcopy(critic.state_dict()) # save snapshot before evaluation
            weights_before_MLP = deepcopy(mlp.state_dict()) # save snapshot before evaluation

            single_loader_test = loader_single_class_test[n] #selezioni il loader della classe n

            # aggiorno la rete su una singola classe
            errD, err_mlp = train_critic_on_task(single_loader_test, test_loader)

            with torch.no_grad():
                out_mlp = mlp(fixed_noise.view(fixed_noise.size(0), -1))
                z_mlp = out_mlp.reshape(batch_size,Z_DIM,1,1)#rimodello il tensore
                fake = gen(z_mlp)

            # salvi le immagini generate
            save_image(fake,f"{PATH}{iteration}_{n}.png", normalize = True) #aggiungi percorso: "path/iterazione_classe.png" es "pippo/20000_3.png"
                
            # ripristini il modello prima dell'aggiornamento su una singola classe
            critic.load_state_dict(weights_before_D) # restore from snapshot   
            mlp.load_state_dict(weights_before_MLP)

        torch.save(critic.state_dict(),f'{PATH_MODEL}discriminator.pth')
        torch.save(gen.state_dict(), f'{PATH_MODEL}generator.pth')
        torch.save(mlp.state_dict(), f'{PATH_MODEL}mlp.pth')

        #mostra il grafico dei Loss per ogni task
        for n in range(NUM_LABELS):
            plt.figure(figsize=(10,5))
            plt.title(f"D, MLP, MLP_cls and G Loss During Training On Task: {n}")
            plt.plot(loss_task_D[n],label="D")
            plt.plot(loss_task_MLP[n],label="MLP")
            plt.plot(loss_task_G[n],label="G")
            plt.xlabel("iterations")
            plt.ylabel("Loss")
            plt.legend()
            #plt.show()
            plt.savefig(f'{PATH}D, MLP, MLP_cls and G Loss During Training On Task: {n}.png')
            plt.close('all')
        
    torch.cuda.empty_cache()
