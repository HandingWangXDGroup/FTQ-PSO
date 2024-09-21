import numpy as np
import copy
import torch.nn as nn
import torch




def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = 224 // 4
        self.l1 = nn.Sequential(nn.Linear(50, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


pop_size = 50
dim = 50
var_max = 1
var_min = 0
Vmax = 0.5 * (var_max - var_min)
Vmin = -Vmax
generator = Generator()
ckpt = torch.load(" ")
generator.load_state_dict(ckpt['generator'])
generator.cuda()
generator.eval()


def initialize():
    pop_pos = []
    pop_vel = []
    for i in range(pop_size):
        pos_rand_value = np.random.rand(dim)
        vel_rand_value = np.random.rand(dim)
        pos = var_min + pos_rand_value * (var_max - var_min)
        vel = Vmin + 2 * Vmax * vel_rand_value
        pop_pos.append(pos)
        pop_vel.append(vel)

    return pop_pos, pop_vel

def evaluate(x,img, label, model, device):



    x = torch.from_numpy(np.array(x))
    x = x.to(device)
    pert = generator(x.float())
    pert = torch.clamp(pert, -8./255, 8./255)


    adv_img = pert + img.repeat((pop_size, 1, 1, 1))
    adv_img = torch.clamp(adv_img, 0, 1)
    adv_img = adv_img.to(device)
    label = label.to(device)


    outputs = model(adv_img)
    org_out = outputs[:, label]
    min_out, _ = torch.min(outputs, dim=1)
    outputs[:, label] = min_out.unsqueeze(dim=1)
    sec_out, _ = torch.max(outputs, dim=1)
    fitness = org_out.squeeze() - sec_out

    fitness = fitness.cpu().detach().numpy()
    return fitness

@torch.no_grad()
def psog(img, label, model, device):
    fes = 0
    max_fes = 500
    pop_pos, pop_vel = initialize()
    ipop = copy.deepcopy(pop_pos)
    pfitness = evaluate(ipop, img, label, model, device)
    fes += pop_size
    pbest = copy.deepcopy(pop_pos)
    gindex = np.argmin(pfitness)
    gbest = pbest[gindex]
    while fes < max_fes:
        if min(pfitness) < 0:
            break
        for i in range(pop_size):
            pop_vel[i] = 0.7298 * pop_vel[i] + 2.05 * np.random.rand(dim) * (pbest[i] - pop_pos[i]) + 2.05 * np.random.rand(dim) * (gbest - pop_pos[i])
            pop_pos[i] = pop_pos[i] + pop_vel[i]
            pop_pos[i] = np.clip(pop_pos[i], var_min, var_max)
        offer_pop = copy.deepcopy(pop_pos)
        offer_fitness = evaluate(offer_pop, img, label, model, device)
        fes += pop_size
        for i in range(pop_size):
            if pfitness[i] > offer_fitness[i]:
                pfitness[i] = offer_fitness[i]
                pbest[i] = pop_pos[i]
        gindex = np.argmin(pfitness)
        gbest = pbest[gindex]

    gbest = torch.from_numpy(np.array(gbest))
    gbest = gbest.unsqueeze(0)
    gbest = gbest.to(device)
    final_pert = generator(gbest.float())
    final_pert = torch.clamp(final_pert, -8./255, 8./255)

    finaladv_img = torch.clamp(img + final_pert, 0, 1)
    adv_norm = torch.norm((finaladv_img - img))

    return finaladv_img, fes, adv_norm

