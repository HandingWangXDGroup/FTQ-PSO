#    Authors:    Chao Li, Tingsong Jiang，Handing Wang, Wen Yao, Donghua Wang
#    Xidian University, China
#    Defense Innovation Institute, Chinese Academy of Military Science, China
#    Zhongyuan University of Technology, China
#    EMAIL:      lichaoedu@126.com
#    DATE:       September 2024
# ------------------------------------------------------------------------
# This code is part of the program that produces the results in the following paper:
#
# Chao Li, Tingsong Jiang，Handing Wang, Wen Yao, Donghua Wang, Optimizing Latent Variables in Integrating Transfer and Query Based Attack Framework, IEEE Transactions on  Pattern Analysis and Machine Intelligence, 2024.
#
# You are free to use it for non-commercial purposes. However, we do not offer any forms of guanrantee or warranty associated with the code. We would appreciate your acknowledgement.
# ------------------------------------------------------------------------

import argparse
import os
import random
import numpy as np
import torch.nn.functional as F
from Wgenerator import Generator, weights_init_normal
import torch
from utils import get_model
from data_loader import get_dataset
import yaml


def train(data_loader, loss_type="cw"):
    fr_num = 0
    total_num = 0
    generator.train()
    for i, batch in enumerate(data_loader):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        latent_dim = torch.rand((args.batch_size, args.latent_dim)).to(device)
        pert = generator(latent_dim)
        pert = torch.clamp(pert, -args.epsilon, args.epsilon)
        adv_image = torch.clamp(images + pert, 0, 1)
        adv_output = model(adv_image)
        loss_perturb = torch.mean(torch.norm(pert.view(pert.shape[0], -1), 2, dim=1))
        if loss_type == "cw":
            # cal adv loss
            probs_model = F.softmax(adv_output, dim=1)
            onehot_labels = torch.eye(1000, device=device)[labels]

            # C&W loss function
            real = torch.sum(onehot_labels * probs_model, dim=1)
            other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
            zeros = torch.zeros_like(other)
            loss_adv = torch.max(real - other, zeros)
            loss_adv = torch.sum(loss_adv)
        else:
            loss_adv = -cropssentropy_loss(adv_output, labels)



        loss = loss_adv + loss_perturb
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            clean_output = model(images)
            clean_pred = torch.argmax(clean_output, dim=1)
            adv_pred = torch.argmax(adv_output, dim=1)

        fr_num += (clean_pred != adv_pred).sum().item()
        total_num += images.size(0)
        fr = round(100 * fr_num / total_num, 2)
        print(f"【{epoch}】/【{args.epochs}】: Loss: {loss.item()}, Fooling rate: {fr}")


@torch.no_grad()
def test(data_loader):
    fr_num = 0
    total_num = 0
    generator.eval()
    for i, batch in enumerate(data_loader):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        latent_dim = torch.rand((50, args.latent_dim)).to(device)
        pert = generator(latent_dim)
        pert = torch.clamp(pert, -args.epsilon, args.epsilon)
        adv_image = torch.clamp(images + pert, 0, 1)
        adv_output = model(adv_image)
        clean_output = model(images)
        clean_pred = torch.argmax(clean_output, dim=1)
        adv_pred = torch.argmax(adv_output, dim=1)
        fr_num += (clean_pred != adv_pred).sum().item()
        total_num += images.size(0)

    fr = round(100 * fr_num / total_num, 2)
    return fr


def set_random_seed(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    set_random_seed()
    device = 'cuda:0' if torch.cuda.is_available() else 'cuda'
    cfg_file = 'config.yml'

    with open(cfg_file, "r") as fr:
        yml_file = yaml.safe_load(fr)
    parser = argparse.ArgumentParser(description="parameters")
    parser.set_defaults(**yml_file)
    args = parser.parse_args()

    generator = Generator(args.image_size, args.latent_dim)
    generator.apply(weights_init_normal)
    generator.to(device)

    model = get_model(args.model_name, device)

    optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)
    cropssentropy_loss = torch.nn.CrossEntropyLoss()

    train_data = get_dataset(args)
    test_data = get_dataset(args, only_train=False)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                               pin_memory=True, num_workers=args.num_works)

    validation_loader = torch.utils.data.DataLoader(test_data, batch_size=50,
                                                    num_workers=args.num_works)
    lr_schular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_fr = 0
    for epoch in range(args.epochs):
        train(train_loader)

        if (epoch + 1) % 5 == 0:
            fr = test(validation_loader)
            print("="*20)
            print(f"Epoch: {epoch}, best fr: {best_fr}, fr: {fr}")
            print("=" * 20)
            if best_fr < fr:
                best_fr = fr
                # save generator_dict and latent_dim
                torch.save({
                    "generator": generator.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "fooling_rate": best_fr
                }, f"/mnt/best_generator_with_fr_{best_fr}.pth")
        lr_schular.step()

    print(f"Finished with best fr: {best_fr}")
