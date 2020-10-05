import torch
import torch.nn.functional as F
import starganv2
import torch.optim as optim
import dataloader
import random
import torchvision.utils as vutils
import os
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
import wandb

"""
since our dataset is not balanced we ensure a balanced dataset (from both domains - male and female) using a sampler. 
"""
def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels) 
    class_weights = 1/class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))

class Trainer:
    def __init__(self, config):
        """
        basic stuff defining variables from our config file. 
        config file is a yaml file and hence we use to ensure easy modifications for running multiple experiments.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

        self.root_dir = self.config["root_dir"]
        self.img_size = self.config["img_size"]
        self.style_dim = self.config["style_dim"]
        self.latent_dim = self.config["latent_dim"]
        self.num_domains = self.config["num_domains"]
        self.lr = self.config["lr"]
        self.map_lr = self.config["map_lr"]
        self.beta1 = self.config["beta1"]
        self.beta2 = self.config["beta2"]
        self.weight_decay = self.config["weight_decay"]
        self.lambda_reg = self.config["lambda_reg"]
        self.lambda_style = self.config["lambda_style"]
        self.lambda_div = self.config["lambda_div"]
        self.lambda_cyc = self.config["lambda_cyc"]
        self.init_lambda_div = self.config["init_lambda_div"]
        self.decay_div = self.config["decay_div"]
        self.batch_size = self.config["batch_size"]
        self.num_workers = self.config["num_workers"]
        self.num_debug = self.config["num_debug"]
        self.valid_every = self.config["valid_every"]
        self.max_iter = self.config["max_iter"]
        self.output_dir = self.config["output_dir"]
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            os.makedirs(os.path.join(self.output_dir, "weights"))
            os.makedirs(os.path.join(self.output_dir, "tr_and_rec"))
            os.makedirs(os.path.join(self.output_dir, "tr_with_lat"))
            os.makedirs(os.path.join(self.output_dir, "tr_with_ref"))

        self.step = 0
        """
        defining model networks and setting up optimizers
        defining datasets and validation and training dataloader.
        """
        self.generator = starganv2.Generator(self.img_size, self.style_dim).to(self.device)
        self.mapping_network = starganv2.MappingNetwork(self.latent_dim, self.style_dim, self.num_domains).to(self.device)
        self.style_encoder = starganv2.StyleEncoder(self.img_size, self.style_dim, self.num_domains).to(self.device)
        self.discriminator = starganv2.Discriminator(self.img_size, self.num_domains).to(self.device)

        self.gen_optim = optim.Adam(params=self.generator.parameters(), lr=self.lr, betas=[self.beta1,self.beta2], weight_decay=self.weight_decay)
        self.map_optim = optim.Adam(params=self.mapping_network.parameters(), lr=self.map_lr, betas=[self.beta1,self.beta2], weight_decay=self.weight_decay)
        self.style_optim = optim.Adam(params=self.style_encoder.parameters(), lr=self.lr, betas=[self.beta1,self.beta2], weight_decay=self.weight_decay)
        self.disc_optim = optim.Adam(params=self.discriminator.parameters(), lr=self.lr, betas=[self.beta1,self.beta2], weight_decay=self.weight_decay)

        train_dir = os.path.join(self.root_dir, "train")
        train_dataset = dataloader.CelebA(train_dir, self.config, transform=True)
        sampler = _make_balanced_sampler(train_dataset.src_labels)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, pin_memory=True, sampler=sampler)

        val_dir = os.path.join(self.root_dir, "val")
        val_dataset = dataloader.CelebA(val_dir, self.config, transform=True)
        sampler = _make_balanced_sampler(val_dataset.src_labels)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, pin_memory=True, sampler=sampler)

    def train(self):
        
        for indx, (src, src_label, ref1, ref2, ref_label, latent1, latent2) in enumerate(self.train_dataloader):
            d_loss_real, d_loss_fake, d_loss_reg, cycle_loss, div_loss, style_loss, adver_loss = 0,0,0,0,0,0,0

            src, src_label, ref1, ref2, ref_label, latent1, latent2 = src.to(self.device), src_label.to(self.device), ref1.to(self.device), ref2.to(self.device), ref_label.to(self.device), latent1.to(self.device), latent2.to(self.device)

            # training discriminator
            ###################################
            """
            (src, src_label) -> disc -> (fake) ---- cross_entropy (loss_real), r1_reg
            (ref-latent1, ref-label) -> mapping_net -> (style_ref1)+(src) -> generator -> (fake_ref) -> disc -> (fake) ---- cross_entropy (loss_fake)
            """
            src.requires_grad_()
            out = self.discriminator(src, src_label)
            loss_real = self.adversarial_loss(out, 1)
            loss_reg = self.r1_reg(out, src)

            with torch.no_grad():
                style_ref1 = self.mapping_network(latent1, ref_label)
                fake_img = self.generator(src, style_ref1)
            
            out = self.discriminator(fake_img, ref_label)
            loss_fake = self.adversarial_loss(out, 0)
            loss = loss_real + loss_fake + self.lambda_reg*loss_reg
            self._reset_grad()
            loss.backward()
            self.disc_optim.step()

            d_loss_real += loss_real.item()
            d_loss_fake += loss_fake.item()
            d_loss_reg += loss_reg.item()

            ####################################
            """
            (src, src_label) -> disc -> (fake) ---- cross_entropy (loss_real), r1_reg
            (ref1, ref-label) -> style_encoder -> (style_ref1)+(src) -> generator -> (fake_ref) -> disc -> (fake) ---- cross_entropy (loss_fake)
            """
            src.requires_grad_()
            out = self.discriminator(src, src_label)
            loss_real = self.adversarial_loss(out, 1)
            loss_reg = self.r1_reg(out, src)

            with torch.no_grad():
                style_ref1 = self.style_encoder(ref1, ref_label)
                fake_img = self.generator(src, style_ref1)
            
            out = self.discriminator(fake_img, ref_label)
            loss_fake = self.adversarial_loss(out, 0)
            loss = loss_real + loss_fake + self.lambda_reg*loss_reg
            self._reset_grad()
            loss.backward()
            self.disc_optim.step()

            d_loss_real += loss_real.item()
            d_loss_fake += loss_fake.item()
            d_loss_reg += loss_reg.item()

            # train generator
            #####################################
            """
            (latent1, ref_label) -> mapping_net -> (style_ref1)+(src) -> generator -> (fake_img1) -> discriminator -> (real) ---- adv_loss
            (fake_img, ref_label) -> style_encoder -> (pred_style_ref1) ---- style reconstruction loss
            (latent2, ref_label) -> mapping_net -> (style_ref2)+(src) -> generator -> (fake_img2) ---- loss diversity (to ensure proper exploration of image space)
            (src, src_label) -> style_encoder -> (style_src) + (fake_img1) -> generator --- cycle loss
            """
            style_ref1 = self.mapping_network(latent1, ref_label)
            fake_img1 = self.generator(src, style_ref1)
            out = self.discriminator(fake_img1, ref_label)
            loss_adv = self.adversarial_loss(out, 1)

            style_pred = self.style_encoder(fake_img1, ref_label)
            loss_style = torch.mean(torch.abs(style_pred-style_ref1))

            style_ref2 = self.mapping_network(latent2, ref_label)
            fake_img2 = self.generator(src, style_ref2)
            fake_img2.detach()
            loss_div = torch.mean(torch.abs(fake_img1 - fake_img2))

            style_src = self.style_encoder(src, src_label)
            src_recon = self.generator(fake_img1, style_src)
            loss_cyc = torch.mean(torch.abs(src_recon-src))

            loss = loss_adv + self.lambda_style*loss_style - self.lambda_div*loss_div + self.lambda_cyc*loss_cyc
            self._reset_grad()
            loss.backward()
            self.gen_optim.step()
            self.map_optim.step()
            self.style_optim.step()

            style_loss += loss_style.item()
            adver_loss += loss_adv.item()
            div_loss += loss_div.item()
            cycle_loss += loss_cyc.item()

            ######################################
            """
            (ref1, ref_label) -> style_encoder -> (style_ref1)+(src) -> generator -> (fake_img1) -> discriminator -> (real) ---- adv_loss
            (fake_img, ref_label) -> style_encoder -> (pred_style_ref1) ---- style reconstruction loss
            (ref2, ref_label) -> style_encoder -> (style_ref2)+(src) -> generator -> (fake_img2) ---- loss diversity (to ensure proper exploration of image space)
            (src, src_label) -> style_encoder -> (style_src) + (fake_img1) -> generator --- cycle loss
            """
            style_ref1 = self.style_encoder(ref1, ref_label)
            fake_img1 = self.generator(src, style_ref1)
            out = self.discriminator(fake_img1, ref_label)
            loss_adv = self.adversarial_loss(out, 1)

            style_pred = self.style_encoder(fake_img1, ref_label)
            loss_style = torch.mean(torch.abs(style_pred-style_ref1))

            style_ref2 = self.style_encoder(ref2, ref_label)
            fake_img2 = self.generator(src, style_ref2)
            fake_img2.detach()
            loss_div = torch.mean(torch.abs(fake_img1 - fake_img2))

            
            style_src = self.style_encoder(src, src_label)
            src_recon = self.generator(fake_img1, style_src)
            loss_cyc = torch.mean(torch.abs(src_recon-src))

            loss = loss_adv + self.lambda_style*loss_style - self.lambda_div*loss_div + self.lambda_cyc*loss_cyc
            self._reset_grad()
            loss.backward()
            self.gen_optim.step()
            self.map_optim.step()
            self.style_optim.step()
            
            style_loss += loss_style.item()
            adver_loss += loss_adv.item()
            div_loss += loss_div.item()
            cycle_loss += loss_cyc.item()
            
            # logging and other stuff!
            #########################################
            log = {
                "loss real": d_loss_real, 
                "loss fake": d_loss_fake, 
                "loss reg": d_loss_reg,
                "style loss": style_loss,
                "adv loss": adver_loss,
                "div loss": div_loss,
                "cyc loss": cycle_loss
                }
            wandb.log(log)
            self.lambda_div -= self.init_lambda_div/self.decay_div
            if self.step%self.valid_every==0:
                self.debug()
            self.step+=1
        
        torch.save({
            "gen": self.generator.state_dict(),
            "map": self.mapping_network.state_dict(),
            "style": self.style_encoder.state_dict(),
            "disc": self.discriminator.state_dict()
        }, os.path.join(self.output_dir, "weights", "{}.tar".format(self.step)))

    def run(self):
        while self.step <= self.max_iter:
            wandb.init(project="star-gan-v2")
            self.train()
    """
    saving image and logging via wandb
    """
    def _save_image(self, x, n_col, filename, name, caption):
        x = (x+1)/2
        x = x.clamp_(0,1)
        vutils.save_image(x.cpu(), filename, nrow=n_col, padding=0)
        wandb.log({name : [wandb.Image(filename, caption=caption)]})

    def _translate_and_rec(self, src, src_label, ref, ref_label):
        N, C, H, W = src.size()
        """
        (ref, ref_label) -> style_encoder -> (style_ref)+(src) -> generator -> (fake_ref)
        (src, src_label) -> style_encoder -> (style_src)+(fake_ref) -> generator -> (fake_src) reconstruction
        """
        style_ref = self.style_encoder(ref, ref_label)
        fake_ref = self.generator(src, style_ref)

        style_src = self.style_encoder(src, src_label)
        rec_src = self.generator(fake_ref, style_src)

        imgs = torch.cat([src.cpu(), ref.cpu(), fake_ref.cpu(), rec_src.cpu()], dim=0).cpu()
        self._save_image(imgs, N, os.path.join(self.output_dir, "tr_and_rec", "{}.png".format(self.step)), "translate and reconstruct", self.step)

    def _translate_with_ref(self, src, ref, ref_label):
        N, C, H, W = src.size()
        wb = torch.ones(1, C, H, W).to(self.device)
        src_with_wb = torch.cat([wb, src], dim=0).cpu() # white space

        """
        (ref, ref_label) -> style_encoder -> (style_refs)+(src) -> generator -> (generated_imgs)
        """
        style_refs = self.style_encoder(ref, ref_label)
        style_refs = style_refs.unsqueeze(1).repeat(1, N, 1)
        imgs = [src_with_wb]
        for i, style_ref in enumerate(style_refs):
            fake = self.generator(src, style_ref)
            fake_with_ref = torch.cat([ref[i:i+1], fake], dim=0)
            imgs += [fake_with_ref.cpu()]
        
        imgs = torch.cat(imgs, dim=0)
        self._save_image(imgs, N+1, os.path.join(self.output_dir, "tr_with_ref", "{}.png".format(self.step)), "translate with reference", self.step)

    def _translate_with_lat(self, src, ref_labels, latent_refs, psi):
        N, C, H, W = src.size()
        imgs = [src.cpu()]
        
        """
        (latent_ref, ref_label) -> mapping network -> (style_ref) + (src) -> generator
        """

        for i, ref_label in enumerate(ref_labels):
            latent_many = torch.randn(10000, self.latent_dim).to(self.device)
            labels_many = torch.LongTensor(10000).to(self.device).fill_(ref_label[0])
            style_many = self.mapping_network(latent_many, labels_many)
            style_avg = torch.mean(style_many, dim=0, keepdim=True)
            style_avg = style_avg.repeat(N, 1)
            # interpolate the latent spaces!
            for latent_ref in latent_refs:
                style_ref = self.mapping_network(latent_ref, ref_label)
                style_ref = torch.lerp(style_avg, style_ref, psi)
                imgs += [self.generator(src, style_ref).cpu()]
            
        imgs = torch.cat(imgs, dim=0)
        self._save_image(imgs, N, os.path.join(self.output_dir, "tr_with_lat", "{}_{}.png".format(self.step, psi)), "translate with latent {}".format(psi), self.step)

    def debug(self):
        try:
            srcs, src_labels, refs, _, ref_labels, _, _ = next(self.val_iter)
        except:
            self.val_iter = iter(self.val_dataloader)
            srcs, src_labels, refs, _, ref_labels, _, _ = next(self.val_iter)

        srcs, src_labels = srcs.to(self.device), src_labels.to(self.device)
        refs, ref_labels = refs.to(self.device), ref_labels.to(self.device)
        N = srcs.size(0)

        # translate and reconstruct
        self._translate_and_rec(srcs, src_labels, refs, ref_labels)

        # randomly choose reference labels and latent vectors
        random_ref_labels = [torch.tensor(y).repeat(N).to(self.device) for y in range(self.num_domains)]
        random_latent = torch.randn(self.num_domains, 1, self.latent_dim).repeat(1,N,1).to(self.device)
        for psi in [0.5, 0.7, 1.0]:
            # translate with different latent vectors
            self._translate_with_lat(srcs, random_ref_labels, random_latent, psi)
        
        self._translate_with_ref(srcs, refs, ref_labels)

    def _reset_grad(self):
        self.gen_optim.zero_grad()
        self.map_optim.zero_grad()
        self.style_optim.zero_grad()
        self.disc_optim.zero_grad()

    """
    normal cross entropy - used to train disrciminator to differentiate between real and fake images, 
    also for training generator in the adversarial loss
    """
    def adversarial_loss(self, logits, targets):
        targets = torch.full_like(logits, fill_value=targets)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        return loss
    
    """
    since we are always training the generator to predict the generated images are false, the training would never reach convergence.
    Hence to account for the fact that after some time the generator would generate real images, we penalise the model
    if large gradients are calculated when the image generated by the generator is real.
    """
    def r1_reg(self, logits, inp):
        batch_size = inp.size(0)
        grad_dout = torch.autograd.grad(outputs=logits.sum(), inputs=inp, create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_dout = grad_dout.pow(2)
        reg = 0.5*grad_dout.view(batch_size, -1).sum(1).mean(0)
        return reg

if __name__ == "__main__":
    import yaml
    config = yaml.safe_load(open("config/config.yaml", "r"))
    trainer = Trainer(config)
    trainer.run()



