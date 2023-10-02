""" Minimizers in FedSMOO implementation """

import random
import torch
from collections import defaultdict


class GSAM(torch.optim.Optimizer):
    def __init__(self, device, params, base_optimizer, rho=0.05,beta=1.0,gamma=1.0,adaptive=False,**kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        self.device = device
        self.beta = beta
        self.gamma = gamma
        self.max_norm = 10

        defaults = dict(rho=rho,adaptive=adaptive, **kwargs)
        super(GSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
            group["adaptive"] = adaptive
        self.paras = None

    @torch.no_grad()
    def first_step(self, mu_list, s_list, zero_grad=False):
        # s - (grad - mu)

        for group in self.param_groups:
            for p, mu, s in zip(group["params"], mu_list, s_list):
                if p.grad is None: continue
                p.grad.add_(mu.to(self.device)+s.to(self.device), alpha=-1)
                
        # norm & update mu & ascent step
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7) / self.beta
            for p, mu, s in zip(group["params"], mu_list, s_list):
                mu = mu.to(self.device)
                s = s.to(self.device)

                p.requires_grad = True 
                if p.grad is None: continue
                #original sam 
                # e_w = p.grad * scale.to(p)
                #asam 
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w * 1)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

                mu += (e_w - s) # update mu

        if zero_grad: self.zero_grad()

    '''
    @torch.no_grad()
    def first_half(self, zero_grad=False):
        #first order sum 
        for group in self.param_groups:
            for p in group["params"]:
                if self.state[p]:
                    p.add_(self.state[p]["e_w"]*0.90)  # climb to the local maximum "w + e(w)"
    '''


    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not self.state[p]: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"
                self.state[p]["e_w"] = 0

                if random.random() > self.beta:
                    p.requires_grad = False

        # self.base_optimizer.step()  # do the actual "sharpness-aware" update

        # if zero_grad: self.zero_grad()

    def step(self, mu, s):
        inputs,targets,loss_fct,model,defined_backward = self.paras
        assert defined_backward is not None, "Sharpness Aware Minimization requires defined_backward, but it was not provided"

        model.require_backward_grad_sync = False
        model.require_forward_param_sync = True


        logits = model(inputs)
        loss = loss_fct(logits,targets)

        loss = loss.mean()
        defined_backward(loss)
        # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.max_norm) # Clip gradients to prevent exploding

        #first step to w + e(w)
        self.first_step(mu, s, True)

        # second forward-backward step
        # self.first_half()

        model.require_backward_grad_sync = True
        model.require_forward_param_sync = False

        loss = loss_fct(model(inputs), targets)
        loss = loss.mean()
        defined_backward(loss)
        # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.max_norm) # Clip gradients to prevent exploding
        self.second_step(True)

        # self.returnthings = (predictions,return_loss)
        return mu 
 

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        #original sam 
                        # p.grad.norm(p=2).to(shared_device)
                        #asam 
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm