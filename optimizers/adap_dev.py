import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np


class AdaP_dev(Optimizer):
    """
    AdaP: latest version, no projection, no normalization
    decouple_wd: False/True, use weight decay or l2 regularization
    """

    def __init__(
        self,
        params,
        lr=required,
        lr_adap=required,
        momentum=0.9,
        gamma=0.9,
        betas=(0.9, 0.999),
        weight_decay=0,
        eps=1e-8,
        decouple_wd=False
    ):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if lr_adap is not required and lr_adap < 0.0:
            raise ValueError("Invalid learning rate for adap: {}".format(lr_adap))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= momentum < 1.0:
           raise ValueError("Invalid moomentum parameter: {}".format(momentum))
        if not 0.0 <= gamma < 1.0:
           raise ValueError("Invalid gamma parameter: {}".format(gamma))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, lr_adap=lr_adap, betas=betas, momentum=momentum, gamma=gamma, eps=eps, weight_decay=weight_decay)
        super(AdaP_dev, self).__init__(params, defaults)

        self.p1, self.p2 = 1, 1
        self.m, self.v = 0, 0
        self.initialized = False
        self.decouple_wd = decouple_wd
    
    def _init_pdir(self):
        '''
        Initialize the principal direction as a zero vector
        (Used to be a random unit vector, but changed to zero vector now)
        '''
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    state["pdir"] = torch.zeros_like(p.data)
                    state["mom"] = torch.zeros_like(p.data)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if not self.initialized:
            self._init_pdir()
            self.initialized = True
        
                
        # (1) Accumulate gradient and compute gradient component
        lp = 0
        gp = 0
        for group in self.param_groups:
            momentum = group["momentum"]
            gamma = group["gamma"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    if not self.decouple_wd:
                        p.grad.add_(p.data, alpha=wd)
                    state["pdir"] = gamma * state["pdir"] + p.grad
                    state["mom"] = momentum * state["mom"] + p.grad
                    lp += torch.sum(state["pdir"]**2)
                    gp += torch.sum(state["pdir"]*p.grad)
        lp = torch.sqrt(lp)
        gp /= lp

        
        # (2) Compute Adam coefficient (betas are assumed identical for all groups)
        self.m = beta1 * self.m + (1-beta1) * gp 
        self.v = beta2 * self.v + (1-beta2) * (gp**2)
        self.p1 *= beta1
        self.p2 *= beta2
        adam_coef = (self.m/(1-self.p1)) / (torch.sqrt(self.v/(1-self.p2))+eps)
        
        
        ## (3) Update the parameters
        for group in self.param_groups:
            lr = group["lr"]
            lr_adap = group["lr_adap"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    if self.decouple_wd:
                        p.data.add_(p.data, alpha=-lr*wd)
                    p.data.add_(state["pdir"], alpha=-lr_adap*adam_coef/lp)
                    p.data.add_(state["mom"], alpha=-lr)

        return loss

