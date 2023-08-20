import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np


class LAdaP(Optimizer):
    """
    LAdaP: layerwise AdaP
    decouple_wd: False/True, use weight decay or l2 regularization
    """

    def __init__(
        self,
        params,
        lr=required,
        gamma=0.9,
        betas=(0.9, 0.999),
        weight_decay=0,
        eps=1e-8,
        decouple_wd=False
    ):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= gamma < 1.0:
           raise ValueError("Invalid gamma parameter: {}".format(gamma))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, gamma=gamma, eps=eps, weight_decay=weight_decay)
        super(LAdaP, self).__init__(params, defaults)

        self.p1, self.p2 = 1, 1
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
                    state["m"] = 0
                    state["v"] = 0
                    state["p1"] = 0
                    state["p2"] = 0


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
        
                
        for group in self.param_groups:
            gamma = group["gamma"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            wd = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    
                    # (1) Accumulate gradient and compute gradient component
                    if not self.decouple_wd:
                        p.grad.add_(p.data, alpha=wd)
                    state["pdir"] = gamma * state["pdir"] + p.grad
                    pdir = state["pdir"] / torch.norm(state["pdir"])
                    gp = torch.sum(pdir*p.grad)
                    

                    # (2) Compute Adam coefficient (betas are assumed identical for all groups)
                    state["m"] = beta1 * state["m"] + (1-beta1) * gp
                    state["v"] = beta2 * state["v"] + (1-beta2) * (gp**2)
                    state["p1"] *= beta1
                    state["p2"] *= beta2
                    adam_coef = ((state["m"]/(1-state["p1"])) /
                                (torch.sqrt(state["v"]/(1-state["p2"]))+eps))

                    ## (3) Update the parameters
                    if self.decouple_wd:
                        p.data.add_(p.data, alpha=-lr*wd)
                    p.data.add_(pdir, alpha=-lr*adam_coef)
                    

        return loss


