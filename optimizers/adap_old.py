import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np


class AdaP3(Optimizer):
    """
    AdaP without projection, use Adam update for magnitude
    The main algorithm now testing
    """

    def __init__(
        self,
        params,
        lr=required,
        gamma=0.9,
        betas=(0.9, 0.999),
        weight_decay=0,
        eps=1e-8
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
        super(AdaP3, self).__init__(params, defaults)

        self.p1, self.p2 = 1, 1
        self.m, self.v = 0, 0
        self.first_step = 1
    
    def _init_pdir(self):
        '''
        Initialize the principal direction as a zero vector
        (Used to be a random unit vector, but changed to zero vector now)
        '''
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    pdir = torch.zeros_like(p.data)
                    state["pdir"] = pdir

    def __setstate__(self, state):
        super(AdaP3, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        # # How should we account for weight decay?
        # weight_decay = group["weight_decay"]
        # if weight_decay != 0:
        #     grad = grad.add(param, alpha=weight_decay)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.first_step:
            self._init_pdir()
            self.first_step = 0
        
        ## update principal direction
        # gradient norm
        lg = 0
        for group in self.param_groups:
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.add_(p, alpha=wd)  # l2 regularization changes the gradient
                    lg += torch.sum(p.grad**2)
        lg = torch.sqrt(lg)
                
        # accumulate gradient
        lp = 0
        for group in self.param_groups:
            gamma = group["gamma"]
            
            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    state["pdir"].add_(p.grad/lg, alpha=(1-gamma))  # note this is 1-gamma, consistent use with momentum
                    lp += torch.sum(state["pdir"]**2)
        lp = torch.sqrt(lp)
        
        ## Normalize principal direction and compute gradient component
        gp = 0
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            
            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    state["pdir"] /= lp
                    gp += torch.sum(state["pdir"]*p.grad)
        
        self.m = beta1 * self.m + (1-beta1) * gp  # betas are assumed identical for all groups
        self.v = beta2 * self.v + (1-beta2) * (gp**2)
        self.p1 *= beta1
        self.p2 *= beta2
        adam_coef = (self.m/(1-self.p1)) / (torch.sqrt(self.v/(1-self.p2))+eps)
        
        
        ## update the parameters and last gradient
        for group in self.param_groups:
            lr = group["lr"]
            
            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    p.add_(state["pdir"], alpha=-lr*adam_coef)

        return loss


class AdaP3W(Optimizer):
    """
    AdaP without projection, use Adam update for magnitude
    Use weight decay instead of l2 regularization
    """

    def __init__(
        self,
        params,
        lr=required,
        gamma=0.9,
        betas=(0.9, 0.999),
        weight_decay=0,
        eps=1e-8
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
        super(AdaP3W, self).__init__(params, defaults)

        self.p1, self.p2 = 1, 1
        self.m, self.v = 0, 0
        self.first_step = 1
    
    def _init_pdir(self):
        '''
        Initialize the principal direction as a zero vector
        (Used to be a random unit vector, but changed to zero vector now)
        '''
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    pdir = torch.zeros_like(p.data)
                    state["pdir"] = pdir

    def __setstate__(self, state):
        super(AdaP3W, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        # # How should we account for weight decay?
        # weight_decay = group["weight_decay"]
        # if weight_decay != 0:
        #     grad = grad.add(param, alpha=weight_decay)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.first_step:
            self._init_pdir()
            self.first_step = 0
        
        ## update principal direction
        # gradient norm
        lg = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    lg += torch.sum(p.grad**2)
        lg = torch.sqrt(lg)
                
        # accumulate gradient
        lp = 0
        for group in self.param_groups:
            gamma = group["gamma"]
            
            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    state["pdir"].add_(p.grad/lg, alpha=(1-gamma))  # note this is 1-gamma, consistent use with momentum
                    lp += torch.sum(state["pdir"]**2)
        lp = torch.sqrt(lp)
        
        ## Normalize principal direction and compute gradient component
        gp = 0
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            
            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    state["pdir"] /= lp
                    gp += torch.sum(state["pdir"]*p.grad)
        
        self.m = beta1 * self.m + (1-beta1) * gp  # betas are assumed identical for all groups
        self.v = beta2 * self.v + (1-beta2) * (gp**2)
        self.p1 *= beta1
        self.p2 *= beta2
        adam_coef = (self.m/(1-self.p1)) / (torch.sqrt(self.v/(1-self.p2))+eps)
        
        
        ## update the parameters and last gradient
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    p.add_(p, alpha=-lr*wd)
                    p.add_(state["pdir"], alpha=-lr*adam_coef)

        return loss
        
        
class AdaP(Optimizer):
    """
    Implement the Adam correction along a principal direction. This optimizer is used together with other base optimizers.
    """

    def __init__(
        self,
        params,
        lr=required,
        gamma=0.99,
        betas=(0.9, 0.999),
        weight_decay=0,
        eps=1e-8
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
        super(AdaP, self).__init__(params, defaults)

        self.p1, self.p2 = 1, 1
        self.m, self.v = 0, 0
        self.first_step = 1
    
    def _init_pdir(self):
        '''
        Initialize the principal direction as a random unit vector
        '''
        l = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    pdir = torch.zeros_like(p.data)
                    l += torch.sum(pdir**2)
                    state["pdir"] = pdir
        l = torch.sqrt(l)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    state["pdir"] = state["pdir"] / l
                
    def _init_buffer(self):
        '''
        Initialize spaces for the previous gradient
        '''
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    state["last_grad"] = torch.zeros_like(p.grad)

    def __setstate__(self, state):
        super(AdaP, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        # # How should we account for weight decay?
        # weight_decay = group["weight_decay"]
        # if weight_decay != 0:
        #     grad = grad.add(param, alpha=weight_decay)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.first_step:
            self._init_pdir()
            self._init_buffer()
            self.first_step = 0
        
        ## update principal direction
        # gradient norm
        lg = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    lg += torch.sum(p.grad**2)
        lg = torch.sqrt(lg)
                
        # accumulate gradient and compute projection coefficients
        nom, denom = 0, 0
        for group in self.param_groups:
            gamma = group["gamma"]
            
            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    state["pdir"].add_(p.grad/lg, alpha=(1-gamma))
                    gnoise = p.grad - state["last_grad"]
                    nom += torch.sum(gnoise * state["pdir"])
                    denom += torch.sum(gnoise**2)
                
        # compute the projection and principal direction length
        lp = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    pdir = state["pdir"].add(p.grad-state["last_grad"], alpha=-nom/denom)
                    # pdir = state["pdir"]
                    lp += torch.sum(pdir**2)
                    state["pdir"] = pdir
        lp = torch.sqrt(lp)
        
        
        ## Normalize principal direction and compute gradient component
        gp = 0
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            
            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    state["pdir"] /= lp
                    gp += torch.sum(state["pdir"]*p.grad)
        
        self.m = beta1 * self.m + (1-beta1) * gp  # betas are assumed identical for all groups
        self.v = beta2 * self.v + (1-beta2) * (gp**2)
        self.p1 *= beta1
        self.p2 *= beta2
        adam_coef = (self.m/(1-self.p1)) / (torch.sqrt(self.v/(1-self.p2))+eps)
        
        
        ## update the parameters and last gradient
        for group in self.param_groups:
            lr = group["lr"]
            
            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    p.add_(state["pdir"], alpha=-lr*adam_coef)
                    state["last_grad"] = torch.clone(p.grad)

        return loss


class AdaP1(Optimizer):
    """
    AdaP without projection, use SGD update for magnitude
    """

    def __init__(
        self,
        params,
        lr=required,
        gamma=0.9,
        betas=(0.9, 0.999),
        weight_decay=0,
        eps=1e-8
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
        super(AdaP1, self).__init__(params, defaults)

        self.first_step = 1
    
    def _init_pdir(self):
        '''
        Initialize the principal direction as a zero vector
        (Used to be a random unit vector, but changed to zero vector now)
        '''
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    pdir = torch.zeros_like(p.data)
                    state["pdir"] = pdir

    def __setstate__(self, state):
        super(AdaP1, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        # # How should we account for weight decay?
        # weight_decay = group["weight_decay"]
        # if weight_decay != 0:
        #     grad = grad.add(param, alpha=weight_decay)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.first_step:
            self._init_pdir()
            self.first_step = 0
        
        ## update principal direction
        # gradient norm
        lg = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    lg += torch.sum(p.grad**2)
        lg = torch.sqrt(lg)
                
        # accumulate gradient
        lp = 0
        for group in self.param_groups:
            gamma = group["gamma"]
            
            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    state["pdir"].add_(p.grad/lg, alpha=(1-gamma))  # note this is 1-gamma, consistent use with momentum
                    lp += torch.sum(state["pdir"]**2)
        lp = torch.sqrt(lp)
        
        ## Normalize principal direction and compute gradient component
        gp = 0
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            
            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    state["pdir"] /= lp
                    gp += torch.sum(state["pdir"]*p.grad)
        
        ## update the parameters and last gradient
        for group in self.param_groups:
            lr = group["lr"]
            
            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    p.add_(state["pdir"], alpha=-lr*gp)

        return loss


class AdaP2(Optimizer):
    """
    AdaP without projection, use momentum update for magnitude
    """

    def __init__(
        self,
        params,
        lr=required,
        gamma=0.9,
        betas=(0.9, 0.999),
        weight_decay=0,
        eps=1e-8
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
        super(AdaP2, self).__init__(params, defaults)

        self.m = 0
        self.first_step = 1
    
    def _init_pdir(self):
        '''
        Initialize the principal direction as a zero vector
        (Used to be a random unit vector, but changed to zero vector now)
        '''
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    pdir = torch.zeros_like(p.data)
                    state["pdir"] = pdir

    def __setstate__(self, state):
        super(AdaP2, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        # # How should we account for weight decay?
        # weight_decay = group["weight_decay"]
        # if weight_decay != 0:
        #     grad = grad.add(param, alpha=weight_decay)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.first_step:
            self._init_pdir()
            self.first_step = 0
        
        ## update principal direction
        # gradient norm
        lg = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    lg += torch.sum(p.grad**2)
        lg = torch.sqrt(lg)
                
        # accumulate gradient
        lp = 0
        for group in self.param_groups:
            gamma = group["gamma"]
            
            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    state["pdir"].add_(p.grad/lg, alpha=(1-gamma))  # note this is 1-gamma, consistent use with momentum
                    lp += torch.sum(state["pdir"]**2)
        lp = torch.sqrt(lp)
        
        ## Normalize principal direction and compute gradient component
        gp = 0
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            
            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    state["pdir"] /= lp
                    gp += torch.sum(state["pdir"]*p.grad)
        
        self.m = beta1 * self.m + gp  # first order momentum
        
        ## update the parameters and last gradient
        for group in self.param_groups:
            lr = group["lr"]
            
            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    p.add_(state["pdir"], alpha=-lr*self.m)

        return loss

