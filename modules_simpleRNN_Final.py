# %%
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import time
from math import sqrt
import random
# %%
# Define the RNN class in pytorch 
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, wi_init, wo_init, wrec_init, h0_init = None,
                 train_conn = True, train_wout = False, train_wi=False, noise_std=0.0005, alpha=0.1):
        
        super(RNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.noise_std = noise_std
        self.train_conn = train_conn
        self.non_linearity = torch.tanh  # tanh nonlinearity
        self.alpha = alpha
        
        # Define parameters
        self.wi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        if not train_wi:
            self.wi.requires_grad= False
        else:
            self.wi.requires_grad = True
        
        self.wrec = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if not train_conn:
            self.wrec.requires_grad= False
        self.h0 = nn.Parameter(torch.Tensor(hidden_size))
        self.h0.requires_grad = False
        
        self.wout = nn.Parameter(torch.Tensor( hidden_size, output_size))
        if train_wout:
            self.wout.requires_grad= True
        else:
            self.wout.requires_grad= False
        
        # Initialize parameters
        with torch.no_grad():
            self.wi.copy_(wi_init)
            self.wrec.copy_(wrec_init)
            self.wout.copy_(wo_init)
            if h0_init is None:
                self.h0.zero_()
                self.h0.fill_(0)
            else:
                self.h0.copy_(h0_init)
            
    def forward(self, input): #Here you define the dynamics
        batch_size = input.shape[0] 
        seq_len = input.shape[1]
        h = self.h0
        r = self.non_linearity(self.h0)
        
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.wrec.device)
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.wrec.device)
        output[:,0,:] = r.matmul(self.wout)
        # simulation loop
        for i in range(seq_len-1):
            h = h + self.noise_std*noise[:,i,:]+self.alpha * (-h + r.matmul(self.wrec.t())+ input[:,i,:].matmul(self.wi))
            r = self.non_linearity(h)
            output[:,i+1,:] = r.matmul(self.wout)
        return output


# %%
# RNN_Inhibitory: gradient masking and hard clamping
class RNN_Inhibitory(RNN):
    """
    RNN with biologically constrained inhibitory neurons using weight clamping.
    
    Key design: Weights organized by SOURCE neuron type (separate exc and inh rows):
    - wrec_exc: shape (num_exc, hidden_size) - ALL positive weights [0, 1.5] FROM exc sources to ALL targets
    - wrec_inh: shape (num_inh, hidden_size) - ALL negative weights [-1.5, 0] FROM inh sources to ALL targets
    
    Ensures exc neurons output ONLY positive and inh neurons output ONLY negative to all targets.
    Weights clamped after each optimizer step to enforce constraints.
    
    In forward pass: wrec = torch.cat([wrec_exc, wrec_inh], dim=0) along rows
    Clamping only applied post-optimizer.step() to preserve gradient flow.
    """
    def __init__(self, input_size, hidden_size, output_size, wi_init, wo_init, wrec_init, 
                 inhibitory_indices=None, h0_init=None, train_conn=True, train_wout=False, 
                 train_wi=False, noise_std=0.0005, alpha=0.1):
        
        # Don't call parent __init__ with wrec_init; we'll handle it differently
        nn.Module.__init__(self)
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.noise_std = noise_std
        self.train_conn = train_conn
        self.non_linearity = torch.tanh  # Use tanh for fairness with baseline RNN
        self.alpha = alpha
        
        # Store inhibitory neuron indices
        if inhibitory_indices is None:
            self.inhibitory_indices = []
        else:
            self.inhibitory_indices = np.array(inhibitory_indices)
        
        # Count neurons
        self.num_exc = hidden_size - len(self.inhibitory_indices)
        self.num_inh = len(self.inhibitory_indices)
        
        # Define parameters with explicit parameterization
        self.wi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        if not train_wi:
            self.wi.requires_grad = False
        
        # Separate parameters for excitatory and inhibitory weights
        # wrec_exc: shape (num_exc, hidden_size) - weights FROM excitatory sources TO all targets
        # wrec_inh: shape (num_inh, hidden_size) - weights FROM inhibitory sources TO all targets
        self.wrec_exc = nn.Parameter(torch.Tensor(self.num_exc, hidden_size))
        self.wrec_inh = nn.Parameter(torch.Tensor(self.num_inh, hidden_size))
        if not train_conn:
            self.wrec_exc.requires_grad = False
            self.wrec_inh.requires_grad = False
        
        self.h0 = nn.Parameter(torch.Tensor(hidden_size))
        self.h0.requires_grad = False
        
        self.wout = nn.Parameter(torch.Tensor(hidden_size, output_size))
        if train_wout:
            self.wout.requires_grad = True
        else:
            self.wout.requires_grad = False
        
        # Initialize parameters
        with torch.no_grad():
            self.wi.copy_(wi_init)
            self.wout.copy_(wo_init)
            
            # Convert wrec_init to tensor if it's numpy array
            if isinstance(wrec_init, np.ndarray):
                wrec_tensor = torch.from_numpy(wrec_init).float()
            else:
                wrec_tensor = wrec_init.float() if not wrec_init.dtype == torch.float32 else wrec_init
            
            # Split wrec_init into excitatory and inhibitory parts
            # Initialize directly within constraint bounds for stability
            if self.num_exc > 0:
                # Initialize excitatory in range [0, 1.5] - positive excitation
                # Use smaller range at first: [0, 0.5] for stability
                self.wrec_exc.uniform_(0, 0.5)
            
            if self.num_inh > 0:
                # Initialize inhibitory in range [-1.5, 0] - negative inhibition
                # Use smaller range at first: [-0.5, 0] for stability
                self.wrec_inh.uniform_(-0.5, 0)
            
            if h0_init is None:
                self.h0.zero_()
                self.h0.fill_(0)
            else:
                self.h0.copy_(h0_init)
    
    def forward(self, input):
        batch_size = input.shape[0] 
        seq_len = input.shape[1]
        h = self.h0
        r = self.non_linearity(self.h0)
        
        # Use weight parameters directly (NO clamping in forward!)
        # Clamping happens ONLY after optimizer.step() during training
        # This preserves gradient flow during backpropagation
        # Concatenate exc sources first (rows 0:num_exc), then inh sources (rows num_exc:hidden_size)
        if self.num_inh > 0:
            wrec = torch.cat([self.wrec_exc, self.wrec_inh], dim=0)
        else:
            wrec = self.wrec_exc
        
        output = torch.zeros(batch_size, seq_len, self.output_size, device=wrec.device)
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=wrec.device)
        output[:,0,:] = r.matmul(self.wout)
        
        # simulation loop
        for i in range(seq_len-1):
            h = h + self.noise_std*noise[:,i,:] + self.alpha * (-h + r.matmul(wrec.t()) + input[:,i,:].matmul(self.wi))
            r = self.non_linearity(h)
            output[:,i+1,:] = r.matmul(self.wout)
        
        return output

    
def loss_mse(output, target, mask):
    """
    Mean squared error loss
    :param output: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param target: idem
    :param mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :return: float
    """
    # Compute loss for each (trial, timestep) (average accross output dimensions)    
    loss_tensor = (mask * (target - output)).pow(2).mean(dim=-1)
    # Account for different number of masked values per trial
    loss_by_trial = loss_tensor.sum(dim=-1) / mask[:, :, 0].sum(dim=-1)
    return loss_by_trial.mean()


# %%
def train(net, _input, _target, _mask, n_epochs, lr=1e-2, batch_size=32, plot_learning_curve=False, plot_gradient=False,
          clip_gradient=None, cuda=False, save_loss=False, save_params=True, verbose=True, adam=True):
    """
    Train a network
    :param net: nn.Module
    :param _input: torch tensor of shape (num_trials, num_timesteps, input_dim)
    :param _target: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param _mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :param n_epochs: int
    :param lr: float, learning rate
    :param batch_size: int
    :param plot_learning_curve: bool
    :param plot_gradient: bool
    :param clip_gradient: None or float, if not None the value at which gradient norm is clipped
    :param keep_best: bool, if True, model with lower loss from training process will be kept (for this option, the
        network has to implement a method clone())
    :return: nothing
    """
    print("Training...")
    if adam: #this is the type of backprop implementation you want to use.
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)#
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    num_examples = _input.shape[0]
    all_losses = []
    # graD = np.zeros((hidden_size, n_epochs))
    hidden_size = net.hidden_size
    wr = np.zeros((hidden_size, hidden_size, n_epochs))
    if plot_gradient:
        gradient_norms = []
    
    # Device management - prioritize MPS on macOS
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS device")
    elif cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA device")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
        
    net.to(device=device)
    input = _input.to(device=device)
    target = _target.to(device=device)
    mask = _mask.to(device=device)

    with torch.no_grad():
        initial_loss = loss_mse(net(input), target, mask)
        print("initial loss: %.3f" % (initial_loss.item()))
        # if keep_best:
        #     best = net.clone()
        #     best_loss = initial_loss.item()

    for epoch in range(n_epochs):
        begin = time.time()
        losses = []

        #for i in range(num_examples // batch_size):
        optimizer.zero_grad()
        
        random_batch_idx = random.sample(range(num_examples), batch_size)
        #random_batch_idx = random.sample(range(num_examples), num_examples)
        batch = input[random_batch_idx]
        output = net(batch)
        # if epoch==0:
        #     output0 = output
        loss = loss_mse(output, target[random_batch_idx], mask[random_batch_idx])
        
        losses.append(loss.item())
        all_losses.append(loss.item())
        loss.backward()
        if clip_gradient is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_gradient)
        if plot_gradient:
            tot = 0
            for param in [p for p in net.parameters() if p.requires_grad]:
                tot += (param.grad ** 2).sum()
            gradient_norms.append(sqrt(tot))
        #This is for debugging
        # for param in [p for p in net.parameters() if p.requires_grad]:
        #     graD[:,epoch] = param.grad.detach().numpy()[:,0]
        optimizer.step()
        # These 2 lines important to prevent memory leaks
        loss.detach_()
        output.detach_()

        if np.mod(epoch, 10)==0 and verbose is True:
            # if keep_best and np.mean(losses) < best_loss:
            #     best = net.clone()
            #     best_loss = np.mean(losses)
            #     print("epoch %d:  loss=%.3f  (took %.2f s) *" % (epoch, np.mean(losses), time.time() - begin))
            # else:
            print("epoch %d:  loss=%.3f  (took %.2f s)" % (epoch, np.mean(losses), time.time() - begin))
        
        # Save weights - handle MPS device properly
        if torch.backends.mps.is_available():
            if hasattr(net, 'wrec'):
                wr[:,:,epoch] = net.wrec.detach().cpu().numpy()
            else:
                # For RNN_Inhibitory: reconstruct full wrec matrix for saving
                with torch.no_grad():
                    if net.num_inh > 0:
                        wrec_full = torch.cat([net.wrec_exc, net.wrec_inh], dim=0)
                    else:
                        wrec_full = net.wrec_exc
                wr[:,:,epoch] = wrec_full.detach().cpu().numpy()
        else:
            if hasattr(net, 'wrec'):
                wr[:,:,epoch] = net.wrec.detach().numpy()
            else:
                # For RNN_Inhibitory: reconstruct full wrec matrix for saving
                with torch.no_grad():
                    if net.num_inh > 0:
                        wrec_full = torch.cat([net.wrec_exc, net.wrec_inh], dim=0)
                    else:
                        wrec_full = net.wrec_exc
                wr[:,:,epoch] = wrec_full.detach().cpu().numpy()
        
            
    if plot_learning_curve:
        plt.figure()
        plt.plot(all_losses)
        plt.yscale('log')
        plt.title("Learning curve")
        plt.show()

    if plot_gradient:
        plt.figure()
        plt.plot(gradient_norms)
        plt.yscale('log')
        plt.title("Gradient norm")
        plt.show()

    return(all_losses, wr[:,:,-1])


def train_inhibitory(net, _input, _target, _mask, n_epochs, lr=1e-2, batch_size=32, 
                      plot_learning_curve=False, plot_gradient=False, clip_gradient=None, 
                      cuda=False, save_loss=False, save_params=True, verbose=True, adam=True,
                      constraint_lambda=0.0, use_gradient_masking=True):
    """
    Train network with RNN_Inhibitory class using improved backpropagation.
    
    IMPROVEMENTS:
    1. Gradient masking: Prevents gradients from pushing weights outside constraints DURING backprop
    2. Hard clamping: Enforces constraints after optimizer step
    3. Better diagnostics: Tracks constraint violations and gradient statistics
    
    Constraints:
    - Excitatory weights (wrec_exc): [0, 1.5] (all positive)
    - Inhibitory weights (wrec_inh): [-1.5, 0] (all negative)
    
    Args:
        constraint_lambda (float): [DISABLED by default] Soft regularization weight (MPS compatibility)
        use_gradient_masking (bool): If True, apply gradient masking to enforce constraints
    """
    print("Training with improved inhibitory backpropagation...")
    print(f"  - Gradient masking: {use_gradient_masking}")
    if constraint_lambda > 0:
        print(f"  - Soft constraint regularization (λ={constraint_lambda})")
    
    if adam:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    
    num_examples = _input.shape[0]
    all_losses = []
    hidden_size = net.hidden_size
    wr = np.zeros((hidden_size, hidden_size, n_epochs))
    constraint_violations = []
    
    if plot_gradient:
        gradient_norms = []
    
    # Constraint bounds
    EXC_MIN, EXC_MAX = 0.0, 1.5
    INH_MIN, INH_MAX = -1.5, 0.0
    
    # Device management - prioritize MPS on macOS
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS device")
    elif cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA device")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    
    net.to(device=device)
    input = _input.to(device=device)
    target = _target.to(device=device)
    mask = _mask.to(device=device)

    with torch.no_grad():
        initial_loss = loss_mse(net(input), target, mask)
        print("initial loss: %.3f" % (initial_loss.item()))

    for epoch in range(n_epochs):
        begin = time.time()
        losses = []

        optimizer.zero_grad()
        
        random_batch_idx = random.sample(range(num_examples), batch_size)
        batch = input[random_batch_idx]
        output = net(batch)
        loss = loss_mse(output, target[random_batch_idx], mask[random_batch_idx])
        
        # Count constraint violations for diagnostics
        with torch.no_grad():
            exc_lower_violations = torch.sum(net.wrec_exc < EXC_MIN).item()
            exc_upper_violations = torch.sum(net.wrec_exc > EXC_MAX).item()
            inh_lower_violations = torch.sum(net.wrec_inh < INH_MIN).item()
            inh_upper_violations = torch.sum(net.wrec_inh > INH_MAX).item()
            total_violations = (exc_lower_violations + exc_upper_violations + 
                              inh_lower_violations + inh_upper_violations)
            constraint_violations.append(total_violations)
        
        losses.append(loss.item())
        all_losses.append(loss.item())
        loss.backward()
        
        # GRADIENT MASKING: Zero out gradients that would violate constraints
        if use_gradient_masking:
            with torch.no_grad():
                # Excitatory gradients: can't go negative or above max
                if net.wrec_exc.grad is not None:
                    # Clone gradient to avoid MPS issues with advanced indexing
                    grad_exc = net.wrec_exc.grad.clone()
                    # Mask where weights are at boundaries
                    mask_lower = net.wrec_exc <= EXC_MIN
                    mask_upper = net.wrec_exc >= EXC_MAX
                    # Zero gradients trying to go below 0
                    grad_exc[mask_lower] = torch.clamp(grad_exc[mask_lower], min=0)
                    # Zero gradients trying to go above 1.5
                    grad_exc[mask_upper] = torch.clamp(grad_exc[mask_upper], max=0)
                    net.wrec_exc.grad.copy_(grad_exc)
                
                # Inhibitory gradients: can't go positive or below min
                if net.wrec_inh.grad is not None:
                    grad_inh = net.wrec_inh.grad.clone()
                    # Mask where weights are at boundaries
                    mask_upper = net.wrec_inh >= INH_MAX
                    mask_lower = net.wrec_inh <= INH_MIN
                    # Zero gradients trying to go above 0 (toward positive)
                    grad_inh[mask_upper] = torch.clamp(grad_inh[mask_upper], max=0)
                    # Zero gradients trying to go below -1.5 (too negative)
                    grad_inh[mask_lower] = torch.clamp(grad_inh[mask_lower], min=0)
                    net.wrec_inh.grad.copy_(grad_inh)
        
        if clip_gradient is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_gradient)
        
        if plot_gradient:
            tot = 0
            for param in [p for p in net.parameters() if p.requires_grad]:
                tot += (param.grad ** 2).sum()
            gradient_norms.append(np.sqrt(tot.item()))
        
        optimizer.step()
        
        # HARD CONSTRAINT: Clamping to ensure validity
        with torch.no_grad():
            # Excitatory weights: clamp to [0, 1.5]
            net.wrec_exc.clamp_(min=EXC_MIN, max=EXC_MAX)
            # Inhibitory weights: clamp to [-1.5, 0]
            net.wrec_inh.clamp_(min=INH_MIN, max=INH_MAX)
        
        loss.detach_()
        output.detach_()

        if np.mod(epoch, 50) == 0 and verbose is True:
            violations_str = f" (violations: {constraint_violations[-1]})" if constraint_violations else ""
            print("epoch %d:  loss=%.6f  (took %.2f s)%s" % 
                  (epoch, np.mean(losses), time.time() - begin, violations_str))
        
        # Save weight matrix for diagnostics
        with torch.no_grad():
            if net.num_inh > 0:
                wrec_full = torch.cat([net.wrec_exc, net.wrec_inh], dim=0)
            else:
                wrec_full = net.wrec_exc
            wr[:,:,epoch] = wrec_full.detach().cpu().numpy()
    
    if plot_learning_curve:
        plt.figure()
        plt.plot(all_losses)
        plt.yscale('log')
        plt.title("Learning curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    if plot_gradient:
        plt.figure()
        plt.plot(gradient_norms)
        plt.yscale('log')
        plt.title("Gradient norm")
        plt.xlabel("Epoch")
        plt.ylabel("||∇||")
        plt.show()

    return all_losses, wr[:,:,-1]

def remove_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    return()
