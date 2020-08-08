import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter #Class to log data.
from utils import linear_model, SSE

torch.manual_seed(0)
np.random.seed(0)


### Data ###

w_true = torch.tensor(np.array([3.,6.,9.]))       
b_true = torch.tensor([3.])                       
X_true = torch.tensor(np.linspace((0,1,2),(1,2,3),10))
Y_true = linear_model(X_true,w_true,b_true)

Y_obs = torch.add(Y_true, torch.randn(Y_true.shape))


### Model Parameters ###

w_hat = torch.randn(w_true.shape, dtype=torch.float64, requires_grad=True) 
b_hat = torch.randn(1, dtype=torch.float64, requires_grad=True)


### Hyperparamters ### 

alpha  = 0.0000001      # Learning rate.
n_iter = 2000         # Time steps (epochs).
optimizer = optim.Adam


### TensorBoard Writer Setup ###

# We tell Pytorch where to save a log of the trained weights and loss values.
log_name = f"{optimizer.__name__}_alpha={alpha}"
writer = SummaryWriter(log_dir=f"runs/{log_name}")
print(f"Learning rate = {alpha}, Optimizer = {optimizer.__name__}")


### Main Optimization Loop ###

optimizer = optim.SGD([w_hat, b_hat], lr=alpha) 

for t in range(n_iter):               
    optimizer.zero_grad()                                         # Set the gradients to zero.   
    current_loss = SSE(linear_model(X_true, w_hat, b_hat),Y_obs)  # For tracking the loss.
    current_loss.backward()                                       # Compute gradients of loss function (scalar-vector).
    optimizer.step()                                              # Update W_hat and b_hat.

    # Write the current values of the weights, and loss to the log.
    # global_step=t tells tensorboard at what step of the training this is.
    writer.add_scalar('bias', b_hat, global_step=t)
    writer.add_scalar('w_1', w_hat[0], global_step=t)
    writer.add_scalar('w_2', w_hat[1], global_step=t)
    writer.add_scalar('w_3', w_hat[2], global_step=t)
    writer.add_scalar('L', current_loss, global_step=t)

writer.close() # After we are done with the writer, we should close the log file.

print("\nTo see tensorboard run:\ntensorboard --logdir=runs/ or tensorboard --logdir=runs/ --host localhost --port 8088\n")
