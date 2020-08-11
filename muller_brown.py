import torch
import models
import train
import argparse
from sampler import MonteCarloSampler


# Set up the classic Muller-Brown potential

dim = 2

icov1 = torch.tensor([[-1., 0.], [0., -10.]])
icov2 = torch.tensor([[-1., 0.], [0., -10.]])
icov3 = torch.tensor([[-6.5, 5.5], [5.5, -6.5]])
icov4 = torch.tensor([[0.7, 0.3], [0.3, 0.7]])

mu1 = torch.tensor([[1., 0.]])
mu2 = torch.tensor([[0., 0.5]])
mu3 = torch.tensor([[-0.5, 1.5]])
mu4 = torch.tensor([[-1., 1.]])

A1 = -20.
A2 = -10.
A3 = -17
A4 = 1.5


def gau(x, mu, icov, amp):
    return amp * torch.exp(torch.matmul(x - mu, torch.matmul(icov, (x - mu).t())))


def V_mb(x):
    return gau(x, mu1, icov1, A1) + gau(x, mu2, icov2, A2) +\
        gau(x, mu3, icov3, A3) + gau(x, mu4, icov4, A4)


def dist(x, y):
    return torch.sqrt(torch.sum((x - y)**2))


# parse the arguments
parser = argparse.ArgumentParser(
    description='Low-dimensional Committor Example')
parser.add_argument('--n-init-data', type=int, default=100, metavar='N')
parser.add_argument('--step-size', type=float, default=1e-2, metavar='N')
parser.add_argument('--beta', type=float, default=1.0, metavar='N')
parser.add_argument('--n-optim-steps', type=int, default=100, metavar='N')
parser.add_argument('--n-windows', type=int, default=15, metavar='N')
parser.add_argument('--n-batch-steps', type=int, default=10, metavar='N')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N')

args = parser.parse_args()

logname = "mullerbrown_nw={:03d}".format(args.n_windows)

mb_system = MonteCarloSampler(
    dim=dim, beta=args.beta, energy_fn=V_mb, step_size=args.step_size)


a_center = torch.tensor([-0.5, 1.5])
b_center = torch.tensor([0.5, 0.])
a_cutoff = 0.2
b_cutoff = 0.2


# initialization
n_init_data = args.n_init_data
init_data = torch.zeros(n_init_data, dim)
for d in range(dim):
    init_data[:, d] = torch.linspace(a_center[d], b_center[d], n_init_data)

dists = torch.sqrt(torch.sum((a_center - init_data) **
                             2, dim=1)).reshape(n_init_data, 1)
init_targets = dists / dists[-1]

q = models.CommittorNet(2, 100, thresh=torch.sigmoid)
loss_vals = train.train_init(
    q, init_data, init_targets, n_init_train_steps=10000)

print("Trained initial representation of the committor")

# sample points for boundary conditions
n_boundary_samples = 100
a_data, b_data = train.get_bc(
    mb_system, n_boundary_samples, a_center, b_center)
print("Generated samples for the boundary conditions")

# run the optimization
train.run_optimization(mb_system, q, args.n_optim_steps, args.n_windows,
                       args.n_batch_steps, args.lr, bc="lagrange", a_data=a_data, b_data=b_data, logname=logname, plotting=False)
