import torch
import models
import train
import argparse
import numpy as np
import plot as plotter
from sampler import MonteCarloSampler


class MagnetSampler(MonteCarloSampler):

    def __init__(self, n, beta=1.0, D=1.0, k=0., k_m=0., h=1., step_size=5e-2):
        super(MagnetSampler, self).__init__(
            dim=n * n, beta=beta, k=k, step_size=step_size)
        self.n = n
        self.h = h
        self.D = D
        self.k_m = k_m
        self.x = torch.zeros(n, n)

    def discrete_laplacian(self, site, bc="dirichlet"):
        neigh_sum = 0.

        if bc == "dirichlet":
            if site[0] == 0:
                neigh_sum += -1. + self.x[site[0] + 1, site[1]]
            elif site[0] == self.n - 1:
                neigh_sum += self.x[site[0] - 1, site[1]] - 1.
            else:
                neigh_sum += self.x[site[0] - 1, site[1]] + \
                    self.x[site[0] + 1, site[1]]

            if site[1] == 0:
                neigh_sum += 1. + self.x[site[0], site[1] + 1]
            elif site[1] == self.n - 1:
                neigh_sum += self.x[site[0], site[1] - 1] + 1.
            else:
                neigh_sum += self.x[site[0], site[1] - 1] + \
                    self.x[site[0], site[1] + 1]

        neigh_sum -= 4 * self.x[tuple(site)]
        return neigh_sum / self.h**2

    def initialize(self, x):
        self.x = x

    def compute_m(self):
        return torch.mean(self.x)

    def sample_step(self, bias_force=None):
        self.x = self.x.view(self.n, self.n)
        randsite = torch.randint(self.n, [2])
        self.x[tuple(randsite)] += self.step_size * (self.x[tuple(randsite)] - self.x[tuple(randsite)]**3) +\
            self.step_size * self.D * self.discrete_laplacian(randsite)
        self.x[tuple(randsite)] += np.sqrt(self.step_size *
                                           self.beta) * np.random.randn()
        if bias_force is not None:
            self.x[tuple(randsite)] -= self.step_size * bias_force

        return self.x.view(-1), None

    def compute_bias_force_m(self, m):
        return (self.k_m * (self.compute_m() - m)).view(-1)[0]

    def compute_bias_force_u(self, q, u):
        return (self.k * (q(self.x.view(-1, self.dim)) - u)).view(-1)[0]

    def compute_bias(self, q, u):
        return (0.5 * self.k * (q(self.x.view(-1, self.dim)) - u)**2).view(-1)[0]

    def run_sweep(self, x, q=None, u=None, m=None, bias_q=False, bias_m=False):
        self.x = x
        bias_energy = None
        for i in range(self.dim):
            if bias_m:
                bias_force = self.compute_bias_force_m(m)
            elif bias_q:
                bias_force = self.compute_bias_force_u(q, u)
                bias_energy = self.compute_bias(q, u)
            else:
                bias_force = None
            self.sample_step(bias_force=bias_force)
        return self.x, self.energy, bias_energy

    def compute_grad_ql(self, n_steps, x, q, u, u_1):
        grad_ql = None
        dG = 0.
        cost_ql = 0.

        x = x.view(-1, self.n, self.n)

        for m in range(n_steps):
            with torch.no_grad():
                x, _, bias = self.run_sweep(x, q=q, u=u, bias_q=True)
                # compute the bias after the sweep for reweighting
                d_bias = bias - self.compute_bias(q, u_1)
                dG += torch.exp(-self.beta * d_bias)
                x = x.detach()
            x.requires_grad = True
            grad_qlx, cost_qlm = self.autograd_compute_cost_grad(
                x.view(-1, self.dim), q, u)
            if grad_ql is None:
                grad_ql = grad_qlx
            else:
                grad_ql = tuple(map(lambda s, t: s + t, grad_ql, grad_qlx))
            cost_ql += cost_qlm
        grad_ql = tuple(map(lambda s: s / n_steps, grad_ql))
        return grad_ql, dG / n_steps, x.detach().view(-1), cost_ql / n_steps


parser = argparse.ArgumentParser(
    description='Curie-Weiss Committor Example')
parser.add_argument('--n-init-data', type=int, default=100, metavar='N')
parser.add_argument('--step-size', type=float, default=5e-2, metavar='N')
parser.add_argument('--k', type=float, default=1.0, metavar='N')
parser.add_argument('--beta', type=float, default=0.1, metavar='N')
parser.add_argument('--n-optim-steps', type=int, default=100, metavar='N')
parser.add_argument('--n-windows', type=int, default=12, metavar='N')
parser.add_argument('--n-batch-steps', type=int, default=25, metavar='N')
parser.add_argument('--n-lattice', type=int, default=12, metavar='N')
parser.add_argument('--lr', type=float, default=1e-1, metavar='N')
parser.add_argument('--random-seed', type=int, default=0, metavar='N')

args = parser.parse_args()


nl = args.n_lattice
dim = nl**2
cw_system = MagnetSampler(args.n_lattice, beta=args.beta,
                          D=1.0, k=args.k, k_m=10.0, step_size=args.step_size)

# set up output files
logname = "cw_n={:03d}_batch={:03d}_seed={:03d}".format(
    nl, args.n_batch_steps, args.random_seed)

logfile = open(logname + ".log", "w")

torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)


# initialization

a_center = -0.5 * torch.ones(dim)
b_center = 0.5 * torch.ones(dim)

n_init_data = 100
init_data = torch.zeros(n_init_data, dim)
for i in range(n_init_data):
    init_data[i, :] = -0.5 + i / (n_init_data - 1) * torch.ones(dim)

ds = torch.sqrt(torch.sum((a_center - init_data)**2, dim=1)
                ).reshape(n_init_data, 1)
init_targets = ds / ds[-1]
n_init_data = args.n_init_data
init_data = torch.zeros(n_init_data, dim)
for d in range(dim):
    init_data[:, d] = torch.linspace(a_center[d], b_center[d], n_init_data)

dists = torch.sqrt(torch.sum((a_center - init_data) **
                             2, dim=1)).reshape(n_init_data, 1)
init_targets = dists / dists[-1]


q = models.CommittorNet(dim, 100, thresh=torch.sigmoid)
loss_vals = train.train_init(
    q, init_data, init_targets, n_init_train_steps=1)

"""
fig, ax = plotter.plot_committor(q, init_data)
fig.savefig("cw_init.pdf")
"""

print("Trained initial representation of the committor")

# sample points for boundary conditions
n_boundary_samples = 100
a_data, b_data = train.get_bc_magnet(
    cw_system, n_boundary_samples, a_center.reshape(nl, nl), b_center.reshape(nl, nl))
print("Generated samples for the boundary conditions")

# run the optimization
train.run_optimization(cw_system, q, args.n_optim_steps, args.n_windows,
                       args.n_batch_steps, args.lr,
                       bc="lagrange", a_data=a_data, b_data=b_data, logname=logname)
