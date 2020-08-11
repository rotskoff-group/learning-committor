import torch
import numpy as np


class MonteCarloSampler():
    def __init__(self, dim=0, beta=1.0, k=0., energy_fn=None, step_size=1e-2):
        self.dim = dim
        self.step_size = step_size
        self.x = torch.zeros(dim)
        self.beta = beta
        self.energy = 0.
        self.bias_energy = 0.
        self.k = k
        self.energy_fn = energy_fn
        self.n_per_sweep = 1

    def initialize(self, x):
        self.x = x.clone()
        self.energy = self.compute_energy(self.x)

    def compute_energy(self, x, args=[], kwargs={}):
        with torch.no_grad():
            return self.energy_fn(x, *args, **kwargs)

    def sample_step(self):
        with torch.no_grad():
            dx = self.step_size * torch.randn(self.x.shape)
            dE = self.compute_energy(self.x + dx) - self.energy
            if dE < 0.:
                self.x += dx
                self.energy += dE
                return self.x, self.energy
            elif np.random.rand() < np.exp(-self.beta * dE):
                self.x += dx
                self.energy += dE
                return self.x, self.energy
            else:
                return self.x, self.energy

    def compute_bias(self, x, q, u):
        with torch.no_grad():
            return 0.5 * self.k * torch.sum((q(x) - u)**2)

    def sample_biased(self, q, u):
        with torch.no_grad():
            dx = self.step_size * torch.randn(self.x.shape)
            dE = self.compute_energy(self.x + dx) - self.energy
            dbias = self.compute_bias(self.x + dx, q, u) - self.bias_energy
            dEtot = dE + dbias
            if dEtot < 0.:
                self.x += dx
                self.energy += dE
                self.bias_energy += dbias
                return self.x, self.energy, self.bias_energy
            elif np.random.rand() < np.exp(-self.beta * dEtot):
                self.x += dx
                self.energy += dE
                self.bias_energy += dbias
                return self.x, self.energy, self.bias_energy
            else:
                return self.x, self.energy, self.bias_energy

    def run_sweep(self, x, q=None, u=None, biased=True):
        self.x = x
        if biased:
            for i in range(self.n_per_sweep):
                self.sample_biased(q, u)
            return self.x, self.energy, self.bias_energy
        else:
            for i in range(self.n_per_sweep):
                self.sample_step()
            return self.x, self.energy, self.bias_energy

    @staticmethod
    def autograd_compute_cost_grad(x, q, u):
        grad_x = torch.autograd.grad(q(x), x, create_graph=True)[0]
        cost_integrand = torch.sum(grad_x * grad_x)
        return tuple([torch.autograd.grad(cost_integrand, q.parameters()), cost_integrand])

    def compute_grad_ql(self, n_steps, x, q, u, u_1):
        grad_ql = None
        dG = 0.
        cost_ql = 0.

        for m in range(n_steps):

            with torch.no_grad():
                x, _, bias = self.run_sweep(x, q=q, u=u, biased=True)
                # compute the bias after the sweep for reweighting
                d_bias = bias - self.compute_bias(x, q, u_1)
                dG += torch.exp(-self.beta * d_bias)
            x = x.detach()
            x.requires_grad = True
            grad_qlx, cost_qlm = self.autograd_compute_cost_grad(x, q, u)
            if grad_ql is None:
                grad_ql = grad_qlx
            else:
                grad_ql = tuple(map(lambda s, t: s + t, grad_ql, grad_qlx))
            cost_ql += cost_qlm
        grad_ql = tuple(map(lambda s: s / n_steps, grad_ql))
        return grad_ql, dG / n_steps, x.detach(), cost_ql / n_steps
