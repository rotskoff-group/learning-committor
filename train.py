import torch
import torch.optim as optim
import plot as plotter
import matplotlib.pyplot as plt


def train_init(q, data, targets, n_init_train_steps=1000, lr=0.5):
    optimizer = optim.SGD(q.parameters(), lr=lr)
    for i in range(n_init_train_steps):
        optimizer.zero_grad()
        q_vals = q(data)
        loss = torch.sum((targets - q_vals)**2) / targets.shape[0]
        loss.backward()
        optimizer.step()
        q.renormalize()
        if i % 5000 == 0:
            print(i, loss.data)


def get_bc(sampler, n_boundary_samples, a_center, b_center):
    with torch.no_grad():
        x_a = a_center.clone()
        a_data = torch.zeros(n_boundary_samples, sampler.dim)
        sampler.initialize(x_a)
        for i in range(n_boundary_samples):
            a_data[i, :], _ = sampler.sample_step()

        x_b = b_center.clone()
        b_data = torch.zeros(n_boundary_samples, sampler.dim)
        sampler.initialize(x_b)
        for i in range(n_boundary_samples):
            b_data[i, :], _ = sampler.sample_step()

        return a_data, b_data


def get_bc_magnet(sampler, n_boundary_samples, a_center, b_center,
                  a_bias=-0.5, b_bias=0.5, n_equil=100):
    with torch.no_grad():
        x_a = a_center.clone()
        a_data = torch.zeros(n_boundary_samples, sampler.dim)
        for i in range(n_equil):
            xa, _, _ = sampler.run_sweep(x_a, m=a_bias, bias_m=True)
        for i in range(n_boundary_samples):
            xa, _, _ = sampler.run_sweep(x_a, m=a_bias, bias_m=True)
        for i in range(n_boundary_samples):
            a_data[i, :] = x_a.view(sampler.dim)

        x_b = b_center.clone()
        b_data = torch.zeros(n_boundary_samples, sampler.dim)
        for i in range(n_equil):
            x_b, _, _ = sampler.run_sweep(x_b, m=b_bias, bias_m=True)
        for i in range(n_boundary_samples):
            x_b, _, _ = sampler.run_sweep(x_b, m=b_bias, bias_m=True)
            b_data[i, :] = x_b.reshape(sampler.dim)
        return a_data, b_data


def compute_grad_q(sampler, q, n_windows, n_sampling_steps, init_confs,
                   bc="lagrange", a_data=None, b_data=None, lambda_bc=25.):
    uls = torch.linspace(0, 1, n_windows)
    grad_q = None
    cost_q = 0.
    dG = torch.zeros(n_windows)
    for l in range(n_windows):
        x_l = init_confs[l, :]
        grad_ql, dG[l], init_confs[l, :], cost_ql = sampler.compute_grad_ql(
            n_sampling_steps, x_l, q, uls[l], uls[max(l - 1, 0)])
        dGl = torch.prod(dG[:l])
        cost_q += dGl * cost_ql
        if grad_q is None:
            grad_q = tuple([grad_qli * dGl for grad_qli in grad_ql])
        else:
            grad_q = tuple(map(lambda s, t: s + t * dGl, grad_q, grad_ql))

    grad_q = tuple([grad_qi / n_windows for grad_qi in grad_q])

    # compute the contribution from the boundary conditions
    if bc == "lagrange":
        grad_q_abc = torch.autograd.grad(
            lambda_bc * torch.mean(q(a_data) * q(a_data)), q.parameters())
        grad_q_bbc = torch.autograd.grad(
            lambda_bc * torch.mean((1 - q(b_data))**2), q.parameters())
        grad_q = tuple(map(lambda s, t, w: s + t + w,
                           grad_q, grad_q_abc, grad_q_bbc))
    elif bc == "dirichlet":
        grad_q_bc = torch.autograd.grad(
            lambda_bc * torch.mean(q(a_data) - q(b_data)), q.parameters())
        grad_q = tuple(map(lambda s, t: s + t, grad_q, grad_q_bc))
    else:
        return NotImplemented

    return grad_q, cost_q


def run_optimization(sampler, q, n_optim_steps, n_windows, n_sampling_steps,
                     lr, bc="lagrange", a_data=None, b_data=None, lambda_bc=25.,
                     logname=None, logging=True, plotting=True, verbose=True):

    init_confs = torch.zeros(n_windows, sampler.dim)
    for i in range(n_windows):
        init_confs[i, :] = -0.5 + i / (n_windows - 1) * torch.ones(sampler.dim)

    if logging:
        logfile = open(logname + ".log", "w")

    for i in range(n_optim_steps):
        grad_q, cost_q = compute_grad_q(
            sampler, q, n_windows, n_sampling_steps, init_confs, bc=bc,
            a_data=a_data, b_data=b_data, lambda_bc=lambda_bc)
        with torch.no_grad():
            params = list(q.parameters())
            for j in range(len(params)):
                params[j] -= grad_q[j] * lr

        if verbose:
            print("Optimization step {:d}, loss: {:f}".format(i, cost_q))

        if logging:
            logfile.write("{:f}\n".format(cost_q.data))
            #torch.save(q.state_dict(), logname + "_{:03d}.ckpt".format(i))

        if plotting:
            fig, ax = plotter.plot_committor(q, init_confs)
            fig.savefig(logname + "_q_{:03d}.pdf".format(i))
            plt.close()
            # plot the conf array
            fig = plotter.plot_conf_path(init_confs)
            fig.savefig(logname + "_confs_{:03d}.pdf".format(i))
            plt.close()
