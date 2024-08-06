import chirho
import pyro.distributions as dist
import pyro
from typing import TypeVar
from collections import OrderedDict
import torch
import seaborn
import matplotlib.pyplot as plt
from pyro.infer.autoguide import AutoMultivariateNormal
from pyro.infer.predictive import Predictive
import warnings
warnings.simplefilter("ignore")
from demo_abm_scm import get_trajectory
# pyro.settings.set(module_local_params=True)
from functools import partial
from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.dynamical.handlers.trajectory import LogTrajectory
from chirho.dynamical.ops import simulate


def to_torch(d: dict) -> dict:
    return {k: torch.tensor(v) for k, v in d.items()}


Simulation = TypeVar('Simulation')


# def get_trajectory(parameters: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
#     return OrderedDict(
#         daily_infected=parameters["transmission_rate"] * torch.linspace(100, 1000, 10)[..., None]  # stand-in for ABM
#     )


def direct_transmission_rate_prior():
    expanded_dist = dist.Uniform(0.01, 20.0).expand((3, 1)).to_event(2)
    transmission_rate = pyro.sample("transmission_rate", expanded_dist)

    return OrderedDict(
        transmission_rate=transmission_rate
    )


def ode_prior():
    # Make a simple prior centered here: dict(c=1., k=3., lam=0.2)), where all are positive. Return an OrderedDicct.
    # c = pyro.sample("c", dist.Uniform(0.9, 1.1))
    # k = pyro.sample("k", dist.Uniform(2.9, 3.1))
    # lam = pyro.sample("lam", dist.Uniform(0.1, 0.3))

    c = pyro.sample("c", dist.Uniform(0.2, 2.))
    k = pyro.sample("k", dist.Uniform(1., 5.))
    lam = pyro.sample("lam", dist.Uniform(0.05, 0.5))

    return OrderedDict(
        c=c,
        k=k,
        lam=lam,
    )


# def augment_parameters_with_transmissability_progression_ode_solution(
#         parameters: OrderedDict[str, torch.Tensor],
#         times: torch.Tensor
# ):
#     def gamma_like_ode_pure(state, atemp_params):
#         y = state['y']
#         t = state['t']
#         c = atemp_params['c']
#         k = atemp_params['k']
#         lam = atemp_params['lam']
#         dydt = c * (t ** (k - 1)) * torch.exp(-lam * t) - y * t
#         return dict(y=dydt)

#     gamma_like_ode_closure = partial(gamma_like_ode_pure, atemp_params=parameters)

#     with LogTrajectory(times=times) as logging_trajectory:
#         with TorchDiffEq():
#             simulate(gamma_like_ode_closure, to_torch(dict(y=0.)), times[0], times[-1])

#     return OrderedDict(
#         transmission_rate=logging_trajectory.trajectory['y'].unsqueeze(-1)
#     )


def ab_model():

    # # ODE trajectory.
    parameters = ode_prior()
    # parameters = augment_parameters_with_transmissability_progression_ode_solution(
    #     parameters,
    #     times=torch.tensor([7.0, 14.0, 21.0])
    # )

    # Direct trajectory prior.
    # parameters = direct_transmission_rate_prior()

    # Stacking ode params into one vector.
    parameters = dict(ode_params=torch.stack([parameters["c"], parameters["k"], parameters["lam"]]))

    trajectory = get_trajectory(parameters)

    daily_infected = pyro.deterministic("daily_infected", trajectory["daily_infected"])
    observed_infected_count = pyro.sample(
        "observed_infected_count",
        dist.Poisson(daily_infected + 1).to_event(2),  # 1 or 2 conditional on time thing? see to_event conditional in DS tutorial
        # dist.Normal(daily_infected, 2.0).to_event(2),
    )

    return OrderedDict(
        **parameters,
        **trajectory,
        observed_infected_count=observed_infected_count
    )


def plot_predictive(model, guide=None, num_samples=100, show=True):
    # Get predictive
    predictive = Predictive(model, guide=guide, num_samples=num_samples)
    samples = predictive()

    for i in range(num_samples):
        seaborn.lineplot(samples["daily_infected"][i, :, 0].detach().numpy(), linewidth=0.1, color="black")
        seaborn.lineplot(samples["observed_infected_count"][i, :, 0].detach().numpy(), linewidth=0.1, color="gray")

    if show:
        plt.show()


def approximate_posterior(conditioned_model, lr, n) -> (AutoMultivariateNormal, list):
    # guide = pyro.infer.autoguide.AutoMultivariateNormal(conditioned_model)
    guide = pyro.infer.autoguide.AutoDelta(conditioned_model)
    elbo = pyro.infer.Trace_ELBO()(conditioned_model, guide)
    elbo()
    optim = torch.optim.Adam(elbo.parameters(), lr=lr)

    losses = []

    for i in range(n):
        # for param in elbo.parameters():
        #     param.grad = None
        optim.zero_grad()

        loss = elbo()
        loss.backward()

        optim.step()

        losses.append(loss.detach().clone().item())

        # Print progress with a carriage return, which requires leading spaces to ensure the
        #  printed string is the same length.
        print(f"\r{i}/{n} loss: {loss.detach().clone().item()}", end="")

    return guide, losses


def main():
    # plot_predictive(ab_model, num_samples=5)

    # # Sanity check of grad propagation.
    # sanity_parameters = OrderedDict(
    #     transmission_rate=torch.tensor([2.5, 2.5, 2.5], requires_grad=True)[:, None]
    # )
    # trajectory = get_trajectory(sanity_parameters)
    # print("Output has grad?:", trajectory["daily_infected"].requires_grad)

    # # Get ground truth.
    # true_parameters = OrderedDict(
    #     transmission_rate=torch.tensor([0.0, 0.0, 5.0])[:, None]
    # )
    true_parameters = OrderedDict(
        transmission_rate=torch.tensor([1.0, 1.0, 1.0])[:, None]
    )
    # Fix parameters to ground truth.
    true_model = pyro.condition(ab_model, data=true_parameters)

    # Plot samples from the true model.
    plot_predictive(true_model, num_samples=5)

    # return

    # Condition model on
    observed_infected_count = Predictive(true_model, num_samples=1)()["observed_infected_count"].squeeze(0).detach().clone()
    conditioned_model = pyro.condition(ab_model, data={"observed_infected_count": observed_infected_count})
    guide, losses = approximate_posterior(conditioned_model, 2e-3, n=1000)

    plt.plot(losses)
    # Also plot a running average of the losses.
    w = 20
    running_average = [sum(losses[(i-w):i])/w for i in range(w, len(losses))]
    plt.plot(range(w, len(losses)), running_average, color="red")
    plt.show()

    plot_predictive(ab_model, guide=guide, num_samples=5, show=False)

    # Plot observed_infected_count in orange.
    plt.plot(observed_infected_count[..., 0].detach().numpy(), color="cyan", linewidth=2.0)

    plt.show()


if __name__ == "__main__":
    main()
