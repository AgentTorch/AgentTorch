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


Simulation = TypeVar('Simulation')


# def get_trajectory(parameters: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
#     return OrderedDict(
#         daily_infected=parameters["transmission_rate"] * torch.linspace(100, 1000, 10)[..., None]  # stand-in for ABM
#     )


def prior():
    expanded_dist = dist.Uniform(2.0, 3.0).expand((3, 1)).to_event(2)
    transmission_rate = pyro.sample("transmission_rate", expanded_dist)

    return OrderedDict(
        transmission_rate=transmission_rate
    )


def ab_model():
    parameters = prior()
    trajectory = get_trajectory(parameters)

    daily_infected = pyro.deterministic("daily_infected", trajectory["daily_infected"])
    observed_infected_count = pyro.sample(
        "observed_infected_count",
        dist.Poisson(daily_infected).to_event(2),  # 1 or 2 conditional on time thing? see to_event conditional in DS tutorial
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
    guide = pyro.infer.autoguide.AutoMultivariateNormal(conditioned_model)
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

    return guide, losses


def main():
    plot_predictive(ab_model, num_samples=5)

    # Sanity check of grad propagation.
    sanity_parameters = OrderedDict(
        transmission_rate=torch.tensor([2.5, 2.5, 2.5], requires_grad=True)[:, None]
    )
    trajectory = get_trajectory(sanity_parameters)
    print("Output has grad?:", trajectory["daily_infected"].requires_grad)

    # Get ground truth.
    observed_infected_count = Predictive(ab_model, num_samples=1)()["observed_infected_count"].squeeze(0).detach().clone()
    conditioned_model = pyro.condition(ab_model, data={"observed_infected_count": observed_infected_count})

    guide, losses = approximate_posterior(conditioned_model, 1e-5, n=100)

    plt.plot(losses)
    plt.show()

    plot_predictive(ab_model, guide=guide, num_samples=100, show=False)

    # Plot observed_infected_count in orange.
    plt.plot(observed_infected_count[..., 0].detach().numpy(), color="cyan", linewidth=2.0)

    plt.show()


if __name__ == "__main__":
    main()
