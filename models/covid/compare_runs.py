# package imports
from epiweeks import Week
import matplotlib.pyplot as plt
import sys
import torch

# relative imports
from utils.data import get_labels, DATA_START_WEEK
from utils.feature import Feature
from utils.misc import week_num_to_epiweek, name_to_neighborhood, subtract_epiweek

# AgentTorch imports
AGENT_TORCH_PATH = "/u/ngkuru/ship/MacroEcon/AgentTorch"
sys.path.insert(0, AGENT_TORCH_PATH)
from AgentTorch.helpers import read_config
from simulator import get_registry, get_runner


def simulate_once(r0_values, config):
    # get runner from config
    registry = get_registry()
    runner = get_runner(config, registry)
    runner.init()

    # figure out some relevant values from config
    device = torch.device(runner.config["simulation_metadata"]["device"])
    NUM_STEPS_PER_EPISODE = runner.config["simulation_metadata"][
        "num_steps_per_episode"
    ]

    # set the r0 values and run the simulation
    runner.initializer.transition_function["0"][
        "new_transmission"
    ].external_R = r0_values
    runner.step(NUM_STEPS_PER_EPISODE)

    # get daily number of infections
    traj = runner.state_trajectory[-1][-1]
    daily_infections_arr = traj["environment"]["daily_infected"].to(device)
    predicted_weekly_cases = (
        daily_infections_arr.reshape(-1, 7).sum(axis=1).to(dtype=torch.float32)
    )

    return predicted_weekly_cases


def compare_runs(
    r0_values,
    run1_label: str,
    run1_config_path: str,
    run2_label: str,
    run2_config_path: str,
):
    """assumes run1 and run2 take place over the same time range in the same neighborhood."""
    # read the configs
    run1_config = read_config(run1_config_path)
    run2_config = read_config(run2_config_path)
    NEIGHBORHOOD = name_to_neighborhood(
        run1_config["simulation_metadata"]["NEIGHBORHOOD"]
    )
    EPIWEEK_START: Week = week_num_to_epiweek(
        run1_config["simulation_metadata"]["START_WEEK"]
    )
    NUM_WEEKS: int = run1_config["simulation_metadata"]["num_steps_per_episode"] // 7

    # get the predicted case numbers
    print("running first simulation...")
    run1_cases = simulate_once(r0_values, run1_config)
    print("\nrunning second simulation...")
    run2_cases = simulate_once(r0_values, run2_config)

    # get the actual case numbers
    ground_truth_cases = get_labels(
        NEIGHBORHOOD, EPIWEEK_START, NUM_WEEKS, Feature.CASES
    )

    # plot all three cases
    start_week_count = subtract_epiweek(EPIWEEK_START, DATA_START_WEEK)
    plt.plot(range(start_week_count, start_week_count + NUM_WEEKS),run1_cases.cpu().data,label=run1_label,)
    plt.plot(
        range(start_week_count, start_week_count + NUM_WEEKS),
        run2_cases.cpu().data,
        label=run2_label,
    )
    plt.plot(range(start_week_count, start_week_count + NUM_WEEKS),ground_truth_cases,label="ground truth",)
    plt.legend()

    breakpoint()
    plt.savefig("plot_runs.png")
    plt.show()


if __name__ == "__main__":
    compare_runs(
        torch.tensor(
            [
                1.4400,
                1.4362,
                1.4318,
                1.4295,
                1.4292,
                1.4295,
                1.4291,
                1.4290,
                1.4286,
                1.4292,
            ]
        ).to(torch.device("cuda")),
        "with llm",
        "experiments/0419_1900_simsim_2e2_mse/config.yaml",
        "no llm",
        "experiments/0419_1900_nollm_2e2_mse/config.yaml",
    )
