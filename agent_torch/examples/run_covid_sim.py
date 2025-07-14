print("--- DEBUGGING EXAMPLE ---")
import warnings

warnings.filterwarnings("ignore")
print("Running covid simulation")
from agent_torch.populations import sample2
from agent_torch.examples.models import covid

print("Imported covid model")
from agent_torch.core.environment import envs


def run_covid_simulation():
    """Run the COVID simulation using our example model."""
    print("\n=== Running COVID Simulation ===")

    # Create the runner using envs.create
    print("\nCreating simulation runner...")
    runner = envs.create(model=covid, population=sample2)

    # Get simulation parameters from config
    sim_steps = runner.config["simulation_metadata"]["num_steps_per_episode"]
    num_episodes = runner.config["simulation_metadata"]["num_episodes"]

    print(f"\nSimulation parameters:")
    print(f"- Steps per episode: {sim_steps}")
    print(f"- Number of episodes: {num_episodes}")
    print(f"- Number of agents: {runner.config['simulation_metadata']['num_agents']}")

    # Run all episodes
    print("\nRunning simulation episodes...")
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")

        # Reset state at the start of each episode
        if episode > 0:
            runner.reset()

        # Run one episode
        runner.step(sim_steps)

        # Print some statistics
        final_state = runner.state
        if "daily_infected" in final_state:
            print(f"- Final daily infected count: {final_state['daily_infected']}")
        if "daily_deaths" in final_state:
            print(f"- Final daily deaths count: {final_state['daily_deaths']}")

    print("\nSimulation completed!")
    return runner


if __name__ == "__main__":
    runner = run_covid_simulation()
