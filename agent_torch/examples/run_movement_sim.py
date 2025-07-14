print("--- RUNNING MOVEMENT SIMULATION ---")
import warnings

warnings.filterwarnings("ignore")

from agent_torch.populations import sample2
from agent_torch.examples.models import movement
from agent_torch.core.environment import envs


def run_movement_simulation():
    """Run the movement simulation using our example model."""
    print("\n=== Running Movement Simulation ===")

    # Create the runner using envs.create
    print("\nCreating simulation runner...")
    runner = envs.create(model=movement, population=sample2)

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

        # Print statistics
        final_state = runner.state
        positions = final_state["agents"]["citizens"]["position"]
        print(f"- Average position: {positions.mean(dim=0)}")

    print("\nSimulation completed!")
    return runner


if __name__ == "__main__":
    runner = run_movement_simulation()
