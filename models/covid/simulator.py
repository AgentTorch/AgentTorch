from agent_torch import Runner, Registry

def get_registry():
    reg = Registry()

    # Substep: New Disease Transmission
    from substeps.new_transmission.action import MakeIsolationDecision
    reg.register(MakeIsolationDecision, "make_isolation_decision", key="policy")

    from substeps.new_transmission.transition import NewTransmission
    reg.register(NewTransmission, "new_transmission", key="transition")

    # Substep: Disease Stage Progression
    from substeps.seirm_progression.transition import SEIRMProgression
    reg.register(SEIRMProgression, "seirm_progression", key="transition")

    from substeps.utils import (
        network_from_file,
        get_lam_gamma_integrals,
        get_mean_agent_interactions,
        get_infected_time,
        get_next_stage_time,
        load_population_attribute,
        initialize_id,
    )

    reg.register(network_from_file, "network_from_file", key="network")
    reg.register(
        get_lam_gamma_integrals, "get_lam_gamma_integrals", key="initialization"
    )
    reg.register(
        get_mean_agent_interactions, "get_mean_agent_interactions", key="initialization"
    )
    reg.register(get_infected_time, "get_infected_time", key="initialization")
    reg.register(get_next_stage_time, "get_next_stage_time", key="initialization")
    reg.register(
        load_population_attribute, "load_population_attribute", key="initialization"
    )
    reg.register(initialize_id, "initialize_id", key="initialization")

    from substeps.utils import read_from_file
    reg.register(read_from_file, "read_from_file", key="initialization")

    return reg


def get_runner(config, registry):
    CovidRunner = Runner(config, registry)

    return CovidRunner
