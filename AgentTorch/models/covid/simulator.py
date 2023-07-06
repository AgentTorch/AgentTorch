import pandas as pd
import numpy as np 
import torch

from AgentTorch import Runner, Registry

def get_registry():
    
    reg = Registry()
    
    from substeps.seirm_progression.transition import SEIRMProgression
    reg.register(SEIRMProgression, "seirm_progression", key="transition")

    from substeps.new_transmission.transition import NewTransmission
    reg.register(NewTransmission, "new_transmission", key="transition")
         
    from substeps.utils import network_from_file, get_lam_gamma_integrals, get_mean_agent_interactions, get_infected_time, get_next_stage_time
    reg.register(network_from_file, "network_from_file", key="network")
    reg.register(get_lam_gamma_integrals, "get_lam_gamma_integrals", key="initialization")
    reg.register(get_mean_agent_interactions, "get_mean_agent_interactions", key="initialization")
    reg.register(get_infected_time, "get_infected_time", key="initialization")
    reg.register(get_next_stage_time, "get_next_stage_time", key="initialization")

    from AgentTorch.helpers import read_from_file
    reg.register(read_from_file, "read_from_file", key="initialization")
    
    return reg    

def get_runner(config, registry):
    CovidRunner = Runner(config, registry)

    return CovidRunner
