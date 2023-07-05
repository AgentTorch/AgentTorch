import pandas as pd
import numpy as np 
import torch

from AgentTorch import Runner, Registry

print("Imports completed..")

def create_registry():
    
    reg = Registry()
    
    # transition
    from substep.seirm_progression.transition import SEIRMProgression
    reg.register(SEIRMProgression, "seirm_progression", key="transition")

    from substep.new_transmission.transition import NewTransmission
    reg.register(NewTransmission, "new_transmission", key="transition")
         
    # initialization and network
    from substep.utils import network_from_file, get_lam_gamma_integrals, get_mean_agent_interactions, get_infected_time, get_next_stage_time
    reg.register(network_from_file, "network_from_file", key="network")
    reg.register(get_lam_gamma_integrals, "get_lam_gamma_integrals", key="initialization")
    reg.register(get_mean_agent_interactions, "get_mean_agent_interactions", key="initialization")
    reg.register(get_infected_time, "get_infected_time", key="initialization")
    reg.register(get_next_stage_time, "get_next_stage_time", key="initialization")

    from AgentTorch.helpers import read_from_file
    reg.register(read_from_file, "read_from_file", key="initialization")
    
    return reg    

if __name__ == '__main__':
    print("The runner file..")
    args = parser.parse_args()
    
    config_file = args.config

    # create runner object
    runner = Runner(config_file)    
    runner.execute()
    
    for name, param in runner.named_parameters(): 
        print(name, param.data)

    import ipdb; ipdb.set_trace()    
