AGENT_TORCH_PATH = '/u/ayushc/projects/GradABM/AgentTorch'

import pandas as pd
import numpy as np 
import torch

import sys
sys.path.insert(0, AGENT_TORCH_PATH)
from AgentTorch import Runner, Registry

def get_registry():
    reg = Registry()
    
    # Substep: New Disease Transmission
    from substeps.new_transmission.transition import NewTransmission
    reg.register(NewTransmission, "new_transmission", key="transition")

    # Substep: Disease Stage Progression
    from substeps.seirm_progression.transition import SEIRMProgression
    reg.register(SEIRMProgression, "seirm_progression", key="transition")

    # Substep: Quarantine Intervention
    from substeps.quarantine.transition import UpdateQuarantineStatus
    reg.register(UpdateQuarantineStatus, "update_quarantine_status", key="transition")
    from substeps.quarantine.action import StartCompliance, BreakCompliance
    reg.register(StartCompliance, "start_compliance", key="policy")
    reg.register(BreakCompliance, "break_compliance", key="policy")
    from substeps.quarantine.observation import GetFromState
    reg.register(GetFromState, "get_from_state", key="observation")
    
    # Substep: Testing Intervention
    from substeps.testing.transition import UpdateTestStatus
    reg.register(UpdateTestStatus, "update_test_status", key="transition")
    from substeps.testing.action import AcceptTest
    reg.register(AcceptTest, "accept_test", key="policy")
    
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
