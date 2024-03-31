import torch
from torch import distributions, nn
from torch_geometric.data import Data
import torch.nn.functional as F
import re
import pdb

from AgentTorch.substep import SubstepTransitionMessagePassing
from AgentTorch.helpers import get_by_path
from substeps.utils import *

class NewTransmission(SubstepTransitionMessagePassing):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

        self.device = self.config['simulation_metadata']['device']
        self.SUSCEPTIBLE_VAR = self.config['simulation_metadata']['SUSCEPTIBLE_VAR']
        self.EXPOSED_VAR = self.config['simulation_metadata']['EXPOSED_VAR']
        self.RECOVERED_VAR = self.config['simulation_metadata']['RECOVERED_VAR']

        self.STAGE_UPDATE_VAR = 1
        self.INFINITY_TIME = self.config['simulation_metadata']['INFINITY_TIME']
        self.EXPOSED_TO_INFECTED_TIME = self.config['simulation_metadata']['EXPOSED_TO_INFECTED_TIME']

    def _lam(self, x_i, x_j, edge_attr, t, R, SFSusceptibility, SFInfector, lam_gamma_integrals):
        S_A_s = SFSusceptibility[x_i[:,0].long()]
        A_s_i = SFInfector[x_j[:,1].long()]
        B_n = edge_attr[1, :]
        integrals = torch.zeros_like(B_n)
        infected_idx = x_j[:, 2].bool()    
        infected_times = t - x_j[infected_idx, 3] - 1
                    
        integrals[infected_idx] =  lam_gamma_integrals[infected_times.long()]
        edge_network_numbers = edge_attr[0, :]
                
        I_bar = torch.gather(x_i[:, 4], 0, edge_network_numbers.long()).view(-1)
        
        not_isolated = 1 - x_j[:, 5] # logical_not(will_isolate)
        
        res = not_isolated*R*S_A_s*A_s_i*B_n*integrals #/I_bar

        return res.view(-1, 1)
    
    def message(self, x_i, x_j, edge_attr, t, R, SFSusceptibility, SFInfector, lam_gamma_integrals):
        return self._lam(x_i, x_j, edge_attr, t, R, SFSusceptibility, SFInfector, lam_gamma_integrals)

    def update_stages(self, current_stages, newly_exposed_today):
        new_stages = current_stages + newly_exposed_today * self.STAGE_UPDATE_VAR        
        return new_stages
    
    def update_transition_times(self, t, current_transition_times, newly_exposed_today):
        '''Note: not differentiable'''
        new_transition_times = torch.clone(current_transition_times).to(current_transition_times.device)
        new_transition_times = newly_exposed_today*(t+ 1 + self.EXPOSED_TO_INFECTED_TIME) + (1 - newly_exposed_today)*current_transition_times
#         current_transition_times[newly_exposed_today] = t + 1 + self.EXPOSED_TO_INFECTED_TIME
        return new_transition_times
    
    def update_infected_times(self, t, agents_infected_times, newly_exposed_today):
        '''Note: not differentiable'''
        updated_infected_times = torch.clone(agents_infected_times).to(agents_infected_times.device)
                
#         updated_infected_times = newly_exposed_today*t + (1 - newly_exposed_today)*agents_infected_times
        updated_infected_times[newly_exposed_today.bool()] = t
#         agents_infected_times[newly_exposed_today] = t
        return updated_infected_times
    
    def forward(self, state, action=None):
        input_variables = self.input_variables
        t = state['current_step']
        print("Substep: Disease Transmission")
        
        # R = state['environment']['R']
        #         R = get_by_path(state, re.split("/", input_variables['R']))
        R = self.learnable_args['R2']
                
        SFSusceptibility = get_by_path(state, re.split("/", input_variables['SFSusceptibility']))
        SFInfector = get_by_path(state, re.split("/", input_variables['SFInfector']))
        all_lam_gamma = get_by_path(state, re.split("/", input_variables['lam_gamma_integrals']))
        
        agents_infected_time = get_by_path(state, re.split("/", input_variables['infected_time']))
        agents_mean_interactions_split = get_by_path(state, re.split("/", input_variables['mean_interactions']))
        agents_ages = get_by_path(state, re.split("/", input_variables['age']))                     
        current_stages = get_by_path(state, re.split("/", input_variables['disease_stage']))
        current_transition_times = get_by_path(state, re.split("/", input_variables['next_stage_time']))
        
        all_edgelist, all_edgeattr = get_by_path(state, re.split("/", input_variables["adjacency_matrix"]))
        
        agents_infected_index = torch.logical_and(current_stages > self.SUSCEPTIBLE_VAR, current_stages < self.RECOVERED_VAR)
        
        will_isolate = action['citizens']['isolation_decision']
                        
        all_node_attr = torch.stack((
                agents_ages,  #0
                current_stages.detach(),  #1
                agents_infected_index, #2
                agents_infected_time, #3
                agents_mean_interactions_split, # *agents_mean_interactions_split,
                torch.unsqueeze(torch.arange(self.config['simulation_metadata']['num_agents']), 1).to(self.device),
                will_isolate)).transpose(0,1).squeeze() #.t()
            
        agents_data = Data(all_node_attr, edge_index=all_edgelist, edge_attr=all_edgeattr, t=t)
                
        new_transmission = self.propagate(agents_data.edge_index, x=agents_data.x, edge_attr=agents_data.edge_attr, t=agents_data.t, R=R, SFSusceptibility=SFSusceptibility, SFInfector=SFInfector, lam_gamma_integrals=all_lam_gamma.squeeze())
        
        prob_not_infected = torch.exp(-1*new_transmission)
        p = torch.hstack((1-prob_not_infected,prob_not_infected))
        cat_logits = torch.log(p+1e-9)
        
        potentially_exposed_today = F.gumbel_softmax(logits=cat_logits,tau=1,hard=True,dim=1)[:,0]        
        newly_exposed_today = (current_stages==self.SUSCEPTIBLE_VAR).squeeze()*potentially_exposed_today

        pdb.set_trace()
        
        # d(newly_exposed_today) / d(R)
                
        newly_exposed_today = newly_exposed_today.unsqueeze(1)
        
        updated_stages = self.update_stages(current_stages, newly_exposed_today)    
        updated_next_stage_times = self.update_transition_times(t, current_transition_times, newly_exposed_today)
        updated_infected_times = self.update_infected_times(t, agents_infected_time, newly_exposed_today)
                        
        return {self.output_variables[0]: updated_stages, 
                self.output_variables[1]: updated_next_stage_times, 
                self.output_variables[2]: updated_infected_times}