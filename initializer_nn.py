import pandas as pd
import torch
import torch.nn as nn
from utils.general import *

# State is a collection of parameters [nn.ParameterDict()], Substeps are a collection of functions. [nn.ModuleDict()]. Each module may also have nn.ParameterDict()
class Initializer(nn.Module):
    
    def __init__(self, registry, config):
        super().__init__()
        self.config = config
        self.registry = registry

        self.state = {}
        for key in self.config["state"].keys():
            self.state[key] = {}
            
        self.learnable_parameters = {}
        self.fixed_parameters = {}
        
        self.observation_function, self.policy_function, self.transition_function, self.reward_function = nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict()
        
        self.initialize()
    
    def environment(self, key="environment"):
        if self.config["state"][key] is None:
            print("skipping: ", key)
            return
                
        for attr in self.config["state"][key].keys():
            env_function = self.config["state"][key][attr]["initialization_function"]["generator"]
            shape, params = self.config["state"][key][attr]["shape"], self.config["state"][key][attr]["initialization_function"]["arguments"]

            self.state[key][attr] = self.registry.initialization_helpers[env_function](shape, params)
                        
            learnable = self.config['state'][key][attr]['learnable']
            if learnable:
                self.learnable_parameters[f"{key}_{attr}"] = self.state[key][attr]
            else:
                self.fixed_parameters[f"{key}_{attr}"] = self.state[key][attr]
    
    
    def agents_objects(self, key="agents"):
        if self.config["state"][key] is None:
            print("skipping: ", key)
            return
        
        for attr in self.config["state"][key].keys():
            if attr == "metadata":
                continue

            self.state[key][attr] = {}
            
            properties_attr = self.config["state"][key][attr]["properties"]
            if properties_attr is None:
                continue
            for p in properties_attr.keys():
                function = self.config["state"][key][attr]["properties"][p]["initialization_function"]["generator"]
                shape, params = self.config["state"][key][attr]["properties"][p]["shape"], self.config["state"][key][attr]["properties"][p]["initialization_function"]["arguments"]
                
                self.state[key][attr][p] = self.registry.initialization_helpers[function](shape, params)
                
                learnable = self.config["state"][key][attr]["properties"][p]["learnable"]
                if learnable:
                    self.learnable_parameters[f"{key}_{attr}_{p}"] = self.state[key][attr][p]
                else:
                    self.fixed_parameters[f"{key}_{attr}_{p}"] = self.state[key][attr][p]

    def network(self, key="network"):
        if self.config["state"][key] is None:
            print("skipping: ", key)
            return
        
        for attr in self.config["state"][key].keys():
            self.state[key][attr] = {}
            
            if self.config["state"][key][attr] is None:
                continue

            for t in self.config["state"][key][attr].keys():
                self.state[key][attr][t] = {}
                network_type = self.config["state"][key][attr][t]["type"]
                params = self.config["state"][key][attr][t]["network_params"]

                self.state[key][attr][t]["graph"], self.state[key][attr][t]["adjacency_matrix"] = self.registry.network_helpers[network_type](params)

    def simulator(self):
        self.environment()
        self.agents_objects("agents")
        self.agents_objects("objects")
        self.network()
        
        self.parameters = nn.ParameterDict(self.learnable_parameters)
        
    def _parse_function(self, function_object):
        generator = function_object["generator"]
        arguments = function_object["arguments"]
        input_variables = function_object["input_variables"]
        output_variables = function_object["output_variables"]

        learnable_args, fixed_args = {}, {}
        if arguments is not None:
            for argument in arguments:
                
                function = argument["initialization_function"]["generator"]
                shape, params = argument["shape"], argument["initialization_function"]["arguments"]
                
                arg_value = self.registry.initialization_helpers[function](shape, params)
                
                if arguments[arg_property]['learnable']:
                    learnable_args[arg_property] = arg_value
                else:
                    fixed_args[arg_property] = arg_value
                    
        arguments = {'learnable': learnable_args, 'fixed': fixed_args}
        
        return input_variables, output_variables, arguments

    def substeps(self):
        '''
            define the observation, policy and transition functions for each active_agents on each substep
        '''
        for substep in self.config["substeps"].keys():
            active_agents = self.config["substeps"][substep]["active_agents"]
            
            self.observation_function[substep], self.policy_function[substep], self.transition_function[substep] = nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict()
            
            for agent_type in active_agents:
                # observation_function
                agent_observations = self.config["substeps"][substep]["observation"][agent_type]
                self.observation_function[substep][agent_type] = nn.ModuleDict()
                
                for obs_func in agent_observations:
                    input_variables, output_variables, arguments = self._parse_function(agent_observations[obs_func])
                    self.observation_function[substep][agent_type][obs_func] = self.registry.observation_helpers[obs_func](self.config, input_variables, output_variables, arguments)
                
                # policy_function
                agent_policies = self.config["substeps"][substep]["policy"]
                self.policy_function[substep][agent_type] = nn.ModuleDict()
                
                for policy_func in agent_policies:
                    input_variables, output_variables, arguments = self._parse_function(agent_policies[policy_func])
                    self.policy_function[substep][agent_type][policy_func] = self.registry.policy_helpers[policy_func](self.config, input_variables, output_variables, arguments)
        
            # transition_function
            substep_transitions = self.config["substeps"][substep]["transition"]
            self.transition_function[substep] = nn.ModuleDict()
            
            for transition_func in substep_transitions:
                input_variables, output_variables, arguments = self._parse_function(substep_transitions[transition_func])
                self.transition_function[substep][transition_func] = self.registry.transition_helpers[transition_func](self.config, input_variables, output_variables, arguments)
                 
                
    def substeps_legacy(self):
        '''
            define the observation, policy and transition function for active_agents on each of the substeps
        '''
        for substep in self.config["substeps"].keys():
            active_agents = self.config["substeps"][substep]["active_agents"]            
            self.observation_function[substep], self.policy_function[substep], self.transition_function[substep] = nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict()
            
            for agent_type in active_agents:
                # observation function
                agent_observations = self.config["substeps"][substep]["observation"][agent_type]
                self.observation_function[substep][agent_type] = nn.ModuleDict()
                
                for obs in agent_observations.keys():                   
                    function = agent_observations[obs]["observation_function"]["generator"]
                    arguments = agent_observations[obs]["observation_function"]["arguments"]
                    input_variables = agent_observations[obs]["observation_function"]["input_variables"]
                    
                    learnable = agent_observations[obs]["observation_function"]["learnable"]                      
                    
                    learnable_params, fixed_params = assign_parameters(arguments, input_variables, learnable=learnable)
                    
                    self.observation_function[substep][agent_type][obs] = self.registry.observation_helpers[function](self.config, arguments, input_variables)
                    #self.observation_function[substep][agent_type][obs] = self.registry.observation_helpers[function](self.config, learnable_params, fixed_params)

                # policy function
                agent_policies = self.config["substeps"][substep]["policy"][agent_type]
                self.policy_function[substep][agent_type] = nn.ModuleDict()
                
                for policy in agent_policies.keys():
                    function = agent_policies[policy]["policy_function"]["generator"]
                    arguments = agent_policies[policy]["policy_function"]["arguments"]
                    input_variables = agent_policies[policy]["policy_function"]["obs_variables"]
                    
                    learnable = agent_policies[policy]["policy_function"]["learnable"]
                    learnable_params, fixed_params = assign_parameters(arguments, input_variables, learnable=learnable)
                    
                    self.policy_function[substep][agent_type][policy] = self.registry.policy_helpers[function](self.config, learnable_params, fixed_params)
    
            # transition function
            substep_transitions = self.config["substeps"][substep]["transition"]
            self.transition_function[substep] = nn.ModuleDict()
        
            for prop_val in substep_transitions.keys():           
                function = substep_transitions[prop_val]["transition_function"]["generator"]
                arguments = substep_transitions[prop_val]["transition_function"]["arguments"]
                input_variables = substep_transitions[prop_val]["transition_function"]["input_variables"]
                
                learnable = substep_transitions[prop_val]["transition_function"]["learnable"]
                #learnable_params, fixed_params = assign_parameters(arguments, input_variables, learnable=learnable)

                self.transition_function[substep][prop_val] = self.registry.transition_helpers[function](self.config, arguments, input_variables)

    def initialize(self):
        self.state["current_step"] = 0
        self.state["current_substep"] = '0' # use string not int for nn.ModuleDict
        
        self.simulator()
        self.substeps()
            
    def forward(self):
        self.initialize()