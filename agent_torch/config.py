from omegaconf import OmegaConf
import torch.nn as nn
import types

from agent_torch.registry import Registry


class Configurator(nn.Module):
    def __init__(self):
        super().__init__()

        self.metadata = OmegaConf.create()

        self.agents = OmegaConf.create()
        self.environment = OmegaConf.create()
        self.objects = OmegaConf.create()
        self.network = OmegaConf.create()

        self.substeps = OmegaConf.create()

        self.state = OmegaConf.create(
            {
                "environment": self.environment,
                "agents": self.agents,
                "objects": self.objects,
                "network": self.network,
            }
        )
        self.config = OmegaConf.create(
            {
                "simulation_metadata": self.metadata,
                "state": self.state,
                "substeps": self.substeps,
            }
        )

        self.substep_counter = 0

        self.reg = Registry()

    def get(self, variable_name):
        return OmegaConf.select(self.config, variable_name)

    def render(self, config_path):
        OmegaConf.save(self.config, config_path)

    def create_variable(
        self,
        key,
        name,
        shape,
        dtype,
        initialization_function=None,
        learnable=False,
        value=None,
    ):
        """Fundamental unit of an AgentTorch simulator which is learnable or not"""
        variable_dict = OmegaConf.create()
        variable_dict.update({"name": name})
        variable_dict.update({"shape": shape})
        variable_dict.update({"initialization_function": initialization_function})
        variable_dict.update({"learnable": learnable})
        variable_dict.update({"dtype": dtype})

        if initialization_function is None:
            assert value is not None
            variable_dict.update({"value": value})

        return OmegaConf.create({key: variable_dict})

    def create_initializer(self, generator, arguments):
        initializer = OmegaConf.create()

        if isinstance(generator, types.FunctionType):
            generator_name = generator.__name__
            self.reg.register(generator, generator_name, key="initialization")
            initializer.update({"generator": generator_name})
        else:
            initializer.update({"generator": generator})

        arguments_dict = OmegaConf.merge(*arguments)
        initializer.update({"arguments": arguments_dict})

        return initializer

    def create_function(
        self,
        generator,
        fn_type,
        arguments=None,
        input_variables=None,
        output_variables=None,
    ):
        substep_function = OmegaConf.create()

        if isinstance(generator, type):
            generator_name = generator.__name__
            self.reg.register(generator, generator_name, key=fn_type)
            substep_function.update({"generator": generator_name})
        else:
            substep_function.update({"generator": generator})

        arguments_dict = arguments
        if arguments is not None:
            arguments_dict = OmegaConf.merge(*arguments)

        substep_function.update({"input_variables": input_variables})
        substep_function.update({"output_variables": output_variables})
        substep_function.update({"arguments": arguments_dict})

        return OmegaConf.create({generator_name: substep_function})

    def add_property(
        self,
        root,
        key,
        name,
        shape,
        dtype,
        initialization_function,
        learnable=False,
        value=None,
    ):
        root_object = self.get(root)
        property_object = self.create_variable(
            key=key,
            name=name,
            shape=shape,
            dtype=dtype,
            initialization_function=initialization_function,
            learnable=learnable,
            value=value,
        )

        root_object["properties"].update({key: property_object[key]})

    def add_function(
        self,
        root,
        generator,
        fn_type,
        active_agent,
        arguments=None,
        input_variables=None,
        output_variables=None,
    ):
        root_object = self.get(root)

        function_object = self.create_variable(
            generator,
            fn_type,
            arguments=arguments,
            input_variables=input_variables,
            output_variables=output_variables,
        )
        root_object[fn_type][active_agent].update({generator.__name__: function_object})

    def add_metadata(self, key, value):
        self.config["simulation_metadata"].update({key: value})

    def add_agents(self, key, number, all_properties=None):
        _created_agent = OmegaConf.create()
        if all_properties is None:
            all_properties = OmegaConf.create()

        _created_agent.update({"number": number, "properties": all_properties})
        self.config["state"]["agents"].update({key: _created_agent})

    def add_objects(self, key, number, all_properties=None):
        _created_object = OmegaConf.create()
        if all_properties is None:
            all_properties = OmegaConf.create()

        _created_object.update({"number": number, "properties": all_properties})
        self.config["state"]["objects"].update({key: _created_object})

    def add_network(
        self, network_name, network_type, arguments, category="agent_agent"
    ):
        _network_obj = OmegaConf.create()

        network_type_key = network_type
        if isinstance(network_type, types.FunctionType):
            network_type_key = network_type.__name__
            self.reg.register(network_type, network_type_key, key="network")

        _network_obj.update(
            {network_name: {"type": network_type_key, "arguments": arguments}}
        )
        self.config["state"]["network"].update({category: _network_obj})

    def add_substep(
        self,
        name,
        active_agents,
        observation_fn=None,
        policy_fn=None,
        transition_fn=None,
    ):
        _created_substep = OmegaConf.create()

        _created_substep.update({"name": name, "active_agents": active_agents})

        if observation_fn is None:
            observation_fn_obj = OmegaConf.create()
            for agent in active_agents:
                observation_fn_obj.update({agent: None})
            _created_substep.update({"observation": observation_fn_obj})
        else:
            for agent in active_agents:
                _created_substep.update(
                    {"observation": {agent: OmegaConf.merge(*observation_fn)}}
                )

        if policy_fn is None:
            policy_fn_obj = OmegaConf.create()
            for agent in active_agents:
                policy_fn_obj.update({agent: None})
            _created_substep.update({"policy": policy_fn_obj})
        else:
            for agent in active_agents:
                _created_substep.update(
                    {"policy": {agent: OmegaConf.merge(*policy_fn)}}
                )

        if transition_fn is None:
            transition_fn_obj = OmegaConf.create()
            _created_substep.update({"transition": transition_fn_obj})
        else:
            _created_substep.update({"transition": OmegaConf.merge(*transition_fn)})

        self.config["substeps"].update({str(self.substep_counter): _created_substep})
        self.substep_counter += 1
