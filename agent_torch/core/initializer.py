import torch
import torch.nn as nn
import os

# import dask.dataframe as dd
from agent_torch.core.helpers.general import *


class Initializer(nn.Module):
    """Constructs the initial simulation state and helper modules.

    cpu: builds everything on host; cuda: enables lightweight stream utilities
    for faster host→device transfers during initialization.
    """
    def __init__(self, config, registry):
        super().__init__()
        self.config = config
        self.registry = registry
        # resolve device from config, supporting 'auto'
        cfg_dev = str(self.config["simulation_metadata"].get("device", "auto")).lower()
        if cfg_dev == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            # fall back to whatever torch recognizes (e.g., 'cpu', 'cuda')
            self.device = torch.device(cfg_dev)
        # write back resolved device so downstream modules don't see 'auto'
        self.config["simulation_metadata"]["device"] = self.device.type

        # enable lightweight cuda utilities without config changes
        self._enable_streaming = (self.device.type == 'cuda' and torch.cuda.is_available())
        if self._enable_streaming:
            try:
                self._num_streams = int(os.getenv('AGENT_TORCH_INIT_NUM_STREAMS', '4'))
            except Exception:
                self._num_streams = 4
            self._streams = [torch.cuda.Stream(device=self.device) for _ in range(max(1, self._num_streams))]
            self._stream_rr = 0

        self.state = {}
        self.environment, self.agents, self.objects, self.networks = {}, {}, {}, {}

        self.fixed_parameters, self.learnable_parameters = {}, {}

        (
            self.observation_function,
            self.policy_function,
            self.transition_function,
            self.reward_function,
        ) = (nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict())

    def _next_stream(self):
        """get next cuda stream for overlapped operations (cuda only)."""
        if not self._enable_streaming:
            return torch.cuda.current_stream(self.device) if self.device.type == 'cuda' and torch.cuda.is_available() else None
        s = self._streams[self._stream_rr]
        self._stream_rr = (self._stream_rr + 1) % len(self._streams)
        return s

    def _to_device_streamed(self, cpu_tensor: torch.Tensor) -> torch.Tensor:
        """pin and transfer to device asynchronously using round‑robin streams."""
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            return cpu_tensor.to(self.device)
        stream = self._next_stream()
        if not cpu_tensor.is_contiguous():
            cpu_tensor = cpu_tensor.contiguous()
        if hasattr(cpu_tensor, 'is_pinned') and not cpu_tensor.is_pinned():
            try:
                cpu_tensor = cpu_tensor.pin_memory()
            except Exception:
                pass
        with torch.cuda.stream(stream):
            gpu_tensor = cpu_tensor.to(self.device, non_blocking=True)
        return gpu_tensor

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """route device transfer via streamed path when available."""
        try:
            return self._to_device_streamed(tensor) if self._enable_streaming else tensor.to(self.device)
        except Exception:
            return tensor.to(self.device)

    def _initialize_from_default(self, src_val, shape):
        processed_shape = shape
        if type(src_val) == str:
            return src_val

        if type(src_val) == list:
            # Convert list to tensor
            src_tensor = torch.tensor(src_val)

            # If shape is specified and different from the source tensor shape,
            # create a tensor of the desired shape and fill it appropriately
            if processed_shape and list(src_tensor.shape) != processed_shape:
                # Create tensor of desired shape
                init_value = torch.zeros(size=processed_shape)

                # Fill the tensor by broadcasting the source values
                # For example: [0.0, 0.0] with shape [1000, 2] becomes a 1000x2 tensor filled with zeros
                if len(src_tensor.shape) == 1 and len(processed_shape) >= 2:
                    # If source is 1D and target is multi-dimensional, broadcast along the last dimension
                    if src_tensor.shape[0] == processed_shape[-1]:
                        # Source matches last dimension, broadcast across all other dimensions
                        init_value = src_tensor.unsqueeze(0).expand(processed_shape)
                    else:
                        # Use first value of source to fill entire tensor
                        init_value.fill_(src_tensor[0].item())
                else:
                    # For other cases, use first value to fill the tensor
                    init_value.fill_(src_tensor.flatten()[0].item())
            else:
                # Shape matches or no shape specified, use tensor as-is
                init_value = src_tensor
        # elif isinstance(src_val, (dd.Series, dd.DataFrame)):  # If Dask data is passed
        #     init_value = torch.tensor(src_val.compute().to_numpy())
        else:
            init_value = src_val * torch.ones(size=processed_shape)

        init_value = self._to_device(init_value)

        return init_value

    def _initialize_from_generator(
        self, initializer_object, initialize_shape, name_root
    ):
        function = initializer_object["generator"]

        params = {}
        for argument in initializer_object["arguments"].keys():
            arg_object = initializer_object["arguments"][argument]

            arg_name = f"{name_root}_{argument}"

            arg_learnable, arg_shape = arg_object["learnable"], arg_object["shape"]
            arg_init_func = arg_object["initialization_function"]

            if arg_init_func is None:
                arg_value = self._initialize_from_default(
                    arg_object["value"], arg_shape
                )
            else:
                print(
                    "dynamic arguments are not currently supported. Setup from fixed value !!!"
                )
                return

            params[argument] = arg_value

            if arg_learnable:
                self.learnable_parameters[arg_name] = arg_value
            else:
                self.fixed_parameters[arg_name] = arg_value

        init_value = self.registry.initialization_helpers[function](
            initialize_shape, params
        )
        init_value = self._to_device(init_value)

        return init_value

    def _initialize_property(self, property_object, property_key):
        property_name = property_object["name"]
        property_shape, property_dtype = (
            property_object["shape"],
            property_object["dtype"],
        )
        property_is_learnable = property_object["learnable"]
        property_initializer = property_object["initialization_function"]

        if property_initializer is None:
            property_value = self._initialize_from_default(
                property_object["value"], property_shape
            )
        else:
            property_value = self._initialize_from_generator(
                property_initializer, property_shape, property_key
            )

        property_value = self._to_device(property_value)

        if property_is_learnable:
            self.learnable_parameters[property_key] = property_value
        else:
            self.fixed_parameters[property_key] = property_value

        return property_value, property_is_learnable

    def init_environment(self, key="environment"):
        if self.config["state"][key] is None:
            return

        for prop in self.config["state"][key].keys():
            property_object = self.config["state"][key][prop]
            property_value, property_is_learnable = self._initialize_property(
                property_object, property_key=f"{key}_{prop}"
            )
            self.environment[prop] = property_value

    def init_agents(self, key="agents"):
        if self.config["state"][key] is None:
            # print("Skipping: ", key)
            return

        for instance_type in self.config["state"][key].keys():
            if instance_type == "metadata":
                continue

            self.agents[instance_type] = {}
            instance_properties = self.config["state"][key][instance_type]["properties"]
            if instance_properties is None:
                continue

            for prop in instance_properties.keys():
                property_object = instance_properties[prop]
                # if isinstance(property_object.get("value"), (dd.Series, dd.DataFrame)):
                #     property_object["value"] = property_object["value"].compute().to_numpy()
                property_value, property_is_learnable = self._initialize_property(
                    property_object, property_key=f"{key}_{instance_type}_{prop}"
                )
                self.agents[instance_type][prop] = property_value

    def init_objects(self, key="objects"):
        if self.config["state"][key] is None:
            # print("Skipping: ", key)
            return

        for instance_type in self.config["state"][key].keys():
            if instance_type == "metadata":
                continue

            self.objects[instance_type] = {}
            instance_properties = self.config["state"][key][instance_type]["properties"]
            if instance_properties is None:
                continue

            for prop in instance_properties.keys():
                property_object = instance_properties[prop]
                property_value, property_is_learnable = self._initialize_property(
                    property_object, property_key=f"{key}_{instance_type}_{prop}"
                )
                self.objects[instance_type][prop] = property_value

    def init_network(self, key="network"):
        if self.config["state"][key] is None:
            # print("Skipping.. ", key)
            return

        for interaction_type in self.config["state"][key].keys():
            self.networks[interaction_type] = {}

            if self.config["state"][key][interaction_type] is None:
                continue

            for contact_network in self.config["state"][key][interaction_type].keys():
                self.networks[interaction_type][contact_network] = {}

                network_type = self.config["state"][key][interaction_type][
                    contact_network
                ]["type"]
                params = self.config["state"][key][interaction_type][contact_network][
                    "arguments"
                ]

                graph, adjacency_matrix = self.registry.network_helpers[network_type](
                    params
                )

                if len(adjacency_matrix) == 2:
                    edge_list, attr_list = adjacency_matrix
                    edge_list = self._to_device(edge_list)
                    attr_list = self._to_device(attr_list)
                    adjacency_matrix = (edge_list, attr_list)

                self.networks[interaction_type][contact_network]["graph"] = graph
                self.networks[interaction_type][contact_network][
                    "adjacency_matrix"
                ] = adjacency_matrix

                # self.networks[interaction_type][contact_network]["adjacency_matrix"] = (
                #     edge_list.to(self.device),
                #     attr_list.to(self.device),
                # )

    def simulator(self):
        self.init_environment()
        self.init_agents(key="agents")
        self.init_objects(key="objects")
        self.init_network()

        # track learnable parameters
        self.parameters_dict = nn.ParameterDict(self.learnable_parameters)

    # ========================================================================================
    # LAZY RECONSTRUCTION: Dense adjacency matrices and NetworkX graphs
    # ========================================================================================

    @property
    def dense_adjacency_matrices(self):
        """
        Get dense adjacency matrices, reconstructing from sparse if needed.
        """
        if not hasattr(self, '_dense_cache'):
            self._dense_cache = {}
        for interaction_type in self.networks:
            for contact_network in self.networks[interaction_type]:
                key = f"{interaction_type}_{contact_network}"
                if key not in self._dense_cache:
                    self._dense_cache[key] = self._reconstruct_dense_for(interaction_type, contact_network)
        return self._dense_cache

    @property
    def networkx_graphs(self):
        """
        Get NetworkX graphs, reconstructing from sparse if needed.
        """
        if not hasattr(self, '_networkx_cache'):
            self._networkx_cache = {}
        for interaction_type in self.networks:
            for contact_network in self.networks[interaction_type]:
                key = f"{interaction_type}_{contact_network}"
                if key not in self._networkx_cache:
                    self._networkx_cache[key] = self._reconstruct_networkx_for(interaction_type, contact_network)
        return self._networkx_cache

    def get_dense_adjacency(self, interaction_type: str, contact_network: str):
        key = f"{interaction_type}_{contact_network}"
        if not hasattr(self, '_dense_cache'):
            self._dense_cache = {}
        if key not in self._dense_cache:
            self._dense_cache[key] = self._reconstruct_dense_for(interaction_type, contact_network)
        return self._dense_cache[key]

    def get_networkx_graph(self, interaction_type: str, contact_network: str):
        key = f"{interaction_type}_{contact_network}"
        if not hasattr(self, '_networkx_cache'):
            self._networkx_cache = {}
        if key not in self._networkx_cache:
            self._networkx_cache[key] = self._reconstruct_networkx_for(interaction_type, contact_network)
        return self._networkx_cache[key]

    def _reconstruct_dense_for(self, interaction_type: str, contact_network: str):
        try:
            network = self.networks[interaction_type][contact_network]
            adjacency_data = network["adjacency_matrix"]
            if isinstance(adjacency_data, tuple) and len(adjacency_data) == 2:
                edge_index, edge_attr = adjacency_data
                if edge_index.numel() == 0:
                    return torch.eye(2, device=self.device)
                num_nodes = int(edge_index.max().item()) + 1
                dense = torch.zeros(num_nodes, num_nodes, device=self.device, dtype=edge_attr.dtype)
                weights = edge_attr[1] if (edge_attr.dim() > 1 and edge_attr.size(0) >= 2) else edge_attr
                dense[edge_index[0], edge_index[1]] = weights
                return dense
            elif torch.is_tensor(adjacency_data):
                return adjacency_data.to(self.device)
            else:
                raise ValueError(f"Unknown adjacency format: {type(adjacency_data)}")
        except Exception:
            return torch.eye(2, device=self.device)

    def _reconstruct_networkx_for(self, interaction_type: str, contact_network: str):
        try:
            import networkx as nx
            network = self.networks[interaction_type][contact_network]
            adjacency_data = network["adjacency_matrix"]
            if isinstance(adjacency_data, tuple) and len(adjacency_data) == 2:
                edge_index, edge_attr = adjacency_data
                G = nx.Graph()
                if edge_index.numel() > 0:
                    edges_cpu = edge_index.cpu().numpy()
                    weights_cpu = edge_attr[1].cpu().numpy() if edge_attr.dim() > 1 else edge_attr.cpu().numpy()
                    for i in range(edges_cpu.shape[1]):
                        src, dst = edges_cpu[0, i], edges_cpu[1, i]
                        weight = weights_cpu[i] if getattr(weights_cpu, 'ndim', 1) > 0 else 1.0
                        G.add_edge(int(src), int(dst), weight=float(weight))
                return G
            elif torch.is_tensor(adjacency_data):
                dense_cpu = adjacency_data.cpu().numpy()
                import networkx as nx
                G = nx.from_numpy_array(dense_cpu)
                return G
            else:
                raise ValueError(f"Unknown adjacency format: {type(adjacency_data)}")
        except Exception:
            import networkx as nx
            return nx.Graph()

    def clear_reconstruction_cache(self):
        if hasattr(self, '_dense_cache'):
            self._dense_cache.clear()
        if hasattr(self, '_networkx_cache'):
            self._networkx_cache.clear()

    def _parse_function(self, function_object, name_root):
        generator = function_object["generator"]
        input_variables = function_object["input_variables"]
        output_variables = function_object["output_variables"]

        arguments = function_object["arguments"]
        learnable_args, fixed_args = {}, {}
        if arguments is not None:
            for argument in arguments:
                arg_name = f"{name_root}_{argument}"

                arg_object = arguments[argument]
                arg_function = arg_object["initialization_function"]
                arg_learnable = arg_object["learnable"]
                arg_shape = arg_object["shape"]

                if arg_function is None:
                    arg_value = self._initialize_from_default(
                        arg_object["value"], arg_shape
                    )
                else:
                    arg_value = self._initialize_from_generator(
                        arg_function, arg_shape, name_root=arg_name
                    )

                if arg_learnable:
                    self.learnable_parameters[arg_name] = arg_value
                    learnable_args[argument] = arg_value
                else:
                    self.fixed_parameters[arg_name] = arg_value
                    fixed_args[argument] = arg_value

        arguments = {"learnable": learnable_args, "fixed": fixed_args}

        return input_variables, output_variables, arguments

    def substeps(self):
        """define observation, policy and transition modules for each substep."""

        for substep in self.config["substeps"].keys():
            active_agents = self.config["substeps"][substep]["active_agents"]

            (
                self.observation_function[substep],
                self.policy_function[substep],
                self.transition_function[substep],
            ) = (nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict())

            for agent_type in active_agents:
                # observation function
                agent_observations = self.config["substeps"][substep]["observation"][
                    agent_type
                ]
                self.observation_function[substep][agent_type] = nn.ModuleDict()
                if agent_observations is not None:
                    for obs_func in agent_observations:
                        input_variables, output_variables, arguments = (
                            self._parse_function(
                                agent_observations[obs_func],
                                name_root=f"{agent_type}_observation_{obs_func}",
                            )
                        )
                        self.observation_function[substep][agent_type][
                            obs_func
                        ] = self.registry.observation_helpers[obs_func](
                            self.config,
                            input_variables,
                            output_variables,
                            arguments,
                        )

                # policy function
                agent_policies = self.config["substeps"][substep]["policy"][agent_type]
                self.policy_function[substep][agent_type] = nn.ModuleDict()

                if agent_policies is not None:
                    for policy_func in agent_policies:
                        input_variables, output_variables, arguments = (
                            self._parse_function(
                                agent_policies[policy_func],
                                name_root=f"{agent_type}_policy_{policy_func}",
                            )
                        )
                        self.policy_function[substep][agent_type][
                            policy_func
                        ] = self.registry.policy_helpers[policy_func](
                            self.config,
                            input_variables,
                            output_variables,
                            arguments,
                        )

            # transition function
            substep_transitions = self.config["substeps"][substep]["transition"]
            self.transition_function[substep] = nn.ModuleDict()

            for transition_func in substep_transitions:
                input_variables, output_variables, arguments = self._parse_function(
                    substep_transitions[transition_func],
                    name_root=f"_transition_{transition_func}",
                )
                self.transition_function[substep][transition_func] = (
                    self.registry.transition_helpers[transition_func](
                        self.config, input_variables, output_variables, arguments
                    )
                )

    def initialize(self):
        """populate `self.state` and build helper module dicts."""
        self.state["current_step"] = 0
        self.state["current_substep"] = "0"  # use string not int for nn.ModuleDict

        self.simulator()
        self.substeps()

        self.state["environment"] = self.environment
        self.state["network"] = self.networks
        self.state["agents"] = self.agents
        self.state["objects"] = self.objects

        self.state["parameters"] = self.parameters_dict

    def forward(self):
        """nn.Module forward delegates to initialize for convenience."""
        self.initialize()

    def __getstate__(self):
        state_dict = self.state.copy()
        state_dict["parameters"] = self.learnable_params_dict.state_dict()
        return state_dict

    def __setstate__(self, state):
        self.learnable_params = nn.ParameterDict(state["parameters"])
        self.state = state
        self.state["parameters"] = self.learnable_params
