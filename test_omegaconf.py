import omegaconf
from omegaconf import OmegaConf

# Define the variables that we want to link
state = omegaconf.OmegaConf.create({'agents': {'consumers': {'number': 6400}}})
metadata = omegaconf.OmegaConf.create({'num_products': 10})

# Create the property using linked variables
property_dict = omegaconf.OmegaConf.create()
property_dict.update({'name': 'Purchased before'})
property_dict.update({'description': 'whether the particular product has been purchased before'})
property_dict.update({'dtype': 'int'})
property_dict.update({'variable_type': 'scalar'})
property_dict.update({'shape': ['${agents.consumers.number}', '${num_products}']})
property_dict.update({'value_range': {'min': 0, 'max': 1}})
property_dict.update({'initialization_function': {'generator': 'zeros', 'arguments': {'dtype': 'int'}}})

omega_all = OmegaConf.merge(state, metadata, property_dict)
resolved = OmegaConf.to_container(omega_all, resolve=True)

resolved_property_dict = OmegaConf.create(resolved)

# Create the YAML string with linked variables
# yaml_str = omegaconf.OmegaConf.to_yaml(property_dict, resolve=True, variables={'state': state, 'objects': omegaconf.OmegaConf.create({'metadata': metadata})})

# print(yaml_str)

print(omegaconf.OmegaConf.to_yaml(resolved_property_dict))
