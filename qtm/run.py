from omegaconf import OmegaConf

OmegaConf.register_new_resolver("concat", lambda x, y: x+y)
_ = OmegaConf.load("chem_config.yaml")
OmegaConf.resolve(_)
print(_)
