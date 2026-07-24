import yaml, types
from util.helpers import convert_namespace_to_dict_recursive

class Config(types.SimpleNamespace):
    """Config class"""

    def __init__(self, **params):
        super().__init__(**params)

    @classmethod
    def from_yaml(cls, config_file):
        """Creates config from yaml"""
        with open(config_file, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        return cls.from_dict(params)

    @classmethod
    def from_dict(cls, params):
        """Creates config from Python dict"""

        def convert_to_namespace_recursive(val):
            # Recursively convert to namespace, so you can access it by attribute, e.g. config.config_data.keep_terminators
            if isinstance(val, dict):
                return types.SimpleNamespace(**{key: convert_to_namespace_recursive(val[key]) for key in val})
            elif isinstance(val, (list, tuple)):
                t = type(val)
                return t([convert_to_namespace_recursive(elem) for elem in val])
            else:
                return val

        params_converted = {k: convert_to_namespace_recursive(v) for k, v in params.items()}
        return cls(**params_converted)

    def to_dict(self):
        """Convert back to Python dict"""
        return {k: convert_namespace_to_dict_recursive(v) for k, v in vars(self).items()}

    def to_yaml(self, file_path=None):
        """Dump config as YAML (to string or file if path given)"""
        data = self.to_dict()
        if file_path:
            with open(file_path, "w") as f:
                yaml.dump(data, f)
        else:
            return yaml.dump(data)
