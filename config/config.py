import yaml

class Config:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.set_defaults()
        self.check_required_keys()

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def set_defaults(self):
        defaults = {
            'api_base_url': 'https://api.openai.com/v1/chat/completions',
            'model_name': 'gpt-4o-mini',
        }

        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if sub_key not in self.config[key]:
                        self.config[key][sub_key] = sub_value

    def check_required_keys(self):
        required_keys = ['openai_api_key']
        for key in required_keys:
            if key not in self.config or not self.config[key]:
                raise ValueError(f"Missing required configuration key: {key}")


    def get(self, key, default=None):
        return self.config.get(key, default)