import os
import yaml


def load_model_config(stage, mode):
    # load special config for each model
    base_path = os.path.dirname(os.path.realpath(__file__))

    # config_path = f'stage_{stage}.yaml'
    config_path = f'{base_path}/stage_{stage}.yaml'
    print(f'[!] load configuration from {config_path}')
    with open(config_path) as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)
        new_config = {}
        for key, value in configuration.items():
            if key in ['train', 'test', 'validation']:
                if mode == key:
                    new_config.update(value)
            else:
                new_config[key] = value
        configuration = new_config
    return configuration


def load_config(args):
    '''the configuration of each model can rewrite the base configuration'''
    # base config
    base_configuration = load_base_config()

    # load one model config
    if args.get('mode'):
        configuration = load_model_config(args['stage'], args['mode'])

        # update and append the special config for base config
        base_configuration.update(configuration)
    configuration = base_configuration
    return configuration


def load_base_config():
    base_path = os.path.dirname(os.path.realpath(__file__))
    # config_path = f'base.yaml'
    config_path = f'{base_path}/base.yaml'
    with open(config_path) as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)
    print(f'[!] load base configuration: {config_path}')
    return configuration
