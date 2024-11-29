import yaml


def read_yaml(path):
    with open(path, 'r') as stream:
        return yaml.safe_load(stream)


def write_yaml(data, path, file_name):
    with open(f'{path}/{file_name}', 'w') as stream:
        yaml.safe_dump(data, stream)
