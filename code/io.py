import csv
import json
import os

import dill

def load_structured_resource(filename, filetype='', **kwargs):
    if (filename.endswith('.dill') or filetype == 'dill'):
        with open(filename, 'rb') as data_file:
            vectors = dill.load(data_file)
        return vectors
    elif (filename.endswith('.json') or filetype == 'json'):
        with open(filename, 'r') as data_file:
            vectors = json.load(data_file)
        return vectors
    elif (filename.endswith('.jsonl') or filetype == 'jsonl'):
        data = []
        with open(filename) as data_file:
            for line in data_file:
                data.append(json.loads(line))
        return data
    elif (filename.endswith('.csvl') or filetype == 'csvl'):
        as_set = kwargs.pop('as_set', True)
        with open(filename) as in_file:
            data = in_file.read().strip().split(',')
        return set(data) if as_set else data
    else:
        raise NotImplementedError


def save_structured_resource(obj, out_file, filetype='', create_intermediary_dirs=True, **kwargs):
    path, _ = os.path.split(out_file)

    if (path != '' and not os.path.exists(path) and create_intermediary_dirs):
        os.makedirs(path)

    if (out_file.endswith('.dill') or filetype == 'dill'):
        with open(out_file, 'wb') as data_file:
            dill.dump(obj, data_file, protocol=kwargs.get('dill_protocol', 3))
    elif (out_file.endswith('.json') or filetype == 'json'):
        with open(out_file, 'w', encoding=kwargs.get('encoding', 'utf-8')) as data_file:
            json.dump(obj, data_file, indent=kwargs.get('json_indent', 4), ensure_ascii=kwargs.get('ensure_ascii', True))
    elif (out_file.endswith('.jsonl') or filetype == 'jsonl'):
        if (isinstance(obj, list)):
            with open(out_file, 'w', encoding=kwargs.get('encoding', 'utf-8')) as data_file:
                for line in obj:
                    data_file.write(f'{json.dumps(line)}\n')

        else:
            raise ValueError(f'obj must be of type `list` but is of type={type(obj)}!')
    elif (out_file.endswith('.csvl') or filetype == 'csvl'):
        with open(out_file, 'w', encoding=kwargs.get('encoding', 'utf-8')) as data_file:
            data_file.write(','.join(obj))
    else:
        raise NotImplementedError


def save_pytorch_model(model, out_file, state_only=False):
    if (state_only):
        torch.save(model.state_dict(), out_file)
    else:
        torch.save(model, out_file)


def load_pytorch_model(filename):
    return torch.load(filename)
