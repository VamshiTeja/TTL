import numpy as np
from easydict import EasyDict as edict

root = edict()


#training settings
root.run_label = 'transfer-learning'
root.gpu = '0'
root.dataset = 'svhn'
root.batch_size = 512
root.batch_size_test = 512
root.learning_rate = 0.0002
root.display_step = 20
root.checkpoint_step = 5
root.task = "classification"
root.z_dim = 10
root.restore = 0
root.num_epochs = 2
root.transfer = False
root.source_task = "autoencoding"
root.target_task = "classification"
root.mode = "test"
root.remove_dims = ''
root.num_test_runs = 10
root.gamma = 3.5

#root.seed = 99
cfg = root

def _merge_a_into_b(a, b):
    """
    Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """
    Load a config file and merge it into the default options.
    """
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, root)

