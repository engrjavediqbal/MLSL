"""
utilities for convenience
"""
import contextlib
import h5py
import logging
import os.path as osp
import yaml
from cStringIO import StringIO
from PIL import Image

import numpy as np


cfg = {}


def as_list(obj):
    """A utility function that treat the argument as a list.

    Parameters
    ----------
    obj : object

    Returns
    -------
    If `obj` is a list, return it. Otherwise, return `[obj]` as a single-element list.
    """
    if isinstance(obj, list):
        return obj
    else:
        return [obj]

def get_interp_method(imh_src, imw_src, imh_dst, imw_dst, default=Image.CUBIC):
    if not cfg.get('choose_interpolation_method', False):
        return default
    if imh_dst < imh_src and imw_dst < imw_src:
        return Image.ANTIALIAS
    elif imh_dst > imh_src and imw_dst > imw_src:
        return Image.CUBIC
    else:
        return Image.LINEAR

def h5py_save(to_path, *data):
    with h5py.File(to_path, 'w') as f:
        for i, datum in enumerate(data):
            f.create_dataset('d{}'.format(i), data=datum)
            
def h5py_load(from_path):
    data = []
    if osp.isfile(from_path):
        with h5py.File(from_path) as f:
            for k in f.keys():
                data.append(f[k][()])
    return tuple(data)

def load_image_with_cache(path, cache=None):
    if cache is not None:
        if not cache.has_key(path):
            with open(path, 'rb') as f:
                cache[path] = f.read()
        return Image.open(StringIO(cache[path]))
    return Image.open(path)

@contextlib.contextmanager
def np_print_options(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)

def read_cfg(cfg_file, cfg_info=None):
    if cfg_file is not None:
        print 'Read config file {}'.format(cfg_file)
        with open(cfg_file) as f:
            cfg_info = yaml.load(f)
    return cfg_info

def set_logger(output_dir=None, log_file=None, debug=False):
    head = '%(asctime)-15s Host %(message)s'
    logger_level = logging.INFO if not debug else logging.DEBUG
    if all((output_dir, log_file)) and len(log_file) > 0:
        logger = logging.getLogger()
        log_path = osp.join(output_dir, log_file)
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(head)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logger_level)
    else:
        logging.basicConfig(level=logger_level, format=head)
        logger = logging.getLogger()
    return logger

