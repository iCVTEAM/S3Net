from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .dogs import dogs
from .cub import cub
from .car import car
__imgfewshot_factory = {
        'dogs': dogs,
        'cub': cub,
        'car': car,
}


def get_names():
    return list(__imgfewshot_factory.keys()) 


def init_imgfewshot_dataset(name, **kwargs):
    if name not in list(__imgfewshot_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__imgfewshot_factory.keys())))
    return __imgfewshot_factory[name](**kwargs)

