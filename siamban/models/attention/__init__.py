from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from siamban.models.attention.non_local import AttnAllLayer

ATTNS={
    'attnalllayer':AttnAllLayer,
}

def get_attn(name, **kwargs):
    return ATTNS[name](**kwargs)