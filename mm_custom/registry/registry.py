

from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine.registry import METRICS as MMENGINE_METRICS

from mmengine.registry import Registry




MODELS = Registry('model', parent=MMENGINE_MODELS, locations=['mm_custom.models'])
DATASETS = Registry('dataset', parent=MMENGINE_DATASETS, locations=['mm_custom.datasets'])
TRANSFORMS = Registry('transform', parent=MMENGINE_TRANSFORMS, locations=['mm_custom.datasets.transforms'])
METRICS = Registry('metric', parent=MMENGINE_METRICS, locations=['mm_custom.evaluation'])



