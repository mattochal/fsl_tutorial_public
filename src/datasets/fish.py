from datasets.dataset_utils import load_dataset_from_pkl, load_dataset_from_from_folder 
from datasets.dataset_template import ColorDatasetInMemory, ColorDatasetOnDisk
import os

def get_FishRecognition(args_per_set, setnames=["train", "val", "test"]):
    """
    Returns FishRecognition datasets.
    """
    datasets = {}
    for setname in setnames:
        args = args_per_set[setname]
        version = args.dataset_version

        if version not in [None, "cache"]:
            raise Exception("Fish Recognition version '{}' not found.".format(version))
        if version is None:
            version = "cache"
        data_path = os.path.abspath(args.data_path)
        filepath = os.path.join(data_path, "fish", "fish-{0}-{1}.pkl".format(version,setname))
        data = load_dataset_from_pkl(filepath)
        dataset_class = ColorDatasetInMemory
            
        datasets[setname] = [data['image_data'], data['class_dict'], args, dataset_class]
        
    return datasets
    