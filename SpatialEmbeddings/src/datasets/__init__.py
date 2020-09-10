from datasets.CityscapesDataset import CityscapesDataset


def get_dataset(name, root_dir, type, class_id, size, transform):
    if name == "cityscapes":
        return CityscapesDataset(root_dir, type, class_id, size, transform)
    else:
        raise RuntimeError("Dataset {} not available".format(name))
