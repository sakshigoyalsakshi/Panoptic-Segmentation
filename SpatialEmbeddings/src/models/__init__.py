from models.BranchedERFNet import BranchedERFNet

def get_model(name, num_classes, encoder=None):
    if name == "branched_erfnet":
        model = BranchedERFNet(num_classes, encoder)
        return model
    else:
        raise RuntimeError("model \"{}\" not available".format(name))