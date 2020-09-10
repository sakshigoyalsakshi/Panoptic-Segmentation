# Real-Time Proposal-Free Panoptic Segmentation 

This code uses **PyTorch** for training and evaluating the **Pan-SECB** model for Panoptic Segmentation.

# Prerequisites

* Pytorch 1.1
* Python 3.6.8 (or higher)
* [Cityscapes](https://www.cityscapes-dataset.com/) + [Scripts](https://github.com/mcordts/cityscapesScripts/tree/master/cityscapesscripts/preparation) (to evaluate the model)

## Training

Training consists out of 2 steps. We first train on 512x512 crops around each object, to avoid computation on background patches. Afterwards, we finetune on larger patches (1024x1024) to account for bigger objects and background features which are not present in the smaller crops.

To generate these crops do the following:
```
$ CITYSCAPES_DIR=/path/to/cityscapes/ python utils/generate_crops.py
```
This will generate the crops of each of the 'things' classes in the given directory with the extension crops_{class_id}. E.g. crops_26 for "Cars".

Afterwards start training on crops:
```
$ CITYSCAPES_DIR=/path/to/cityscapes/ python train.py`
```
For fine tuning modify the train_config as follows:
* Set the resume_path to latest training checkpoint.
* Change on_crops to False.
* Change size to 1024x1024.
* Change batch_size to 2.
* Increase the n_epochs to 250
* Switch the foreground weights for fine tuning

Then train the model using the same command as above.

## Testing
For testing, change the `checkpoint_path` in test_config file to the latest training checkpoint. 
###  For Accuracy
To test the accuracy of the model on the Cityscapes validation set run:
```
$ CITYSCAPES_DIR=/path/to/cityscapes/ python test.py
```
This will save the images in a format that is required for the cityscapes preparation scripts for panoptic segmentation.
 
 To evaluate Panoptic Quality (PQ) run these two following scripts:
 * [createPanopticImgs.py](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createPanopticImgs.py "createPanopticImgs.py") . It converts the *instanceIds.png annotations of the Cityscapes dataset to COCO-style panoptic segmentation format.
 * [evalPanopticSemanticLabeling.py](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalPanopticSemanticLabeling.py "evalPanopticSemanticLabeling.py"). The evaluation script for panoptic segmentation.

### For Runtime
To evaluate the runtime of the model run the following:
```
$ CITYSCAPES_DIR=/path/to/cityscapes/ python Forward_time.py
```
