train.py will be an educational sample that will be asy to follow 
without too much overzealous error handling, unnecessary comments, etc.

However, it will also be functional in a sense that it will actually fit
a model to data effectively.

Create train.py with the following requirements:
- Make the code in style of quantize.py
- Make the code style in a way that it will be easy to integrate with sagemager-run.py later (do not eactually implement) 
- Data will bee stored in ../data/ by default.
- We will use MobileNetV2 keras model instantiated as base.
- We will pick an adequate subset of imagenet-val data from ../data/imagenetval/<n9999100x for validation
- Ensure that we pick up *.jpg, *.webp, *.avif or *.png as well.
- We will have new set classes already added into ImageNetLabels.txt
- The new classes will be recognized by the model. 
- Training images will be at ../data/train/<synset_id>/*.jpg, *.webp, *.avif or *.png
- Adequate care should be taken to ensure that the new classes are not hurting the accuracy for existing classes significantly 
due to the new classes being fitted
- the fine-tuned model will be saved as ../data/mobilenetv2-finetuned.keras (default name in default model-dir), but cli will accept a new name and model dir
- train.py must apply heavy augmentation specifically for new-class images:
random rotation (±30°), brightness/contrast jitter, random crop+resize, horizontal flip, slight color shift, and random background noise/blur. 
Effectively multiply the training set ~10–20x per epoch.
- train.py should print a warning if counts are below 10 images for any class,

For this revision, we will introduce two new classes:
- Development Board (development board) class (1-based) index 1001 - synset id: n99991001
- STM32 MP135f-DK (stm32-mp135f-dk) class index 1002 - synset id: n99991002
- Note: ImageNetLabels.txt index 0 is "background" so it is compatible with 1001-class models


