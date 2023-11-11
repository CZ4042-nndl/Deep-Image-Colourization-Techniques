# Analysis of Deep Image Colourization Techniques
![](https://github.com/CZ4042-nndl/Deep-Koalaizer/blob/Submission/Diagrams/Trees.png)
## Authors
1. Harsh Rao Dhanyamraju [@HarshRaoD](https://github.com/HarshRaoD)
2. Ang Boon Leng [@jimmysqqr](https://github.com/jimmysqqr)
3. Kshitij Parashar [@xitij27](https://github.com/xitij27)

## Inference
### Setup
1. Please download the model checkpoints [here](https://entuedu-my.sharepoint.com/:f:/g/personal/harshrao001_e_ntu_edu_sg/EnyTSKVK_uVAhFunaIadof4BFGhEGOvaf7dbDtRv_arS1w?e=9Fnp2i)
2. Create a new virtual environment
3. ```pip install -r requierments.txt```

### Running Tests
1. Make sure you have Trained_Colourization_Models.py and Model_Testing.ipynb in the same directory
2. You dont need to download the Coco Datset for inference you can use the Images in the Sample_Images Directory
3. Start a Jupyter Server and Model_Testing.ipynb
4. Create a Custom Dataset:
```
import Trained_Colourization_Models as tcm
test_dataset = tcm.CustomDataset(<Your-Path-here>,'test')
```
5. Load the testing Image
```
ti2 = tcm.Testing_Image(test_dataset, filename=<Your-file-name-here>)
# file name should be in the same directory as specified in test_dataset
```
6. Load a Model Runner and generate output by passing the Testing_Image Object
```
model_runner = tcm.Default_Model_Runner()
output_img = model_runner.get_image_output(ti2)
```
7. Visualise the output
```
plt.imshow(output_img)  # For model outputs
plt.imshow(ti2.get_gray(), cmap='gray')  # For Input Images
plt.imshow(ti2..get_rgb())  # For ground truth Images
```
## Training
1. Create a new virtual environment
2. ```pip install -r requierments.txt```
3. Navigate to the directory with the training script ```cd Training_Scripts/<experiment_dir>```
4. Create a new directory ```/Models``` to store the model checkpoints created during training
5. Change the necessary configurations in the `Configuration` class and the data paths in the code
   1. set `load_model_to_train = False`
   2. set data path in `CustomDataset`
6. Run the file to begin training ```python training_script.py```

### Resume Training
1. Navigate to the directory with the training script ```cd Training_Scripts/<experiment_dir>```
2. Change the necessary configurations in the `Configuration` class and the data paths in the code
   1. `load_model_file_name`: path to checkpoint file
   2. set `load_model_to_train = True`
   3. set `starting_epoch = (current checkpoint epoch + 1)`
   4. set data path in `CustomDataset`
3. Run the file to resume training ```python training_script.py```

## Downloading COCO Dataset
1. You can run DataAnalysis.ipynb to download the COCO Dataset and reduce the size of all images (to get training data)
2. You'll need a valid kaggle api key and atleast 55 GB of free space
