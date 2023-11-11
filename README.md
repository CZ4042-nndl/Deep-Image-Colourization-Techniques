# Analysis of Deep Image Colourization Techniques
## Authors
1. Harsh Rao Dhanyamraju [@HarshRaoD](https://github.com/HarshRaoD)
2. Ang Boon Leng [@jimmysqqr](https://github.com/jimmysqqr)
3. Kshitij Parashar [@xitij27](https://github.com/xitij27)

## Inference
### Setup
1. Please download the model checkpoints
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
