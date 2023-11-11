# Analysis of Deep Image Colourization Techniques
## Authors
1. Harsh Rao Dhanyamraju [@HarshRaoD](https://github.com/HarshRaoD)
2. Ang Boon Leng [@jimmysqqr](https://github.com/jimmysqqr)
3. Kshitij Parashar [@xitij27](https://github.com/xitij27)

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

## Inference
### Setup
1. Please download the model checkpoints
2. Create a new virtual environment
3. ```pip install -r requierments.txt```

