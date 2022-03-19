# Torchism 
General template for my PyTorch projects.
I will add more examples other than supervised mnist in the future.

## Dependency 
```
pip install -r requirements.txt
```

## Sanity run 
Try using the script in `data_generator` or download mnist data (.csv) from `https://github.com/pjreddie/mnist-csv-png` and save it same as the paths in `configs/train/sample.yaml`. Then run:
```
python train.py --config configs/train/sample.yaml --gpus 0
tensorboard --logdir=runs 
```