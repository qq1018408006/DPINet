# DPINet

This project hosts the code for implementing the DPINet algorithm for visual tracking. 

The raw results are [here](https://drive.google.com/file/d/164EIV3Zhu6kX7Yxs4hRQ25J9GAYmAM9j/view?usp=sharing). The code based on the [PySOT](https://github.com/STVIR/pysot).



## Installation

Please find installation instructions in [`INSTALL.md`](INSTALL.md).

## Quick Start: Using DPINet

### Add DPINetto your PYTHONPATH

```bash
export PYTHONPATH=/path/to/DPINet:$PYTHONPATH
```

### Download models

Download models from [here]() and put the `model.pth` in the correct directory in experiments

### Webcam demo

```bash
python tools/demo.py \
    --config experiments/vot/config.yaml \
    --snapshot experiments/vot/model.pth
    # --video demo/bag.avi # (in case you don't have webcam)
```

### Download testing datasets

Download datasets and put them into `testing_dataset` directory. Jsons of commonly used datasets can be downloaded from [here](https://github.com/Giveupfree/SOTDrawRect/tree/main/SOT_eval). If you want to test tracker on new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to setting `testing_dataset`. 

### Test tracker

```bash
cd experiments/vot
python -u ../../tools/test.py 	\
	--snapshot model.pth 	\ # model path
	--dataset VOT2018 	\ # dataset name
	--config config.yaml	  # config file
```

The testing results will in the current directory(results/dataset/model_name/)

### Eval tracker

assume still in `experiments/vot`

``` bash
python ../../tools/eval.py 	 \
	--tracker_path ./results \ # result path
	--dataset VOT2018        \ # dataset name
	--num 1 		 \ # number thread to eval
	--tracker_prefix 'model'   # tracker_name
```

###  Training :wrench:

See [TRAIN.md](TRAIN.md) for detailed instruction.

## License

This project is released under the [Apache 2.0 license](LICENSE). 
