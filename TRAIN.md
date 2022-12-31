# DPINet Training Tutorial

This implements training of DPINet.
### Add DPINet to your PYTHONPATH
```bash
export PYTHONPATH=/path/to/DPINet:$PYTHONPATH
```

## Prepare training dataset
DPINet is trained using the following datasets. The pre-processed datasets compatible to [pysot](https://github.com/STVIR/pysot) are [here](https://pan.baidu.com/s/1_Rg6dKhHUSI5LC0E6ae02w) (code: nbp4). All datasets should be listed in [training_dataset](training_dataset) directory.
* [VID](http://image-net.org/challenges/LSVRC/2017/)

* [YOUTUBEBB](https://research.google.com/youtube-bb/)

* [DET](http://image-net.org/challenges/LSVRC/2017/)

* [COCO](http://cocodataset.org)

* [GOT10K](http://got-10k.aitestunion.com/)

* [LASOT](https://cis.temple.edu/lasot/)



## Download pretrained backbones
Download pretrained backbones from [here](https://drive.google.com/drive/folders/1DuXVWVYIeynAcvt9uxtkuleV6bs6e3T9) and put them in `pretrained_models` directory

## Training

To train a model, run `train.py` with the desired configs:

```bash
cd experiments/vot
```

#### Single GPU:
```bash
python ../../tools/train.py --cfg config.yaml
```

## Testing
```bash 
python ../../tools/test.py \
        --snapshot shnapshot/checkpoint_e20.pth \
	--config config.yaml \
	--dataset VOT2018
```

## Evaluation

```bash
python ../../tools/eval.py 	 \
	--dataset VOT2018        \ # dataset name
```

## Hyper-parameter Search

The tuning toolkit will not stop unless you do.

```bash
python ../../tools/tune.py  \
    --dataset VOT2018  \
    --snapshot snapshot/checkpoint_e20.pth  \
```

