# Codes for *Paraphrasing Is All You Need  for Novel Object Captioning*

### Installation
- Follow the instruction from here: [link](https://github.com/microsoft/Oscar/blob/master/INSTALL.md) to set up the environment of our codebase, and replace the directory oscar/ with p2c/
- Install CLIP from the offical repo: [link](https://github.com/openai/CLIP), and place the CLIP directory under the root directory
- Download the pre-trained CLIP model and save it using the following code: `torch.save(_use_new_zipfile_serialization=False)`
Then replace line 381 in CLIP/clip/model.py with this line:
`for attr in [*[f"{s}_proj_weight" for s in ["in"]], "in_proj_bias"]:`
- Install the object detection model TSD from here: [link](https://github.com/Sense-X/TSD)

### Data
- It is recommended to download large files with AzCopy for faster speed
```
path/to/azcopy copy <link> <target folder> --recursive
```
- For COCO dataset, download the region features and detection tags from here: [link](https://biglmdiag.blob.core.windows.net/vinvl/datasets/coco_caption)
- Download the testing data for nocaps from here: [link](https://biglmdiag.blob.core.windows.net/vinvl/datasets/nocaps) 
Note that this link only provides data in validation set. For testing set, please extract them from the data below.
- For Open Images Dataset, download the region features from here: [link](https://biglmdiag.blob.core.windows.net/vinvl/image_features/oi_X152C4_frcnnbig2_exp168model_0060000model.roi_heads.nm_filter_2_model.roi_heads.score_thresh_0.2/model_0060000/)
- For Open Images Dataset, use the TSD model to generate the detection tags.
- Parse the region features and detection tags into the same format as the COCO dataset, and place the generated files in p2c/data. 
Codes for generating the tsv file can be found here: [link](https://github.com/microsoft/scene_graph_benchmark/blob/main/tools/mini_tsv/tsv_demo.py)

### Training
#### P2C:
- To perform VIVO pre-training
```
$ bash scripts/vivo.sh
```
- To perform our stage 1 training (Describing novel objects with linguistic fluency)
```
$ bash scripts/stage1.sh
```
- To perform our stage 2 training (Learning novel object captions with fidelity and adequacy)
```
$ bash scripts/stage2.sh
```

#### Reproducing VinVL+VIVO:
- To perform VIVO pre-training
```
$ bash scripts/vivo.sh
```
- To perform cross-entropy optimization
```
$ bash scripts/xe.sh
```
- To perform SCST optimization
```
$ bash scripts/scst.sh
```


### Testing
To evaluate the result on the testing set, replace val.yaml with test.yaml
- To perform greedy decoding
```
$ bash scripts/inference.sh --num_beams 1 --test_yaml data/val.yaml
```
- To perform beam search
```
$ bash scripts/inference.sh --num_beams 5 --test_yaml data/val.yaml
```
- To perform Constrained Beam Search (CBS)
```
$ bash scripts/cbs.sh --test_yaml data/val.yaml
```
