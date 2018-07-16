#  Image-to-Image Translation with Conditional Adversarial Nets
An implementation of pix2pix described in the paper using tensorflow.
* [ Image-to-Image Translation with Conditional Adversarial Nets](https://arxiv.org/pdf/1611.07004.pdf)

Published in CVPR 2017, written by P. Isola, J-Y Zhu, T. Zhou and A. Efros

## Requirement
- Python 3.6.4
- Tensorflow 1.8.0 
- Pillow 5.0.0
- numpy 1.14.5

## How to Run
If you want to see more example commands, please refer to run.sh file in my repository.

Train

```
python main.py --tr_data_path ./dataSet/facades/train --transfer_type B_to_A --model_name facades
```

test

```
python main.py --mode test --val_data_path ./dataSet/facedes/val --transfer_type B_to_A --load_size 256 --pre_trained_model ./model/pix2pix-facades
```

## Datasets
Shell script which downloads datasets will be uploaded soon!

## Pre-trained model
Pre-trained model will be uploaded soon!


## Experimental Results

Experimental results on map dataset(map to ariel)

| Input | result | Target |
| --- | --- | --- |
| <img src="images/map2ariel/71_input.png" width="256px"> |<img src="images/map2ariel/71_gene.png" width="256px"> | <img src="images/map2ariel/71_GT.png" width="256px"> |
| <img src="images/map2ariel/122_input.png" width="256px"> | <img src="images/map2ariel/122_gene.png" width="256px"> | <img src="images/map2ariel/122_GT.png" width="256px"> |
| <img src="images/map2ariel/611_input.png" width="256px"> | <img src="images/map2ariel/611_gene.png" width="256px"> | <img src="images/map2ariel/611_GT.png" width="256px"> |
| <img src="images/map2ariel/613_input.png" width="256px"> | <img src="images/map2ariel/613_gene.png" width="256px"> | <img src="images/map2ariel/613_GT.png" width="256px"> |

Experimental results on map dataset(ariel to map)

| Input | result | Target |
| --- | --- | --- |
| <img src="images/ariel2map/3_input.png" width="256px"> | <img src="images/ariel2map/3_gene.png" width="256px"> | <img src="images/ariel2map/3_GT.png" width="256px"> |
| <img src="images/ariel2map/6_input.png" width="256px"> | <img src="images/ariel2map/6_gene.png" width="256px"> | <img src="images/ariel2map/6_GT.png" width="256px"> |
| <img src="images/ariel2map/228_input.png" width="256px"> | <img src="images/ariel2map/228_gene.png" width="256px"> | <img src="images/ariel2map/228_GT.png" width="256px"> |
| <img src="images/ariel2map/501_input.png" width="256px"> | <img src="images/ariel2map/501_gene.png" width="256px"> | <img src="images/ariel2map/501_GT.png" width="256px"> |

Experimental results on facades dataset

| Input | result | Target |
| --- | --- | --- |
| <img src="images/facades/58_input.png" width="256px"> | <img src="images/facades/58_gene.png" width="256px"> | <img src="images/facades/58_GT.png" width="256px"> |
| <img src="images/facades/59_input.png" width="256px"> | <img src="images/facades/59_gene.png" width="256px"> | <img src="images/facades/59_GT.png" width="256px"> |
| <img src="images/facades/75_input.png" width="256px"> | <img src="images/facades/75_gene.png" width="256px"> | <img src="images/facades/75_GT.png" width="256px"> |
| <img src="images/facades/89_input.png" width="256px"> | <img src="images/facades/89_gene.png" width="256px"> | <img src="images/facades/89_GT.png" width="256px"> |


Experimental results on edges2shoes dataset

| Input | result | Target |
| --- | --- | --- |
| <img src="images/edges2shoes/4_AB_input.png" width="256px"> | <img src="images/edges2shoes/4_AB_gene.png" width="256px"> | <img src="images/edges2shoes/4_AB_GT.png" width="256px"> |
| <img src="images/edges2shoes/158_AB_input.png" width="256px"> | <img src="images/edges2shoes/158_AB_gene.png" width="256px"> | <img src="images/edges2shoes/158_AB_GT.png" width="256px"> |
| <img src="images/edges2shoes/31_AB_input.png" width="256px"> | <img src="images/edges2shoes/31_AB_gene.png" width="256px"> | <img src="images/edges2shoes/31_AB_GT.png" width="256px"> |
| <img src="images/edges2shoes/199_AB_input.png" width="256px"> | <img src="images/edges2shoes/199_AB_gene.png" width="256px"> | <img src="images/edges2shoes/199_AB_GT.png" width="256px"> |


## Comments
If you have any questions or comments on my codes, please email to me. [son1113@snu.ac.kr](mailto:son1113@snu.ac.kr)

### Reference
[1] https://github.com/phillipi/pix2pix
