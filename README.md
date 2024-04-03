# PercepTreeV1


Official code repository for the papers:

<div align="left">
  <img width="100%" alt="DINO illustration" src=".github/figure6.png">
</div>

- [Tree Detection and Diameter Estimation Based on Deep Learning](https://academic.oup.com/forestry/advance-article-abstract/doi/10.1093/forestry/cpac043/6779886?utm_source=advanceaccess&utm_campaign=forestry&utm_medium=email), published in *Forestry: An International Journal Of Forest Research*. Preprint version (soon).

<div align="left">
  <img width="100%" alt="DINO illustration" src=".github/detection_synth.jpg">
</div>

- [Training Deep Learning Algorithms on Synthetic Forest Images for Tree Detection](http://arxiv.org/abs/2210.04104), presented at *ICRA 2022 IFRRIA Workshop*. The video presentation is [available](https://www.youtube.com/watch?v=8KT97ZFMC0g&list=PLbiomSAe-K8896UHcLVkNWP66DaFpS7j5&index=2).



<!-- The version 1 of this project is done using synthetic forest dataset `SynthTree43k`, but soon we will release models fine-tuned on real-wolrd images. Plans to release SynthTree43k are underway.

The gif below shows how well the models trained on SynthTree43k transfer to real-world, without any fine-tuning on real-world images. -->
<!-- <div align="center">
  <img width="100%" alt="DINO illustration" src=".github/pred_synth_to_real.gif">
</div> -->

## Datasets
All our datasets are made available to increase the adoption of deep learning for many precision forestry problems.

<table>
  <tr>
    <th>Dataset name</th>
    <th>Description</th>
    <th>Download</th>
  </tr>
  <tr>
    <td>SynthTree43k</td>
    <td>A dataset containing 43 000 synthetic images and over 190 000 annotated trees. Includes images, train, test, and validation splits. (84.6 GB) </td>
    <td><a href="http://norlab.s3.valeria.science/SynthTree43k.zip?AWSAccessKeyId=VCI7FLOHYPGLOOOAH0S5&Expires=2274019241&Signature=KfOgwrHX8WHejopspqQ8XMwlMJE%3D">S3 storage</a></td>
  <tr>
  <tr>
    <td>SynthTree43k</td>
    <td>Depth images.</td>
    <td><a href="https://ulavaldti-my.sharepoint.com/:u:/g/personal/vigro7_ulaval_ca/EfglPMp555FGvwKGDEp9eRwBn_jXK-7vMPfYxDAVHbzTgg?e=u3rIb5">OneDrive </a></td>
  <tr>
  <tr>
    <td>CanaTree100</td>
    <td>A dataset containing 100 real images and over 920 annotated trees collected in Canadian forests. Includes images, train, test, and validation splits for all five folds.</td>
    <td><a href="https://ulavaldti-my.sharepoint.com/:u:/g/personal/vigro7_ulaval_ca/EdxLqaVszr9LnSAcMaKnZtcBxLD19RY_yyJnrNzXZXU6sw?e=b8hI1G">OneDrive </a></td>
  <tr>
</table>

The annotations files are already included in the download link, but some users requested the annotations for entire trees:
<a href="https://drive.google.com/file/d/1AZUtdrNJGPWgqEwUrRin6OKwE_KGavZq/view?usp=sharing">train_RGB_entire_tree.json</a>,
<a href="https://drive.google.com/file/d/1doTRoLvQ1pGaNb75mx-SOr5aEVBLNnZe/view?usp=sharing">val_RGB_entire_tree.json</a>,
<a href="https://drive.google.com/file/d/1ZMYqFylSrx2KDHR-2TSoXFq-_uoyb6Qp/view?usp=share_link">test_RGB_entire_tree.json</a>.
Beware that it can result in worse detection performance (in my experience), but maybe there is something to do with models not based on RPN (square ROIs), such as <a href="https://github.com/facebookresearch/Mask2Former">Mask2Former</a>.

## Pre-trained models
Pre-trained models weights are compatible with Detectron2 config files.
All models are trained on our synthetic dataset SynthTree43k.
We provide a demo file to try it out.

### Mask R-CNN trained on synthetic images (`SynthTree43k`)
<table>
  <tr>
    <th>Backbone</th>
    <th>Modality</th>
    <th>box AP50</th>
    <th>mask AP50</th>
    <th colspan="6">Download</th>
  </tr>
  <tr>
    <td>R-50-FPN</td>
    <td>RGB</td>
    <td>87.74</td>
    <td>69.36</td>
    <td><a href="https://drive.google.com/file/d/1pnJZ3Vc0SVTn_J8l_pwR4w1LMYnFHzhV/view?usp=sharing">model</a></td>
  <tr>
    <td>R-101-FPN</td>
    <td>RGB</td>
    <td>88.51</td>
    <td>70.53</td>
    <td><a href="https://drive.google.com/file/d/1ApKm914PuKm24kPl0sP7-XgG_Ottx5tJ/view?usp=sharing">model</a></td>
  <tr>
    <td>X-101-FPN</td>
    <td>RGB</td>
    <td>88.91</td>
    <td>71.07</td>
    <td><a href="https://drive.google.com/file/d/1Q5KV5beWVZXK_vlIED1jgpf4XJgN71ky/view?usp=sharing">model</a></td>
  </tr>
  <tr>
    <td>R-50-FPN</td>
    <td>Depth</td>
    <td>89.67</td>
    <td>70.66</td>
    <td><a href="https://drive.google.com/file/d/1bnH7ZSXWoOJx5AkbNeHf_McV46qiKIkY/view?usp=sharing">model</a></td>
  <tr>
    <td>R-101-FPN</td>
    <td>Depth</td>
    <td>89.89</td>
    <td>71.65</td>
    <td><a href="https://drive.google.com/file/d/1DgMscnTIGty7y9-VNcq1zERrevfT3b_L/view?usp=sharing">model</a></td>
  <tr>
    <td>X-101-FPN</td>
    <td>Depth</td>
    <td>87.41</td>
    <td>68.19</td>
    <td><a href="https://drive.google.com/file/d/1rsCbLSvFf2I47FJK4vhhv0du5uCV6zjO/view?usp=sharing">model</a></td>
  </tr>
</table>

### Mask R-CNN finetuned on real images (`CanaTree100`)
<table>
  <tr>
    <th>Backbone</th>
    <th>Description</th>
    <th colspan="6">Download</th>
  </tr>
  <tr>
    <td>X-101-FPN</td>
    <td>Trained on fold 01, good for inference.</td>
    <td><a href="https://drive.google.com/file/d/108tORWyD2BFFfO5kYim9jP0wIVNcw0OJ/view?usp=sharing">model</a></td>
  </tr>
</table>

## Demos
Once you have a working Detectron2 and OpenCV installation, running the demo is easy.

### Demo on a single image
- Download the pre-trained model weight and save it in the `/output` folder (of your local PercepTreeV1 repos).
-Open `demo_single_frame.py` and uncomment the model config corresponding to pre-trained model weights you downloaded previously, comment the others. Default is X-101. Set the `model_name` to the same name as your downloaded model ex.: 'X-101_RGB_60k.pth'
- In `demo_single_frame.py`, specify path to the image you want to try it on by setting the `image_path` variable.

### Demo on video
- Download the pre-trained model weight and save it in the `/output` folder (of your local PercepTreeV1 repos).
-Open `demo_video.py` and uncomment the model config corresponding to pre-trained model weights you downloaded previously, comment the others. Default is X-101.
- In `demo_video.py`, specify path to the video you want to try it on by setting the `video_path` variable.

<div align="left">
  <img width="70%" alt="DINO illustration" src=".github/trailer_0.gif">
</div>

# Bibtex
If you find our work helpful for your research, please consider citing the following BibTeX entry.   
```bibtex
@article{grondin2022tree,
    author = {Grondin, Vincent and Fortin, Jean-Michel and Pomerleau, François and Giguère, Philippe},
    title = {Tree detection and diameter estimation based on deep learning},
    journal = {Forestry: An International Journal of Forest Research},
    year = {2022},
    month = {10},
}

@inproceedings{grondin2022training,
  title={Training Deep Learning Algorithms on Synthetic Forest Images for Tree Detection},
  author={Grondin, Vincent and Pomerleau, Fran{\c{c}}ois and Gigu{\`e}re, Philippe},
  booktitle={ICRA 2022 Workshop in Innovation in Forestry Robotics: Research and Industry Adoption},
  year={2022}
}
```
