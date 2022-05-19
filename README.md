<div align="center">    
 
# Deep Image Hashing     

[//]: # ([![Paper]&#40;http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg&#41;]&#40;https://www.nature.com/articles/nature14539&#41;)

[//]: # ([![Conference]&#40;http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg&#41;]&#40;https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018&#41;)

[//]: # ([![Conference]&#40;http://img.shields.io/badge/ICLR-2019-4b44ce.svg&#41;]&#40;https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018&#41;)

[//]: # ([![Conference]&#40;http://img.shields.io/badge/AnyConference-year-4b44ce.svg&#41;]&#40;https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018&#41;  )

[//]: # (<!--)

[//]: # (ARXIV   )

[//]: # ([![Paper]&#40;http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg&#41;]&#40;https://www.nature.com/articles/nature14539&#41;)

[//]: # (-->)

[//]: # (![CI testing]&#40;https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push&#41;)

[//]: # ()
[//]: # ()
[//]: # (<!--  )

[//]: # (Conference   )

[//]: # (-->   )
</div>
 
## Description   
Implementation of state-of-the-art deep hashing methods   

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/bgswaroop/deep-hashing

# install project   
cd deep-hashing 
pip install -e .   
pip install -r requirements.txt
 ```   
Next, navigate to any file and run it.   
 ```bash
# module folder
cd project

# run module   
python sota_2016_CVPR_DSH.py    
```

## SOTA deep hashing
#### Implementation of other state-of-the-art deep hashing methods in PyTorch lightning

The following table summarizes the provided implementations of other deep hashing methods 
along with their performance on popular datasets. The evaluation metric is mAP (mean average precision)   


<table>
    <thead>
        <tr>
            <th rowspan=2>Supervised Methods</th>
            <th colspan=2>CIFAR-10</th>
        </tr>
        <tr>
            <th>12-bit</th>
            <th>48-bit</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Deep supervised hashing for fast image retrieval <br/> <b>DSH</b> (<a href="https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_Deep_Supervised_Hashing_CVPR_2016_paper.pdf">paper</a>, <a href="https://github.com/bgswaroop/deep-hashing/blob/master/project/sota_2016_CVPR_DSH.py">code</a>) - CVPR 2016 </td>
            <td>0.6249<br/><small>scratch</small></td>
            <td>0.7794<br/><small>fine-tune</small></td>
        </tr>
        <tr>
            <td>Deep supervised discrete hashing <br/> <b>DSDH</b> (<a href="https://proceedings.neurips.cc/paper/2017/file/e94f63f579e05cb49c05c2d050ead9c0-Paper.pdf">paper</a>, <a href="https://github.com/bgswaroop/deep-hashing/blob/master/project/sota_2017_NIPS_DSDH.py">code</a>) - NIPS 2017 </td>
            <td></td>
            <td></td>
        </tr>
    </tbody>
</table>

[//]: # (### Citation   )

[//]: # (```)

[//]: # (@article{YourName,)

[//]: # (  title={Your Title},)

[//]: # (  author={Your team},)

[//]: # (  journal={Location},)

[//]: # (  year={Year})

[//]: # (})

[//]: # (```   )
