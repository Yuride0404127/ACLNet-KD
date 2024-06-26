# ACLNet-KD
Asymmetrical Contrastive Learning Network via Knowledge Distillation for No-Service Rail Surface Defect Detection

## The model structure is as follows：

![KD_2](https://github.com/Yuride0404127/ACLNet-KD/blob/main/Picture/KD_2_v3.png)



## Requirements

Python >=3.6

Pytorch >= 1.7.1

Cuda = 10.2



## Feature Maps

The results graphs of our tests are saved in the：

[Baidu Clould]<https://pan.baidu.com/s/1d0poeuS5JQt4dDQ9FhibJg?pwd=cim9>   提取码：cim9 


## Training Weights

All the training weights of our model will be saved in the: 

[Baidu Clould] <https://pan.baidu.com/s/13FeF2hgFtm6SrsEkhnT-lQ?pwd=dsy6>  提取码：dsy6 



## Comparison of results table

Table I Evaluation metrics obtained from compared methods.

| Model                  | *S<sub>m</sub>* ↑ | *MAE*  ↓  | *maxF<sub>m</sub>* ↑ | *maxE<sub>m</sub>* ↑ | *FLOPs/G* ↓ | *Params/M*  ↓ |
| ---------------------- | ----------------- | --------- | -------------------- | -------------------- | ----------- | ------------- |
| S2MA<sub>2020</sub>    | 0.775             | 0.141     | 0.817                | 0.864                | 108.1       | 86.7          |
| EDR<sub>2020</sub>     | 0.811             | 0.082     | 0.850                | 0.893                | 32.2        | 39.3          |
| BBS<sub>2020  </sub>   | 0.828             | 0.074     | 0.867                | 0.909                | 12.7        | 49.8          |
| HAI<sub>2021 </sub>    | 0.718             | 0.171     | 0.803                | 0.829                | 73.5        | 59.8          |
| EMI<sub>2021  </sub>   | 0.800             | 0.104     | 0.850                | 0.876                | 106.9       | 99.1          |
| SPNet<sub>2021  </sub> | 0.830             | 0.072     | 0.877                | 0.915                | 55.0        | 175.3         |
| DAC<sub>2022  </sub>   | 0.824             | 0.071     | 0.875                | 0.911                | 109.3       | 98.4          |
| CSEP<sub>2022  </sub>  | 0.814             | 0.085     | 0.866                | 0.899                | 45.4        | 18.8          |
| CIR<sub>2022  </sub>   | 0.809             | 0.086     | 0.856                | 0.894                | 17.3        | 103.2         |
| RD3D<sub>2022  </sub>  | 0.797             | 0.093     | 0.839                | 0.883                | 17.6        | 28.9          |
| DRER<sub>2022  </sub>  | 0.844             | 0.059     | 0.891                | 0.933                | 17.3        | 69.8          |
| CLA<sub>2022  </sub>   | 0.835             | 0.069     | 0.878                | 0.920                | 82.6        | 184.2         |
| LENO<sub>2022  </sub>  | 0.817             | 0.083     | 0.857                | 0.900                | 162.1       | 131.0         |
| ICON<sub>2023  </sub>  | 0.843             | 0.066     | 0.884                | 0.925                | 34.8        | 65.7          |
| CAVER<sub>2023  </sub> | 0.809             | 0.090     | 0.848                | 0.891                | 63.0        | 93.8          |
| ACLNet-T               | **0.860**         | **0.055** | **0.897**            | **0.938**            | 20.7        | 79.1          |
| ACLNet-S               | 0.842             | 0.063     | 0.883                | 0.926                | **5.6**     | **11.9**      |
| ACLNet-S*              | **0.856**         | **0.056** | **0.896**            | **0.936**            | **5.6**     | **11.9**      |



##  Table II. Test and distillation results on RGBD-SOD public datasets.



| Model                  |                   | NJU2K     |                      |                      |                   | NLPR      |                      |                      |                   | STERE     |                      |                      |
| ---------------------- | ----------------- | --------- | -------------------- | -------------------- | ----------------- | --------- | -------------------- | -------------------- | ----------------- | --------- | -------------------- | -------------------- |
|                        | *S<sub>m</sub>* ↑ | *MAE*  ↓  | *maxF<sub>m</sub>* ↑ | *maxE<sub>m</sub>* ↑ | *S<sub>m</sub>* ↑ | *MAE*  ↓  | *maxF<sub>m</sub>* ↑ | *maxE<sub>m</sub>* ↑ | *S<sub>m</sub>* ↑ | *MAE*  ↓  | *maxF<sub>m</sub>* ↑ | *maxE<sub>m</sub>* ↑ |
| S2MA<sub>2020   </sub> | 0.894             | 0.053     | 0.930                | 0.889                | 0.915             | 0.030     | 0.953                | 0.902                | 0.890             | 0.051     | 0.932                | 0.882                |
| EDR<sub>2020  </sub>   | 0.839             | 0.076     | 0.883                | 0.823                | 0.886             | 0.038     | 0.928                | 0.860                | 0.851             | 0.065     | 0.899                | 0.831                |
| BBS<sub>2020  </sub>   | 0.921             | 0.035     | 0.949                | 0.920                | 0.930             | 0.023     | 0.961                | 0.918                | 0.908             | 0.041     | 0.942                | 0.903                |
| HAI<sub>2021  </sub>   | 0.912             | 0.038     | 0.944                | 0.915                | 0.921             | 0.024     | 0.960                | 0.915                | 0.907             | 0.040     | 0.944                | 0.906                |
| EMI<sub>2021  </sub>   | 0.881             | 0.050     | 0.924                | 0.876                | 0.916             | 0.026     | 0.953                | 0.902                | 0.897             | 0.042     | 0.938                | 0.894                |
| DAC<sub>2022  </sub>   | 0.890             | 0.046     | 0.929                | 0.887                | 0.913             | 0.025     | 0.949                | 0.897                | 0.899             | 0.043     | 0.936                | 0.892                |
| MC<sub>2022  </sub>    | 0.902             | 0.042     | 0.939                | 0.900                | 0.918             | 0.025     | 0.956                | 0.907                | 0.903             | 0.042     | 0.945                | 0.898                |
| CIR<sub>2022  </sub>   | 0.925             | 0.035     | 0.955                | 0.927                | 0.933             | 0.023     | 0.966                | 0.924                | 0.917             | 0.039     | 0.950                | 0.916                |
| DRER<sub>2022  </sub>  | 0.906             | 0.038     | 0.943                | 0.907                | 0.915             | 0.024     | 0.953                | 0.901                | 0.895             | 0.042     | 0.943                | 0.891                |
| LENO<sub>2022  </sub>  | 0.838             | 0.073     | 0.888                | 0.824                | 0.878             | 0.040     | 0.921                | 0.845                | 0.856             | 0.062     | 0.906                | 0.840                |
| ICON<sub>2023  </sub>  | 0.893             | 0.051     | 0.937                | 0.891                | 0.904             | 0.032     | 0.951                | 0.885                | 0.899             | 0.047     | 0.945                | 0.890                |
| CAVER<sub>2023</sub>   | 0.926             | 0.030     | 0.958                | 0.928                | 0.934             | 0.021     | 0.970                | 0.928                | 0.918             | 0.033     | 0.955                | 0.916                |
| ACLNet-T               | **0.929**         | **0.028** | **0.960**            | **0.932**            | **0.937**         | **0.018** | **0.971**            | **0.929**            | **0.918**         | **0.032** | **0.955**            | **0.915**            |
| ACLNet-S               | 0.911             | 0.038     | 0.949                | 0.910                | 0.927             | 0.023     | 0.965                | 0.918                | 0.916             | 0.036     | 0.952                | 0.909                |
| ACLNet-S*              | 0.913             | 0.037     | 0.951                | 0.914                | 0.930             | 0.021     | 0.966                | 0.919                | 0.917             | 0.035     | 0.953                | 0.911                |



