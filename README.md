# Обнаружение и классификация

## Оглавление
- [Классические решения](#Классические-решения)
- [Нейронные сети](#Нейронные-сети)
    - [Универсальные сверточные](#Универсальные-сверточные)
    - [Yolo (You Only Look Once)](#Yolo-based)
    - [Трансформеры](#Трансформеры)
    - [Diffusion-модели](#Diffusion-модели)
    - [Оценка позы](#Оценка-позы)
## Классические решения
- **k-ближайших соседей (K-Nearest Neighbors)**  
[документация scikit-learn](https://scikit-learn.org/stable/modules/neighbors.html)
- **Метод опорных векторов (Support Vector Machines - SVM)**  
[документация sciki-learn](https://scikit-learn.org/stable/modules/svm.html)
- **Решающие деревья (Decision Trees)**  
[документация sciki-learn](https://scikit-learn.org/stable/modules/tree.html)
- **Случайный лес (Random Forrest)**  
[документация sciki-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- **Наивный байесовский метод (Naive Bayes)**  
[документация sciki-learn](https://scikit-learn.org/stable/modules/naive_bayes.html)
- **Линейный дискриминантный анализ (Linear Discriminant Analysis)**  
[документация sciki-learn](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
- **Логистическая регрессия (Logistic Regression)**  
[документация sciki-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- **Градиентный бустинг XGBoost**  
[документация](https://xgboost.readthedocs.io/en/stable/)
---
## Нейронные сети
### Универсальные сверточные
| Нейросеть | Примечание |         Ссылки         |
|-----------|------------|--------|
| **RegNet** | *Модификация ResNet с добавлением механизма памяти* | [Paper (2021.01)](https://arxiv.org/abs/2101.00590) <br/> [Pytorch code](https://pytorch.org/vision/main/models/regnet.html) |
| **EfficientNet** | *Из описания: "тщательная балансировка глубины, ширины и разрешения сети может привести к повышению производительности"* | [Paper (2020.09)](https://arxiv.org/abs/1905.11946) <br/> [Github (pytorch)](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/efficientnet) |
| **ESE VovNet** | *Модификация ResNet. Использует "пространственно-управляемую маску"* | [Paper (2020.04)](https://arxiv.org/abs/1911.06667v6) <br/> [Example (HuggingFace)](https://huggingface.co/docs/timm/models/ese-vovnet) |
| **MixNet** | *CNN, использующая ядра разных размеров в одном и том же сверточном слое.* | [Paper (2019.12)](https://arxiv.org/pdf/1907.09595.pdf) <br/> [Github 1 (pytorch)](https://github.com/romulus0914/MixNet-PyTorch) <br/> [Github 2 (pytorch)](https://github.com/linksense/MixNet-PyTorch) |
| **CSPResNeXt** | *CNN со стратегией CSPNet, примененной к ResNeXt* | [Paper (2019.11)](https://arxiv.org/abs/1911.11929v1) <br/> [Example (pytorch)](https://rwightman.github.io/pytorch-image-models/models/csp-resnext/) |
| **CSPNet** | *CNN со стратегией CSPNet, примененной к ResNet* | [Paper (2019.11)](https://arxiv.org/abs/1911.11929v1) <br/> [Example (HuggingFace)](https://huggingface.co/docs/timm/models/csp-resnet) |
| **SEResNeXt** | *ResNeXt - модифицированный вариант ResNet. Улучшение достигается за счет введения блоков адаптивной калибровки каналов сверточных фильтров.* | [Paper (2019.05)](https://arxiv.org/abs/1709.01507v4) <br/> [Example (pytorch)](https://rwightman.github.io/pytorch-image-models/models/seresnext/) |
| **FBNet** | *FBNet — оптимизированная MobileNetv2 с помощью метода поиска нейросетевой архитектуры DNAS.* | [Paper (2019.05)](https://arxiv.org/pdf/1812.03443v3.pdf) <br/> [Example (HuggingFace)](https://huggingface.co/docs/timm/models/fbnet) |
| **SKResNeXt** | *CNN с адаптивной регулировкой размера рецептивного поля* | [Paper (2019.03)](https://arxiv.org/abs/1903.06586v2) <br/> [Example (HuggingFace)](https://huggingface.co/docs/timm/models/skresnext) |
| **DPN** | *Dual Path Networks - сети, использующие идеи ResNet и DenseNet одновременно.* | [Paper (2017.08)](https://arxiv.org/abs/1707.01629) | [Example (HuggingFace)](https://huggingface.co/docs/timm/models/dpn) |
| **MobileNet** | *Семейство моделей с оптимизированной архитектурой для мобильных приложений* | [Paper (2017.04)](https://arxiv.org/abs/1704.04861) <br/> [Example (pytorch)](https://rwightman.github.io/pytorch-image-models/models/mobilenet-v3/) |
| **ResNeXt** | *Модифицированный вариант ResNet* | [Paper (2017.04)](https://arxiv.org/abs/1611.05431) <br/> [Example (pytorch)](https://pytorch.org/hub/pytorch_vision_resnext/) |
| **ResNet** | *Т.н. "остаточная" сверточная нейросеть. Базовый вариант для начальных экспериментов.* | [Paper (2015.12)](https://arxiv.org/abs/1512.03385) <br/> [Example (pytorch)](https://pytorch.org/hub/pytorch_vision_resnet/) |

### Yolo-based
| Нейросеть | Примечание |         Ссылки         |
|-----------|------------|--------|
| **Yolo v8** | *Обнаружение и классификации объектов в реальном масштабе времени.*          | [документация](https://docs.ultralytics.com/) <br/> [Github](https://github.com/ultralytics/ultralytics) |
| **Yolo v7** | *Обнаружение и классификации объектов в реальном масштабе времени.*          | [Paper](https://arxiv.org/abs/2207.02696v1) <br/> [Github](https://github.com/wongkinyiu/yolov7) <br/> [Dataset](https://paperswithcode.com/dataset/coco) |
| **CSPDarknet** | *CNN со стратегией CSPNet, использующая в качестве основы Yolo v4* | [Paper (2020.04)](https://arxiv.org/abs/2004.10934v1) <br/> [Example (pytorch)](https://rwightman.github.io/pytorch-image-models/models/csp-darknet/) |

### Трансформеры
| Нейросеть | Примечание |         Ссылки         |
|-----------|------------|--------|
| **GPViT: A High Resolution Non-Hierarchical Vision Transformer with Group Propagation** | *Высокопроизводительное обнаружение объектов с групповой моделью мобильности.* | [Paper (2022.12)](https://arxiv.org/pdf/2212.06795.pdf) <br/> [Github](https://github.com/chenhongyiyang/gpvit) |
| **Global Context Vision Transformer (GC ViT)** | *Обнаружение, классификация [объектов] и семантическая сегментация.* | [Paper (2022.10)](https://arxiv.org/pdf/2206.09959.pdf) <br/> [Github](https://github.com/NVlabs/GCViT) |
| **OWL-ViT: Open-World Object Detection with Vision Transformers** | *Нейросеть для обнаружения объектов с открытым словарем. Имея изображение и произвольный текстовый запрос, ИНС находит на изображении объекты, соответствующие этому запросу. Может обнаруживать объекты на основе одного примера изображения.* | [Paper (2022.07)](https://arxiv.org/abs/2205.06230) <br/> [Github](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit) <br/> [OWL-ViT minimal example (Colab)](https://colab.research.google.com/github/google-research/scenic/blob/main/scenic/projects/owl_vit/notebooks/OWL_ViT_minimal_example.ipynb) <br/> [OWL-ViT inference playground](https://colab.research.google.com/github/google-research/scenic/blob/main/scenic/projects/owl_vit/notebooks/OWL_ViT_inference_playground.ipynb) |

### Diffusion-модели
- **DiffusionDet: Diffusion Model for Object Detection**  
*DiffusionDet - одна из первых диффузионных моделей для обнаружения объектов.*  
    - [Paper (2022.11)](https://arxiv.org/abs/2211.09788)
    - [Github](https://github.com/shoufachen/diffusiondet)
### Оценка позы
- **AlphaPose**  
*Множественное оценивание поз людей на "сцене".*  
    - [Paper (2022.11)](https://arxiv.org/abs/2211.03375)
    - [Github](https://github.com/MVIG-SJTU/AlphaPose)
