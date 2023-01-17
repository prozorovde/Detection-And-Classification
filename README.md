# Обнаружение и классификация

## Оглавление
- [Классические решения](#Классические-решения)
- [Нейронные сети](#Нейронные-сети)
    - [Универсальные сверточные](#Универсальные-сверточные)
    - [Yolo (You Only Look Once)](#Yolo)
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
| **RegNet** | *Модификация ResNet с добавлением механизма памяти* | [Pytorch code](https://pytorch.org/vision/main/models/regnet.html) <br/> [Paper (2021.01)](https://arxiv.org/abs/2101.00590) |
| **EfficientNet** | *Из описания: "тщательная балансировка глубины, ширины и разрешения сети может привести к повышению производительности"* | [Github (pytorch)](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/efficientnet) <br/> [Paper (2020.09)](https://arxiv.org/abs/1905.11946) |
| **SKResNeXt** | *CNN с адаптивной регулировкой размера рецептивного поля* | [Example (HuggingFace)](https://huggingface.co/docs/timm/models/skresnext) <br/> [Paper (2019.03)](https://arxiv.org/abs/1903.06586v2) |

### Yolo
| Нейросеть | Примечание |         Ссылки         |
|-----------|------------|--------|
| **Yolo v7** | *Обнаружение и классификации объектов в реальном масштабе времени.*          | [Github](https://github.com/wongkinyiu/yolov7) <br/> [Paper](https://arxiv.org/abs/2207.02696v1) <br/> [Dataset](https://paperswithcode.com/dataset/coco) |
| **Yolo v8** | *Обнаружение и классификации объектов в реальном масштабе времени.*          | [Github](https://github.com/ultralytics/ultralytics) <br/> [документация](https://docs.ultralytics.com/) |

### Трансформеры
| Нейросеть | Примечание |         Ссылки         |
|-----------|------------|--------|
| **GPViT: A High Resolution Non-Hierarchical Vision Transformer with Group Propagation** | *Высокопроизводительное обнаружение объектов с групповой моделью мобильности.* | [Github](https://github.com/chenhongyiyang/gpvit) <br/> [Paper (2022.12)](https://arxiv.org/pdf/2212.06795.pdf) |
| **Global Context Vision Transformer (GC ViT)** | *Обнаружение, классификация [объектов] и семантическая сегментация.* | [Github](https://github.com/NVlabs/GCViT) <br/> [Paper (2022.10)](https://arxiv.org/pdf/2206.09959.pdf) |
| **OWL-ViT: Open-World Object Detection with Vision Transformers** | *Нейросеть для обнаружения объектов с открытым словарем. Имея изображение и произвольный текстовый запрос, ИНС находит на изображении объекты, соответствующие этому запросу. Может обнаруживать объекты на основе одного примера изображения.* | [Github](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit) <br/> [Paper (2022.07)](https://arxiv.org/abs/2205.06230) <br/> [OWL-ViT minimal example (Colab)](https://colab.research.google.com/github/google-research/scenic/blob/main/scenic/projects/owl_vit/notebooks/OWL_ViT_minimal_example.ipynb) <br/> [OWL-ViT inference playground](https://colab.research.google.com/github/google-research/scenic/blob/main/scenic/projects/owl_vit/notebooks/OWL_ViT_inference_playground.ipynb) |

### Diffusion-модели
- **DiffusionDet: Diffusion Model for Object Detection**  
*DiffusionDet - одна из первых диффузионных моделей для обнаружения объектов.*  
    - [Github](https://github.com/shoufachen/diffusiondet)
    - [Paper (2022.11)](https://arxiv.org/abs/2211.09788)
### Оценка позы
- **AlphaPose**  
*Множественное оценивание поз людей на "сцене".*  
    - [Github](https://github.com/MVIG-SJTU/AlphaPose)
    - [Paper](https://arxiv.org/abs/2211.03375)
