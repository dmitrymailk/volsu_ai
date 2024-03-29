## Ссылки на репозитории с хорошими курсами по ML

- [Машинное обучение в МФТИ](https://github.com/girafe-ai/ml-mipt)

## Lab 1

### Оригинал

Значитца так. Слушаем все внимательно. Я в служебной командировке, в связи с текущей ситуацией. Тут интернет не очень, да и глушилки могут работать. Поэтому онлайн-пары на субботу отменяются. У вас на эту субботу по плану должна быть практика, лабы. Пока будете разбираться в 0 лабе, я все равно сильно не нужен, там все по книжке Жерона (вверху). Значит задание по 0 лабе: рассчитать индекс удовлетворенности жизнью для соответствующего датасета из книжки (как его загружать, там указано), на основании известных данных ВВП на душу (оттуда же) для не менее 20 точек, не считая известных. Рассчитывать будете двумя способами - методом ближайших соседей и простой линейной регрессией (одномерной) с двумя коэффициентами. Сделать свои функции для ближайших соседей и регрессии (для регрессии обязательно провести минимизацию методом наименьших квадратов!) и использовать готовые функции scikit-learn. Результаты для ваших и библиотечных методов должны совпадать. В методе ближайших соседей посчитать для числа соседей 1 и 3. Сравнить и сделать выводы. Реализация на языке Python. Не забудьте установить библиотеку scikit-learn.

### Переделано

Пока будете разбираться в 0 лабе, я все равно сильно не нужен, там [все по книжке Жерона](./lab_task_materials/%D0%9F%D1%80%D0%B8%D0%BA%D0%BB%D0%B0%D0%B4%D0%BD%D0%BE%D0%B5%20%D0%BC%D0%B0%D1%88%D0%B8%D0%BD%D0%BD%D0%BE%D0%B5%20%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5%20%D1%81%20%D0%BF%D0%BE%D0%BC%D0%BE%D1%89%D1%8C%D1%8E%20Scikit-Learn%20%D0%B8%20TensorFlow%20%D0%9E%D1%80%D0%B5%D0%BB%D1%8C%D0%B5%D0%BD%20%D0%96%D0%B5%D1%80%D0%BE%D0%BD.pdf) - [github book](https://github.com/ageron/handson-ml)

- Значит задание по 0 лабе: рассчитать индекс удовлетворенности жизнью для [соответствующего датасета из книжки](https://github.com/ageron/handson-ml/tree/master/datasets/lifesat), на основании известных данных ВВП на душу (оттуда же) для не менее 20 точек, не считая известных.

- [ноутбук из книги где описывается как работать с датасетами](https://github.com/ageron/handson-ml/blob/master/01_the_machine_learning_landscape.ipynb)

- Рассчитывать будете двумя способами:

  - методом ближайших соседей
  - простой линейной регрессией (одномерной) с двумя коэффициентами

- Сделать свои функции для ближайших соседей и регрессии
  - для регрессии обязательно провести минимизацию методом наименьших квадратов! (и использовать готовые функции scikit-learn)
  - Результаты для ваших и библиотечных методов должны совпадать.
  - В методе ближайших соседей посчитать для числа соседей 1 и 3.
- Сравнить и сделать выводы. Реализация на языке Python. Не забудьте установить библиотеку scikit-learn.

### Теория

- [Linear Regression and Gradient Descent Using Only Numpy](https://towardsdatascience.com/linear-regression-and-gradient-descent-using-only-numpy-53104a834f75)

### Lab 2

### Оригинал

Добрый день. Следующая вторая лаба - Наивный Байесовский классификатор (на номер не смотрите, это порядковый номер, в котором я делал лабы сам). В качестве датасета используйте Fashion-MNIST (он есть в качестве встроенного в библиотеке scikit-learn, да и в других тоже имеется). Постарайтесь добиться точности хотя бы в 75%, хотя желательно все-таки свыше 80%. Данные крутите, как хотите, но максимальную точность классификации обеспечьте

- [доп материал с разьяснениями](./lab_task_materials/Lab_4_ML.docx)

### Переделано

- Наивный Байесовский классификатор
- В качестве датасета используйте Fashion-MNIST
- Точность 75%-80%

# Полезные ссылки

- [create environment python windows 10](https://docs.python.org/3/library/venv.html)

```
python -m venv volsu_ai
```

```
.\volsu_ai\Scripts\Activate.ps1
```

- [gitignore for python environments](https://github.com/github/gitignore/blob/main/Python.gitignore)
- [save packages to file](https://stackoverflow.com/questions/31684375/automatically-create-requirements-txt)

```
pip freeze > requirements.txt
```

- [vs code enable debugging in jupyter notebooks](https://medium.com/geekculture/debug-jupyter-notebooks-in-vscode-21b2be259f9d)

```
"jupyter.experimental.debugging": true,
```

## Sklearn

- [list of metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values)

# Pytorch

Install pytorch

```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio===0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
