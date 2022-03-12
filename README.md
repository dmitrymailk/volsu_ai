## Lab 1

### Оригинал

Значитца так. Слушаем все внимательно. Я в служебной командировке, в связи с текущей ситуацией. Тут интернет не очень, да и глушилки могут работать. Поэтому онлайн-пары на субботу отменяются. У вас на эту субботу по плану должна быть практика, лабы. Пока будете разбираться в 0 лабе, я все равно сильно не нужен, там все по книжке Жерона (вверху). Значит задание по 0 лабе: рассчитать индекс удовлетворенности жизнью для соответствующего датасета из книжки (как его загружать, там указано), на основании известных данных ВВП на душу (оттуда же) для не менее 20 точек, не считая известных. Рассчитывать будете двумя способами - методом ближайших соседей и простой линейной регрессией (одномерной) с двумя коэффициентами. Сделать свои функции для ближайших соседей и регрессии (для регрессии обязательно провести минимизацию методом наименьших квадратов!) и использовать готовые функции scikit-learn. Результаты для ваших и библиотечных методов должны совпадать. В методе ближайших соседей посчитать для числа соседей 1 и 3. Сравнить и сделать выводы. Реализация на языке Python. Не забудьте установить библиотеку scikit-learn.

### Переделано

Пока будете разбираться в 0 лабе, я все равно сильно не нужен, там все по книжке Жерона (вверху).

- Значит задание по 0 лабе: рассчитать индекс удовлетворенности жизнью для соответствующего датасета из книжки (как его загружать, там указано), на основании известных данных ВВП на душу (оттуда же) для не менее 20 точек, не считая известных.
- Рассчитывать будете двумя способами -

  - методом ближайших соседей
  - простой линейной регрессией (одномерной) с двумя коэффициентами

- Сделать свои функции для ближайших соседей и регрессии
  - для регрессии обязательно провести минимизацию методом наименьших квадратов! (и использовать готовые функции scikit-learn)
  - Результаты для ваших и библиотечных методов должны совпадать.
  - В методе ближайших соседей посчитать для числа соседей 1 и 3.
- Сравнить и сделать выводы. Реализация на языке Python. Не забудьте установить библиотеку scikit-learn.

### Lab 2

### Оригинал

Добрый день. Следующая вторая лаба - Наивный Байесовский классификатор (на номер не смотрите, это порядковый номер, в котором я делал лабы сам). В качестве датасета используйте Fashion-MNIST (он есть в качестве встроенного в библиотеке scikit-learn, да и в других тоже имеется). Постарайтесь добиться точности хотя бы в 75%, хотя желательно все-таки свыше 80%. Данные крутите, как хотите, но максимальную точность классификации обеспечьте

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
