**Приклад елементарної одношарової штучної нейронної мережі**


---

## Запуск

В корні проекту відкрийте термінал та запустіть інтерпретатор Python командою `python3`:
```
$ python3
Python 3.6.4 |Anaconda, Inc.| (default, Jan 16 2018, 18:10:19) 
[GCC 7.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> 
```

Імпортуйте клас NeuralNetwork із модуля neural_network
```
>>> from neural_network import NeuralNetwork
```


Ініціалізуйте екземпляр нейронної мережі на базі класу NeuralNetwork
```
>>> my_network = NeuralNetwork("arg")
```


Натренуйте вашу нейронну мережу
```
>>> my_networ.train(input_data)
```


Запитайте вашу нейронну мережу та отримайте результат її розрахунків
```
>>> result = my_networ.query(input_data)
```

Для виходу з інтерпретатора Python натисніть `Ctrl+D`

---

## Набір даних

Повний набір даних MNIST завеликий для зберігання на GitHub тому він відсутній в репозиторії. 
Повний MNIST dataset у форматі CSV доступний за посиланнями 
[train set](https://pjreddie.com/media/files/mnist_train.csv) тa 
[test set](https://pjreddie.com/media/files/mnist_test.csv) 
на сайті [https://pjreddie.com/](http://pjreddie.com/projects/mnist-in-csv/)
Завантажте та покладіть тренувальний (з назвою `mnist_train_60000_full.csv`) 
та тестувальний (з назвою `mnist_test_10000_full.csv`) набори 
в директорію `/classify_MNIST_handwritten_digits/dataset/MNIST`

---

## Джерело

Проект створено на базі [прикладів](https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork) з книги "Створюємо нейронну мережу" авт. Тарік Рашид  ("Make your own neural network" by Tariq Rashid).
