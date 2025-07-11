# Разработка планирования дорожного трафика с помощью Reinforcement Learning

## Выполненные задачи

### Заменить алгоритм Флойда-Уоршелла на A*

В ходе выполнения данного задания было принято решение отказаться от A* в пользу Дейкстры, который, как и алгоритм Форда-Уоршелла, вызывался при инициализации всех классов до запуска симуляции. Идея была в том, чтобы просчитать все маршруты от стартовых точек (края карты для создания инициализационного трафика + генераторы Пуассоновских потоков). A* если и стоит использовать, потому что он считает расстояние от конкретной вершины до другой конкретной вершины. Следовательно, его удобнее использовать в runtime режиме, а не в режиме предпосчёта, так как в режиме предпосчёта он будет проигрывать как минимум Дейкстре (скорее всего, по факту не проверялось). В runtime режиме его использовать тоже не стоит, так как в программе есть узкое место, которое A* не сможет значительно улучшить (формирование пути из матрицы для восстановления пути). При большом количестве машин, отправляющихся в одно время, программа долго выполняет код для формирование пути из точки A в точку Б. 

### Внедрить Пуассоновские потоки генерации трафика

На данный момент время отправки между двумя автомобилями не подчиняется экспоненциальному распределению, но методы для генерации и маршрутизации трафика из заданных точек реализованы.

### Реализовать возможность передавать через конфигурационный файл данные об интенсивностях трафика и количестве генераторов потоков

Ввиду того, что предыдущий пункт не доделан, данное задание не выполнено.


## Задачи на следующую итерацию
1. Внедрить Пуассоновские потоки генерации трафика.
2. Реализовать возможность передавать через конфигурационный файл данные об интенсивностях трафика и количестве генераторов потоков.
3. Внедрить критерий конца генерации инициализирующего трафика с последующим переходом к генерации с помощью Пуассоновских потоков.
 

