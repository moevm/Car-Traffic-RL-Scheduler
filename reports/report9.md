# Разработка планирования дорожного трафика с помощью Reinforcement Learning

## Выполненные задачи

### Исправлены баги

Баги исправлены, теперь симуляция не падает.

### Внедрить светофоры: установить светофоры (простейшие, без поворотов) на каждом перекрёстке и по середине каждого ребра, длина которого больше некоторой константы

1. Светофоры установлены на каждый перекрёсток.
2. На середине ребра установить светофор не удалось ввиду того, что sumo позволяет устанавливать светофоры только на узлах графа. В городской среде светофоры устаналиваются каждые 200-300 метров, поэтому можно генерировать карты, у которых длина каждого ребра не больше 200-300 метров и в конфигурационный файл карты в формате xml вставлять светофоры в узлы, которые имеют только два смежных узла.
3. Добавлена возможность регулировать количество активных светофоров на карте.

### Попробовать внедрить простейший RL-алгоритм.

С помощью TraciAPI были исследованы возможности работы со светофорными фазами в sumo. Базовые настройки светофоров визуально показали, что в городе стало меньше пробок пробок по сравнению
с предыдущей версией проекта, когда светофоры отсутствовали.


 ## Задачи на следующую итерацию

 1. Обновить Readme.
 2. Продумать какие будут Actions.
 3. Ознакомиться с gymnasium и создать с его помощью окружение для обучения.
 4. Ознакомиться с sumo-rl.

 