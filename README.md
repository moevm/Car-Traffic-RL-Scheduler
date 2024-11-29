# Car-Traffic-RL-Scheduler

## Описание

Данное приложение представляет собой систему планирования дорожного трафика, использующую методы обучения с подкреплением для решения проблемы максимизации пропускной способности сети. Система предназначена для анализа и оптимизации потоков транспортных средств в сложных городских условиях, что позволяет улучшить эффективность дорожной инфраструктуры и снизить время в пути для пользователей.

Для работы с дорожным трафиком использовался симулятор SUMO (https://github.com/eclipse-sumo/sumo) версии 1.23.0.

## Запуск через Docker

1. Склонируйте репозиторий с помощью следующей команды:
```bash
https://github.com/moevm/Car-Traffic-RL-Scheduler.git
```

2. Перейдите в директорию с приложением:
```bash
cd Car-Traffic-RL-Scheduler
```

3. Запустите скрипт run-docker.sh, который сделает образ приложения, включит на локальной машине X-сервер, а затем запустит контейнер на основе данного образа:
```bash
sh run-docker.sh
```

4. Оказавшись внутри контейнера, включите виртуальную переменную окружения (она уже создана):
```bash
source sumovenv/bin/activate
```

5. Контейнер готов к работе.

## Интерфейс работы с приложением

Для запуска симуляции дорожной сети необходимо предварительно сгенерировать карту. 

### Генерация карт

#### Описание

Для генерации дорожных карт в директории src данного приложения существует скрипт mapGenerator.py. С его помощью можно сгенерировать несколько карт, которые в той или иной степени буду похожи. Степень схожести карт между собой Вы можете регулировать с помощью параметров генерации. Данный скрипт является надстройкой над утилитой netgenerate из SUMO (https://sumo.dlr.de/docs/netgenerate.html). Команды генерации для карт указаны в [src/grid-config.json](https://github.com/moevm/Car-Traffic-RL-Scheduler/blob/main/src/base-cli-params/grid-config.json), [src/rand-config.json](https://github.com/moevm/Car-Traffic-RL-Scheduler/blob/main/src/base-cli-params/rand-config.json), [src/spider-config.json](https://github.com/moevm/Car-Traffic-RL-Scheduler/blob/main/src/base-cli-params/spider-config.json). Числа в данных json-файлах нужны лишь для сигнализации типов параметров и не более. Никакого реального смысла они не несут. 

Команды наследуют тот же смысл, что и в утилите netgenerate, а команды с окончаниями -low и -high и значения после них задают границы генерации соответствующего параметра для каждой карты по равномерному распределению.

#### Примеры генерации карт

```bash
python3 mapGenerator.py -g --number-networks 5 --grid.x-length-low 10 --grid.x-length-high 100 --grid.y-length-low 50 --grid.y-length-high 500 --grid.x-number-low 5 --grid.x-number-high 50 --grid.y-number-low 10 --grid.y-number-high 100
```

Данная команда генерирует 5 карт типа сетка, у каждой из которых:
1. длина горизонтальных рёбер находится в пределе от 10 до 100 метров;
2. длина вертикальных рёбер находится в пределе от 50 до 500 метров;
3. количество узлов по горизонтали находится в пределе от 5 до 50;
4. количество узлов по вертикали находится в пределе от 10 до 100;

```bash
python3 mapGenerator.py -r --number-networks 5 --rand.iterations-low 500 --rand.iterations-high 1000 --rand.max-distance-low 200 --rand.max-distance-high 400 --rand.min-distance-low 50 --rand.min-distance-high 150 --rand.min-angle-low 30 --rand.min-angle-high 60 --rand.neighbor-dist5-low 5 --rand.neighbor-dist5-high 10
```

Данная команда генерирует 5 случайных карт, у каждой из которых:
1. количество узлов от 500 до 1000;
2. максимальная длина ребра находится в пределе от 200 до 400 метров;
3. минимальная длина ребра находится в пределе от 50 до 150 метров;
4. минимальный угол между рёбрами находится в пределе от 30 до 60 градусов;
5. вероятность того, что у узла пять соседей, находится в пределе от 5 до 10. Стоит отметить, что на самом как таковая вероятность не используется, поэтому такие странные значения (>1). Подробнее: https://sumo.dlr.de/docs/netgenerate.html

```bash
python3 mapGenerator.py -s --number-networks 5 --spider.arm-number-low 10 --spider.arm-number-high 50 --spider.circle-number-low 5 --spider.circle-number-high 50 --spider.space-radius-low 50 --spider.space-radius-high 150 --spider.omit-center-low false --spider.omit-center-high true
```
Данная команда генерирует 5 карт в форме паутины, у каждой из которых:
1. количество граней каждого круга паутины находится в промежутку от 10 до 50;
2. количество кругов паутины находится в промежутке от 5 до 50 метров;
3. радиус паутины находится в промежутке от 50 до 150 метров;
4. центр паутины либо есть, либо нет.

В зависимости от типа карты она сохраняется либо в /src/configs/grid-configs, либо в /src/configs/rand-configs, /src/spider-configs.

### Запуск приложения

#### Описание

После генерации карт приложение готово к запуску.

#### Инструкции по запуску

**В случае запуска не в Docker:**
1. Установите SUMO: https://sumo.dlr.de/docs/Installing/index.html
2. Задайте локальную переменную SUMO_HOME (https://sumo.dlr.de/docs/Installing/index.html):
```bash
export SUMO="your/path/to/sumo"
```
3. Склонируйте репозиторий:
```bash
https://github.com/moevm/Car-Traffic-RL-Scheduler.git
```
4. Перейдите в папку с исходным кодом:
```bash
cd Car-Traffic-RL-Scheduler/src
```
5. Создайте виртуальную переменную окружения python (в случае запуска приложения через Docker пропустите данный шаг):
```bash
python3 -m venv <your_venv_name>
```
6. Активируйте переменную окружения:
```bash
source <your_venv_name>/bin/activate
```
7. Установите зависимости:
```bash
pip install -r ../requirements.txt
```
**Команды далее применимы и для Docker после того, как Вы вошли в контейнер:**

8. Если карты уже сгенерированы, то данный шаг следует опустить. В ином случае можно, например сгенерировать 5 случайных карт, параметры который были описаны ранее:
```bash
python3 mapGenerator.py -r --number-networks 5 --rand.iterations-low 500 --rand.iterations-high 1000 --rand.max-distance-low 200 --rand.max-distance-high 400 --rand.min-distance-low 50 --rand.min-distance-high 150 --rand.min-angle-low 30 --rand.min-angle-high 60 --rand.neighbor-dist5-low 5 --rand.neighbor-dist5-high 10
```
9. Запуск генерации приложения:
```bash
python3 main.py configs/rand-configs/rand3.sumocfg
```
10. Перед Вами откроется sumo-gui. Загрузка карты может занять время. После загрузки карты установите Delay в значение 100 ms и нажмите на зелёный треугольник. В терминале можно наблюдать за логами.

[gif]