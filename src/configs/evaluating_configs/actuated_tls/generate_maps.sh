#!/bin/sh
netgenerate --rand --rand.iterations 270 --rand.max-distance 300 --rand.min-distance 200 --rand.min-angle 90 --rand.neighbor-dist1 10 --rand.neighbor-dist2 40 --rand.neighbor-dist3 25 --rand.neighbor-dist4 25 --rand.neighbor-dist5 0 --rand.neighbor-dist6 0 -L 2 --rand.grid --tls.guess --tls.guess.threshold 0 --tls.default-type actuated --tls.min-dur 15 --tls.max-dur 84 -o rand_80.net.xml

netgenerate --rand --rand.iterations 62 --rand.max-distance 300 --rand.min-distance 200 --rand.min-angle 90 --rand.neighbor-dist1 10 --rand.neighbor-dist2 40 --rand.neighbor-dist3 25 --rand.neighbor-dist4 25 --rand.neighbor-dist5 0 --rand.neighbor-dist6 0 -L 2 --rand.grid --tls.guess --tls.guess.threshold 0 --tls.default-type actuated --tls.min-dur 15 --tls.max-dur 84 -o rand_20.net.xml

netgenerate --rand --rand.iterations 20 --rand.max-distance 300 --rand.min-distance 200 --rand.min-angle 90 --rand.neighbor-dist1 10 --rand.neighbor-dist2 40 --rand.neighbor-dist3 25 --rand.neighbor-dist4 25 --rand.neighbor-dist5 0 --rand.neighbor-dist6 0 -L 2 --rand.grid --tls.guess --tls.guess.threshold 0 --tls.default-type actuated --tls.min-dur 15 --tls.max-dur 84 -o rand_4.net.xml

netgenerate --rand --rand.iterations 140 --rand.max-distance 300 --rand.min-distance 200 --rand.min-angle 90 --rand.neighbor-dist1 10 --rand.neighbor-dist2 40 --rand.neighbor-dist3 25 --rand.neighbor-dist4 25 --rand.neighbor-dist5 0 --rand.neighbor-dist6 0 -L 2 --rand.grid --tls.guess --tls.guess.threshold 0 --tls.default-type actuated --tls.min-dur 15 --tls.max-dur 84 -o rand_40.net.xml
