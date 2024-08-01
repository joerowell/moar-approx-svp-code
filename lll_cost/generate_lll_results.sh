two=2

for i in {50..500}
do
    latticegen -randseed 0 q $i $((i / two)) 30 b | ./lll_cost &>> lll_cost0 &
    latticegen -randseed 1 q $i $((i / two)) 30 b | ./lll_cost &>> lll_cost1 &
    latticegen -randseed 2 q $i $((i / two)) 30 b | ./lll_cost &>> lll_cost2 &
    latticegen -randseed 3 q $i $((i / two)) 30 b | ./lll_cost &>> lll_cost3 &
    latticegen -randseed 4 q $i $((i / two)) 30 b | ./lll_cost &>> lll_cost4 &
    latticegen -randseed 5 q $i $((i / two)) 30 b | ./lll_cost &>> lll_cost5 &
    latticegen -randseed 6 q $i $((i / two)) 30 b | ./lll_cost &>> lll_cost6 &
    latticegen -randseed 7 q $i $((i / two)) 30 b | ./lll_cost &>> lll_cost7 &
    latticegen -randseed 8 q $i $((i / two)) 30 b | ./lll_cost &>> lll_cost8 &
    latticegen -randseed 9 q $i $((i / two)) 30 b | ./lll_cost &>> lll_cost9
done
