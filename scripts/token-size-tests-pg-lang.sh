for l in hu tr da it lv
do
for c in no tag m both
do
echo ${l}'-'${c};
parallel ./token-size-tests-pg-${c}char.sh ${l} $1 $2 :::: sh-params/sizes-${l} sh-params/samples; ./mine_results.sh . ${l}-pg-${c}char; ./remove_devout.sh; mv log-${l}-* logs_token_exp/;
done;
done;
