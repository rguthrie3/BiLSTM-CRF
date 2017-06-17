for l in hu tr he es ta
for c in m both
do
echo ${l}'-'${c};
parallel ./token-size-tests-pg-alllstm-${c}char.sh ${l} $1 $2 :::: sh-params/sizes-${l} sh-params/samples; ./mine_results.sh . ${l}-pg-${c}char; mv log-${l}-* logs_token_exp/;
done;
done;
