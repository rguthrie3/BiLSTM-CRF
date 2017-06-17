for c in no tag
do
echo 'en-'${c};
parallel ./token-size-tests-ft-en-${c}char.sh $1 $2 :::: sh-params/sizes-en sh-params/samples; ./mine_results.sh . en-ft-${c}char; mv log-en-* logs_token_exp/;
done;