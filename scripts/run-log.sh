# 12/20
python model.py --dataset results/en_mtags-dd.pkl --word-embeddings data/embs/en_wiki_vectors_varembed_ud.txt --log-dir log-noseq-varinit-char-05dr-avgtags --dev-output log-noseq-varinit-char-05dr-avgtags/devout.txt --dropout 0.5 --use-char-rnn --no-sequence-model --num-epochs 40;
python model.py --dataset results/en_mtags-dd.pkl --log-dir log-noseq-randinit-char-05dr-avgtags --dev-output log-noseq-randinit-char-05dr-avgtags/devout.txt --dropout 0.5 --use-char-rnn --no-sequence-model --num-epochs 40;
python model.py --dataset results/en_mtags-dd.pkl --word-embeddings data/embs/en_wiki_vectors_varembed_ud.txt --log-dir log-noseq-varinit-nochar-05dr-avgtags --dev-output log-noseq-varinit-nochar-05dr-avgtags/devout.txt --dropout 0.5 --no-sequence-model --num-epochs 40;
python model.py --dataset results/en_mtags-pos.pkl --word-embeddings data/embs/en_wiki_vectors_varembed_ud.txt --log-dir log-onlypos-noseq-varinit-char-05dr --dev-output log-onlypos-noseq-varinit-char-05dr/devout.txt --dropout 0.5 --use-char-rnn --no-sequence-model --num-epochs 40
# After some commits (12/21)
python model.py --dataset results/en_mtags-dd.pkl --log-dir log-noseq-randinit-nochar-05dr-avgtags --dropout 0.5 --no-sequence-model --num-epochs 40
# 12/21
python model.py --dataset results/en_mtags-pos.pkl --word-embeddings data/embs/en_wiki_vectors_varembed_ud.txt --log-dir log-vit-zero-onlypos-varinit-nochar-avgloss --viterbi --loss-margin zero --num-epochs 20
python model.py --dataset results/en_mtags-dd.pkl --word-embeddings data/embs/en_wiki_vectors_varembed_ud.txt --log-dir log-vit-zero-varinit-nochar-avgloss --viterbi --loss-margin zero --num-epochs 20
python model.py --dataset results/en_mtags-pos.pkl --word-embeddings data/embs/en_wiki_vectors_varembed_ud.txt --log-dir log-vit-one-onlypos-varinit-nochar-avgloss --viterbi --loss-margin one --num-epochs 20
python model.py --dataset results/en_mtags-dd.pkl --word-embeddings data/embs/en_wiki_vectors_varembed_ud.txt --log-dir log-vit-one-varinit-nochar-avgloss --viterbi --loss-margin one --num-epochs 20
python model.py --dataset results/en_mtags-dd.pkl --word-embeddings data/embs/en_wiki_vectors_varembed_ud.txt --log-dir log-vit-attprop-varinit-nochar-avgloss --viterbi --loss-margin att-prop --num-epochs 20
# After changing to sum, and enabling more configs (12/22)
python model.py --dataset results/en_mtags-dd.pkl --word-embeddings data/embs/en_wiki_vectors_varembed_ud.txt --log-dir log-vit-zero-varinit-nochar-sumloss --viterbi --loss-margin zero --num-epochs 20
python model.py --dataset results/en_mtags-dd.pkl --word-embeddings data/embs/en_wiki_vectors_varembed_ud.txt --log-dir log-vit-one-varinit-nochar-sumloss --viterbi --loss-margin one --num-epochs 20
python model.py --dataset results/en_mtags-dd.pkl --word-embeddings data/embs/en_wiki_vectors_varembed_ud.txt --log-dir log-vit-attprop-varinit-nochar-sumloss --viterbi --loss-margin att-prop --num-epochs 20
python model.py --dataset results/en_mtags-pos.pkl --word-embeddings data/embs/en_wiki_vectors_varembed_ud.txt --log-dir log-vit-zero-onlypos-varinit-char-sumloss --viterbi --use-char-rnn --loss-margin zero --num-epochs 20
python model.py --dataset results/en_mtags-pos.pkl --word-embeddings data/embs/en_wiki_vectors_varembed_ud.txt --log-dir log-vit-one-onlypos-varinit-char-sumloss --viterbi --use-char-rnn --loss-margin one --num-epochs 20
python model.py --dataset results/en_mtags-dd.pkl --word-embeddings data/embs/en_wiki_vectors_varembed_ud.txt --log-dir log-vit-zero-varinit-char-sumloss --use-char-rnn --viterbi --loss-margin zero --num-epochs 20
python model.py --dataset results/en_mtags-dd.pkl --word-embeddings data/embs/en_wiki_vectors_varembed_ud.txt --log-dir log-vit-one-varinit-char-sumloss --use-char-rnn --viterbi --loss-margin one --num-epochs 20
python model.py --dataset results/en_mtags-dd.pkl --word-embeddings data/embs/en_wiki_vectors_varembed_ud.txt --log-dir log-vit-attprop-varinit-char-sumloss --use-char-rnn --viterbi --loss-margin att-prop --num-epochs 20
# After fixing model saving and loading for tests (12/23)
python model.py --dataset results/en_mtags-pos.pkl --log-dir log-vit-zero-onlypos-randinit-nochar-sumloss --viterbi --loss-margin zero --num-epochs 20;
python model.py --dataset results/en_mtags-pos.pkl --log-dir log-vit-zero-onlypos-randinit-char-sumloss --viterbi --use-char-rnn --loss-margin zero --num-epochs 20;
python model.py --dataset results/en_mtags-dd.pkl --log-dir log-vit-zero-randinit-char-sumloss --use-char-rnn --viterbi --loss-margin zero --num-epochs 20;
python model.py --dataset results/en_mtags-dd.pkl --log-dir log-vit-attprop-randinit-char-sumloss --use-char-rnn --viterbi --loss-margin att-prop --num-epochs 20;
python model.py --dataset results/en_mtags-dd.pkl --log-dir log-vit-one-randinit-char-sumloss --use-char-rnn --viterbi --loss-margin one --num-epochs 20;
