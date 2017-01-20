nJ# 12/20
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
# Danish and Italian (12/24?)
python model.py --dataset results/it_mtags-pos.pkl --word-embeddings data/embs/it_wiki_vectors_varembed_ud.txt --log-dir log-it-vit-zero-onlypos-varinit-nochar-sumloss --viterbi --loss-margin zero --num-epochs 20;
python model.py --dataset results/da_mtags-pos.pkl --word-embeddings data/embs/da_wiki_vectors_varembed_ud.txt --log-dir log-da-vit-zero-onlypos-varinit-nochar-sumloss --viterbi --loss-margin zero --num-epochs 20;
python model.py --dataset results/it_mtags-dd.pkl --word-embeddings data/embs/it_wiki_vectors_varembed_ud.txt --log-dir log-it-vit-zero-varinit-nochar-sumloss --viterbi --loss-margin zero --num-epochs 20;
python model.py --dataset results/da_mtags-dd.pkl --word-embeddings data/embs/da_wiki_vectors_varembed_ud.txt --log-dir log-da-vit-zero-varinit-nochar-sumloss --viterbi --loss-margin zero --num-epochs 20;
python model.py --dataset results/it_mtags-pos.pkl --word-embeddings data/embs/it_wiki_vectors_varembed_ud.txt --log-dir log-it-vit-zero-onlypos-varinit-char-sumloss --viterbi --use-char-rnn --loss-margin zero --num-epochs 20;
python model.py --dataset results/da_mtags-pos.pkl --word-embeddings data/embs/da_wiki_vectors_varembed_ud.txt --log-dir log-da-vit-zero-onlypos-varinit-char-sumloss --viterbi --use-char-rnn --loss-margin zero --num-epochs 20;
python model.py --dataset results/it_mtags-dd.pkl --word-embeddings data/embs/it_wiki_vectors_varembed_ud.txt --log-dir log-it-vit-zero-varinit-char-sumloss --viterbi --use-char-rnn --loss-margin zero --num-epochs 20;
python model.py --dataset results/da_mtags-dd.pkl --word-embeddings data/embs/da_wiki_vectors_varembed_ud.txt --log-dir log-da-vit-zero-varinit-char-sumloss --viterbi --use-char-rnn --loss-margin zero --num-epochs 20;
python model.py --dataset results/it_mtags-pos.pkl --word-embeddings data/embs/it_wiki_vectors_varembed_ud.txt --log-dir log-it-noseq-onlypos-varinit-nochar --no-sequence-model --num-epochs 20;
python model.py --dataset results/da_mtags-pos.pkl --word-embeddings data/embs/da_wiki_vectors_varembed_ud.txt --log-dir log-da-noseq-onlypos-varinit-nochar --no-sequence-model --num-epochs 20;
python model.py --dataset results/it_mtags-dd.pkl --word-embeddings data/embs/it_wiki_vectors_varembed_ud.txt --log-dir log-it-noseq-varinit-nochar --no-sequence-model --num-epochs 20;
python model.py --dataset results/da_mtags-dd.pkl --word-embeddings data/embs/da_wiki_vectors_varembed_ud.txt --log-dir log-da-noseq-varinit-nochar --no-sequence-model --num-epochs 20;
python model.py --dataset results/it_mtags-pos.pkl --word-embeddings data/embs/it_wiki_vectors_varembed_ud.txt --log-dir log-it-noseq-onlypos-varinit-char --no-sequence-model --use-char-rnn --num-epochs 20;
python model.py --dataset results/da_mtags-pos.pkl --word-embeddings data/embs/da_wiki_vectors_varembed_ud.txt --log-dir log-da-noseq-onlypos-varinit-char --no-sequence-model --use-char-rnn --num-epochs 20;
python model.py --dataset results/it_mtags-dd.pkl --word-embeddings data/embs/it_wiki_vectors_varembed_ud.txt --log-dir log-it-noseq-varinit-char --no-sequence-model --use-char-rnn --num-epochs 20;
python model.py --dataset results/da_mtags-dd.pkl --word-embeddings data/embs/da_wiki_vectors_varembed_ud.txt --log-dir log-da-noseq-varinit-char --no-sequence-model --use-char-rnn --num-epochs 20;
python model.py --dataset results/it_mtags-pos.pkl --log-dir log-it-noseq-onlypos-randinit-nochar --no-sequence-model --num-epochs 20;
python model.py --dataset results/da_mtags-pos.pkl --log-dir log-da-noseq-onlypos-randinit-nochar --no-sequence-model --num-epochs 20;
python model.py --dataset results/it_mtags-dd.pkl --log-dir log-it-noseq-randinit-nochar --no-sequence-model --num-epochs 20;
python model.py --dataset results/da_mtags-dd.pkl --log-dir log-da-noseq-randinit-nochar --no-sequence-model --num-epochs 20;
python model.py --dataset results/it_mtags-pos.pkl --log-dir log-it-noseq-onlypos-randinit-char --no-sequence-model --use-char-rnn --num-epochs 20;
python model.py --dataset results/da_mtags-pos.pkl --log-dir log-da-noseq-onlypos-randinit-char --no-sequence-model --use-char-rnn --num-epochs 20;
python model.py --dataset results/it_mtags-dd.pkl --log-dir log-it-noseq-randinit-char --no-sequence-model --use-char-rnn --num-epochs 20;
python model.py --dataset results/da_mtags-dd.pkl --log-dir log-da-noseq-randinit-char --no-sequence-model --use-char-rnn --num-epochs 20;
# CELEX initialization (1/4, then 1/18 again):
python model.py --dataset results/en_mtags-pos.pkl --word-embeddings data/embs/en_celex_vectors_varembed_ud.txt --log-dir log-onlypos-noseq-celinit-char-05dr --dropout 0.5 --use-char-rnn --no-sequence-model --num-epochs 40;
python model.py --dataset results/en_mtags-dd.pkl --word-embeddings data/embs/en_celex_vectors_varembed_ud.txt --log-dir log-noseq-celinit-char-05dr-sumtags --dropout 0.5 --use-char-rnn --no-sequence-model --num-epochs 40;
python model.py --dataset results/en_mtags-dd.pkl --word-embeddings data/embs/en_celex_vectors_varembed_ud.txt --log-dir log-noseq-celinit-nochar-05dr-sumtags --dropout 0.5 --no-sequence-model --num-epochs 40;
# Checking on 1.2 dataset (1/4):
python model.py --dataset results/en_mtags-12-pos.pkl --word-embeddings data/embs/en_wiki_vectors_varembed_ud.txt --log-dir log-ud12-onlypos-noseq-varinit-char-05dr --dropout 0.5 --use-char-rnn --no-sequence-model --num-epochs 40;
python model.py --dataset results/en_mtags-12-dd.pkl --word-embeddings data/embs/en_wiki_vectors_varembed_ud.txt --log-dir log-ud12-noseq-varinit-char-05dr-sumtags --dropout 0.5 --use-char-rnn --no-sequence-model --num-epochs 40;
python model.py --dataset results/en_mtags-12-dd.pkl --word-embeddings data/embs/en_wiki_vectors_varembed_ud.txt --log-dir log-ud12-noseq-varinit-nochar-05dr-sumtags --dropout 0.5 --no-sequence-model --num-epochs 40;
# Char-LSTM embeddings (1/4):
python model.py --dataset results/en_mtags-pos.pkl --word-embeddings data/embs/en-char-lstm-embeddings.txt --log-dir log-onlypos-noseq-charinit-char-05dr --dropout 0.5 --use-char-rnn --no-sequence-model --num-epochs 40;
python model.py --dataset results/en_mtags-dd.pkl --word-embeddings data/embs/en-char-lstm-embeddings.txt --log-dir log-noseq-charinit-char-05dr-sumtags --dropout 0.5 --use-char-rnn --no-sequence-model --num-epochs 40;
python model.py --dataset results/en_mtags-dd.pkl --word-embeddings data/embs/en-char-lstm-embeddings.txt --log-dir log-noseq-charinit-nochar-05dr-sumtags --dropout 0.5 --no-sequence-model --num-epochs 40;
# Retesting basic LSTM arch for sum loss instead of average (1/4):
python model.py --dataset results/en_mtags-pos.pkl --word-embeddings data/embs/en_wiki_vectors_varembed_ud.txt --log-dir log-onlypos-noseq-varinit-char-05dr --dropout 0.5 --use-char-rnn --no-sequence-model --num-epochs 40;
python model.py --dataset results/en_mtags-dd.pkl --word-embeddings data/embs/en_wiki_vectors_varembed_ud.txt --log-dir log-noseq-varinit-char-05dr-sumtags --dropout 0.5 --use-char-rnn --no-sequence-model --num-epochs 40;
python model.py --dataset results/en_mtags-dd.pkl --word-embeddings data/embs/en_wiki_vectors_varembed_ud.txt --log-dir log-noseq-varinit-nochar-05dr-sumtags --dropout 0.5 --no-sequence-model --num-epochs 40;
python model.py --dataset results/en_mtags-dd.pkl --log-dir log-noseq-randinit-char-05dr-sumtags --dropout 0.5 --use-char-rnn --no-sequence-model --num-epochs 40;
python model.py --dataset results/en_mtags-dd.pkl --log-dir log-noseq-randinit-nochar-05dr-sumtags --dropout 0.5 --no-sequence-model --num-epochs 40;
# Jan 5
python model.py --dataset results/en_mtags-pos.pkl --log-dir log-onlypos-noseq-randinit-char-05dr --dropout 0.5 --use-char-rnn --no-sequence-model --num-epochs 40;
python model.py --dataset results/en_mtags-pos.pkl --log-dir log-onlypos-noseq-randinit-nochar-05dr --dropout 0.5 --no-sequence-model --num-epochs 40;
python model.py --dataset results/en_mtags-pos.pkl --word-embeddings data/embs/en_wiki_vectors_varembed_ud.txt --log-dir log-onlypos-noseq-varinit-nochar-05dr --dropout 0.5 --no-sequence-model --num-epochs 40;
python model.py --dataset results/en_mtags-pos.pkl --word-embeddings data/embs/en-char-lstm-embeddings.txt --log-dir log-onlypos-noseq-charinit-nochar-05dr --dropout 0.5 --no-sequence-model --num-epochs 40;
python model.py --dataset results/en_mtags-pos.pkl --word-embeddings data/embs/en_celex_vectors_varembed_ud.txt --log-dir log-onlypos-noseq-celinit-nochar-05dr --dropout 0.5 --no-sequence-model --num-epochs 40;
# Jan 6
python model.py --dataset results/it_mtags-pos.pkl --word-embeddings data/embs/it-char-embeds.txt --log-dir log-it-onlypos-noseq-charinit-char-05dr --dropout 0.5 --use-char-rnn --no-sequence-model --num-epochs 40;
python model.py --dataset results/it_mtags-pos.pkl --word-embeddings data/embs/it-char-embeds.txt --log-dir log-it-onlypos-noseq-charinit-nochar-05dr --dropout 0.5 --no-sequence-model --num-epochs 40;
python model.py --dataset results/it_mtags-dd.pkl --word-embeddings data/embs/it-char-embeds.txt --log-dir log-it-noseq-charinit-char-05dr-sumtags --dropout 0.5 --use-char-rnn --no-sequence-model --num-epochs 40;
python model.py --dataset results/it_mtags-dd.pkl --word-embeddings data/embs/it-char-embeds.txt --log-dir log-it-noseq-charinit-nochar-05dr-sumtags --dropout 0.5 --no-sequence-model --num-epochs 40;
python model.py --dataset results/it_mtags-pos.pkl --log-dir log-it-noseq-onlypos-randinit-nochar-05dr --no-sequence-model --dropout 0.5 --num-epochs 40;
python model.py --dataset results/it_mtags-dd.pkl --log-dir log-it-noseq-randinit-nochar-05dr --no-sequence-model --dropout 0.5 --num-epochs 40;
python model.py --dataset results/it_mtags-pos.pkl --log-dir log-it-noseq-onlypos-randinit-char-05dr --no-sequence-model --use-char-rnn --dropout 0.5 --num-epochs 40;
python model.py --dataset results/it_mtags-dd.pkl --log-dir log-it-noseq-randinit-char-05dr --no-sequence-model --use-char-rnn --dropout 0.5 --num-epochs 40;
python model.py --dataset results/it_mtags-pos.pkl --word-embeddings data/embs/it_wiki_vectors_varembed_ud.txt --log-dir log-it-noseq-onlypos-varinit-nochar-05dr --no-sequence-model --dropout 0.5 --num-epochs 40;
python model.py --dataset results/it_mtags-dd.pkl --word-embeddings data/embs/it_wiki_vectors_varembed_ud.txt --log-dir log-it-noseq-varinit-nochar-05dr --no-sequence-model --dropout 0.5 --num-epochs 40;
python model.py --dataset results/it_mtags-pos.pkl --word-embeddings data/embs/it_wiki_vectors_varembed_ud.txt --log-dir log-it-noseq-onlypos-varinit-char-05dr --no-sequence-model --use-char-rnn --dropout 0.5 --num-epochs 40;
python model.py --dataset results/it_mtags-dd.pkl --word-embeddings data/embs/it_wiki_vectors_varembed_ud.txt --log-dir log-it-noseq-varinit-char-05dr --no-sequence-model --use-char-rnn --dropout 0.5 --num-epochs 40;
# Jan 8
python model.py --dataset results/da_mtags-pos.pkl --word-embeddings data/embs/da-char-embeds.txt --log-dir log-da-onlypos-noseq-charinit-char-05dr --dropout 0.5 --use-char-rnn --no-sequence-model --num-epochs 40;
python model.py --dataset results/da_mtags-pos.pkl --word-embeddings data/embs/da-char-embeds.txt --log-dir log-da-onlypos-noseq-charinit-nochar-05dr --dropout 0.5 --no-sequence-model --num-epochs 40;
python model.py --dataset results/da_mtags-dd.pkl --word-embeddings data/embs/da-char-embeds.txt --log-dir log-da-noseq-charinit-char-05dr-sumtags --dropout 0.5 --use-char-rnn --no-sequence-model --num-epochs 40;
python model.py --dataset results/da_mtags-dd.pkl --word-embeddings data/embs/da-char-embeds.txt --log-dir log-da-noseq-charinit-nochar-05dr-sumtags --dropout 0.5 --no-sequence-model --num-epochs 40;
python model.py --dataset results/da_mtags-pos.pkl --log-dir log-da-noseq-onlypos-randinit-nochar-05dr --no-sequence-model --dropout 0.5 --num-epochs 40;
python model.py --dataset results/da_mtags-dd.pkl --log-dir log-da-noseq-randinit-nochar-05dr --no-sequence-model --dropout 0.5 --num-epochs 40;
python model.py --dataset results/da_mtags-pos.pkl --log-dir log-da-noseq-onlypos-randinit-char-05dr --no-sequence-model --use-char-rnn --dropout 0.5 --num-epochs 40;
python model.py --dataset results/da_mtags-dd.pkl --log-dir log-da-noseq-randinit-char-05dr --no-sequence-model --use-char-rnn --dropout 0.5 --num-epochs 40;
python model.py --dataset results/da_mtags-pos.pkl --word-embeddings data/embs/da_wiki_vectors_varembed_ud.txt --log-dir log-da-noseq-onlypos-varinit-nochar-05dr --no-sequence-model --dropout 0.5 --num-epochs 40;
python model.py --dataset results/da_mtags-dd.pkl --word-embeddings data/embs/da_wiki_vectors_varembed_ud.txt --log-dir log-da-noseq-varinit-nochar-05dr --no-sequence-model --dropout 0.5 --num-epochs 40;
python model.py --dataset results/da_mtags-pos.pkl --word-embeddings data/embs/da_wiki_vectors_varembed_ud.txt --log-dir log-da-noseq-onlypos-varinit-char-05dr --no-sequence-model --use-char-rnn --dropout 0.5 --num-epochs 40;
python model.py --dataset results/da_mtags-dd.pkl --word-embeddings data/embs/da_wiki_vectors_varembed_ud.txt --log-dir log-da-noseq-varinit-char-05dr --no-sequence-model --use-char-rnn --dropout 0.5 --num-epochs 40;
# Jan 13 - Polyglot
python model.py --dataset results/en_mtags-dd.pkl --word-embeddings data/embs/en_polyglot_vectors_ud.txt --log-dir log-noseq-pginit-nochar-05dr-sumtags --dropout 0.5 --no-sequence-model --num-epochs 40;
python model.py --dataset results/en_mtags-dd.pkl --word-embeddings data/embs/en_polyglot_vectors_ud.txt --log-dir log-noseq-pginit-char-05dr-sumtags --dropout 0.5 --no-sequence-model --use-char-rnn --num-epochs 40;
python model.py --dataset results/en_mtags-pos.pkl --word-embeddings data/embs/en_polyglot_vectors_ud.txt --log-dir log-onlypos-noseq-pginit-nochar-05dr-sumtags --dropout 0.5 --no-sequence-model --num-epochs 40;
python model.py --dataset results/en_mtags-pos.pkl --word-embeddings data/embs/en_polyglot_vectors_ud.txt --log-dir log-onlypos-noseq-pginit-char-05dr-sumtags --dropout 0.5 --no-sequence-model --use-char-rnn --num-epochs 40;
python model.py --dataset results/da_mtags-dd.pkl --word-embeddings data/embs/da_polyglot_vectors_ud.txt --log-dir log-da-noseq-pginit-nochar-05dr-sumtags --dropout 0.5 --no-sequence-model --num-epochs 40;
python model.py --dataset results/da_mtags-dd.pkl --word-embeddings data/embs/da_polyglot_vectors_ud.txt --log-dir log-da-noseq-pginit-char-05dr-sumtags --dropout 0.5 --no-sequence-model --use-char-rnn --num-epochs 40;
python model.py --dataset results/da_mtags-pos.pkl --word-embeddings data/embs/da_polyglot_vectors_ud.txt --log-dir log-da-onlypos-noseq-pginit-nochar-05dr-sumtags --dropout 0.5 --no-sequence-model --num-epochs 40;
python model.py --dataset results/da_mtags-pos.pkl --word-embeddings data/embs/da_polyglot_vectors_ud.txt --log-dir log-da-onlypos-noseq-pginit-char-05dr-sumtags --dropout 0.5 --no-sequence-model --use-char-rnn --num-epochs 40;
python model.py --dataset results/it_mtags-dd.pkl --word-embeddings data/embs/it_polyglot_vectors_ud.txt --log-dir log-it-noseq-pginit-nochar-05dr-sumtags --dropout 0.5 --no-sequence-model --num-epochs 40;
python model.py --dataset results/it_mtags-dd.pkl --word-embeddings data/embs/it_polyglot_vectors_ud.txt --log-dir log-it-noseq-pginit-char-05dr-sumtags --dropout 0.5 --no-sequence-model --use-char-rnn --num-epochs 40;
python model.py --dataset results/it_mtags-pos.pkl --word-embeddings data/embs/it_polyglot_vectors_ud.txt --log-dir log-it-onlypos-noseq-pginit-nochar-05dr-sumtags --dropout 0.5 --no-sequence-model --num-epochs 40;
python model.py --dataset results/it_mtags-pos.pkl --word-embeddings data/embs/it_polyglot_vectors_ud.txt --log-dir log-it-onlypos-noseq-pginit-char-05dr-sumtags --dropout 0.5 --no-sequence-model --use-char-rnn --num-epochs 40;
# Jan ??? - Semi-supervised
python model.py --dataset results/en_mtags-dd.pkl --log-dir log-noseq-semisup-randinit-nochar-05dr-sumtags --dropout 0.5 --no-sequence-model --semi-supervised --dynet-mem 250 --num-epochs 40;
python model.py --dataset results/en_mtags-dd.pkl --log-dir log-noseq-semisup-randinit-char-05dr-sumtags --dropout 0.5 --no-sequence-model --use-char-rnn --semi-supervised --dynet-mem 250 --num-epochs 40;
python model.py --dataset results/en_mtags-dd.pkl --word-embeddings data/embs/en_wiki_vectors_varembed_ud.txt --log-dir log-noseq-semisup-varinit-nochar-05dr-sumtags --dropout 0.5 --no-sequence-model --semi-supervised --dynet-mem 250 --num-epochs 40;
python model.py --dataset results/en_mtags-dd.pkl --word-embeddings data/embs/en_wiki_vectors_varembed_ud.txt --log-dir log-noseq-semisup-varinit-char-05dr-sumtags --dropout 0.5 --no-sequence-model --semi-supervised --dynet-mem 250 --use-char-rnn --num-epochs 40;
python model.py --dataset results/en_mtags-pos.pkl --log-dir log-onlypos-noseq-semisup-randinit-nochar-05dr-sumtags --dropout 0.5 --no-sequence-model --semi-supervised --dynet-mem 250 --num-epochs 40;
python model.py --dataset results/en_mtags-pos.pkl --log-dir log-onlypos-noseq-semisup-randinit-char-05dr-sumtags --dropout 0.5 --no-sequence-model --use-char-rnn --semi-supervised --dynet-mem 250 --num-epochs 40;
python model.py --dataset results/en_mtags-pos.pkl --word-embeddings data/embs/en_wiki_vectors_varembed_ud.txt --log-dir log-onlypos-noseq-semisup-varinit-nochar-05dr-sumtags --dropout 0.5 --no-sequence-model --semi-supervised --dynet-mem 250 --num-epochs 40;
python model.py --dataset results/en_mtags-pos.pkl --word-embeddings data/embs/en_wiki_vectors_varembed_ud.txt --log-dir log-onlypos-noseq-semisup-varinit-char-05dr-sumtags --dropout 0.5 --no-sequence-model --semi-supervised --dynet-mem 250 --use-char-rnn --num-epochs 40;
# More for re-running CELEX (1/18 - see 1/4 and 1/5 for more celinit)
python model.py --dataset results/en_mtags-pos.pkl --word-embeddings data/embs/en_celex_vectors_varembed_ud-nolc.txt --log-dir log-onlypos-noseq-celinit-nolc-nochar-05dr --dropout 0.5 --no-sequence-model --num-epochs 40;
python model.py --dataset results/en_mtags-pos.pkl --word-embeddings data/embs/en_celex_vectors_varembed_ud-morf.txt --log-dir log-onlypos-noseq-celinit-morf-nochar-05dr --dropout 0.5 --no-sequence-model --num-epochs 40;
python model.py --dataset results/en_mtags-pos.pkl --word-embeddings data/embs/en_celex_vectors_varembed_ud-morf.txt --log-dir log-onlypos-noseq-celinit-morf-char-05dr --dropout 0.5 --use-char-rnn --no-sequence-model --num-epochs 40;
python model.py --dataset results/en_mtags-dd.pkl --word-embeddings data/embs/en_celex_vectors_varembed_ud-morf.txt --log-dir log-noseq-celinit-morf-char-05dr-sumtags --dropout 0.5 --use-char-rnn --no-sequence-model --num-epochs 40;
python model.py --dataset results/en_mtags-dd.pkl --word-embeddings data/embs/en_celex_vectors_varembed_ud-morf.txt --log-dir log-noseq-celinit-morf-nochar-05dr-sumtags --dropout 0.5 --no-sequence-model --num-epochs 40;
# Canonical (Jan 20)
python model.py --dataset results/en_mtags-pos.pkl --word-embeddings data/embs/en_canonical_vectors_varembed_ud-morf.txt --log-dir log-onlypos-noseq-caninit-morf-nochar-05dr --dropout 0.5 --no-sequence-model --num-epochs 40;
python model.py --dataset results/en_mtags-pos.pkl --word-embeddings data/embs/en_canonical_vectors_varembed_ud-morf.txt --log-dir log-onlypos-noseq-caninit-morf-char-05dr --dropout 0.5 --use-char-rnn --no-sequence-model --num-epochs 40;
python model.py --dataset results/en_mtags-dd.pkl --word-embeddings data/embs/en_canonical_vectors_varembed_ud-morf.txt --log-dir log-noseq-caninit-morf-char-05dr-sumtags --dropout 0.5 --use-char-rnn --no-sequence-model --num-epochs 40;
python model.py --dataset results/en_mtags-dd.pkl --word-embeddings data/embs/en_canonical_vectors_varembed_ud-morf.txt --log-dir log-noseq-caninit-morf-nochar-05dr-sumtags --dropout 0.5 --no-sequence-model --num-epochs 40;
