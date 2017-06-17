for pos in NOUN VERB ADV ADJ
do
python make_dataset.py --training-data ../data/ud/UD_Vietnamese/vi-ud-train.conllu --dev-data ../data/ud/UD_Vietnamese/vi-ud-dev.conllu --test-data ../data/ud/UD_Vietnamese/vi-ud-test.conllu --tags-included ${pos} --ud-tags -o data/vi-ud-${pos}s-mtpred.pkl --vocab-file data/vi-vocab-${pos}s.txt
done
