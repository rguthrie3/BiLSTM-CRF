import numpy as np

LOGSUM_LEN = 10 # lines
REPEATS = 5 # times per experiment setup

def title_line(line_batch):
    return line_batch[0]

def dev_pos_acc(line_batch):
    return float(line_batch[1].strip().split()[-1])

def test_pos_acc(line_batch):
    return float(line_batch[2].strip().split()[-1])

def dev_mic_att_acc(line_batch):
    return float(line_batch[3].strip().split()[3])

def test_mic_att_acc(line_batch):
    return float(line_batch[4].strip().split()[3])

def dev_mac_att_acc(line_batch):
    return float(line_batch[3].strip().split()[5])

def test_mac_att_acc(line_batch):
    return float(line_batch[4].strip().split()[5])

def dev_pos_oov_acc(line_batch):
    return float(line_batch[5].strip().split()[-1])

def test_pos_oov_acc(line_batch):
    return float(line_batch[6].strip().split()[-1])

langs = "vi hu tr he ta fa hi lv bg cs ru it es da sv en".split()
# langs = ['da', 'en', 'hu', 'it', 'lv', 'tr', 'vi']
# langs = ['ta', 'fa', 'ru', 'sv', 'he', 'bg', 'hi', 'cs', 'es']
# langs = ['ta', 'fa', 'ru', 'sv', 'he', 'bg']
# langs = "hu da lv en".split()
# langs = "hu tr he es ta".split()
# langs = "hi cs es".split()
# langs = ["hi"]
# langs = ["lv"]
# langs = ["en"]
# models = ['nochar-nolc','tagchar-nolc','mchar','bothchar']
models = ['nochar','tagchar','mchar','bothchar']
# models = ['mchar','bothchar']
# models = ['nochar','tagchar']
# models = ['nochar','tagchar','mbochar','bothbochar']

test = True
# test = False
oov = False
# oov = True
# added="-lcbo"
# added="-alllstm"
added=""
# emb_algo = "ft"
emb_algo = "pg"

with open('{}-summary{}{}{}-{}.txt'.format(emb_algo, "-oov" if oov else "", "-test" if test else "", added, "-".join(langs)),'w') as sum_file:
    for lg in langs:
        pos_acc_averages = {}
        pos_oov_acc_averages = {}
        att_f1_averages = {}
        pos_acc_stddevs = {}
        pos_oov_acc_stddevs = {}
        att_f1_stddevs = {}
        for m in models:
            with open('results-{}-{}{}-{}.txt'.format(lg, emb_algo, added, m), 'r') as res_file:
                lines = res_file.readlines()
                res_batches = [lines[i:i+LOGSUM_LEN] for i in xrange(0, len(lines), LOGSUM_LEN)]

                # init file-through reporting vars
                tr_tok_headers = []
                pos_acc_averages[m] = []
                pos_oov_acc_averages[m] = []
                att_f1_averages[m] = []
                pos_acc_stddevs[m] = []
                pos_oov_acc_stddevs[m] = []
                att_f1_stddevs[m] = []

                # init per-setup vars
                tr_toks = 0
                run = 0
                dev_pos_accs = []
                dev_pos_oov_accs = []
                dev_att_f1s = []
                test_pos_accs = []
                test_pos_oov_accs = []
                test_att_f1s = []
                # TODO add macro f1s

                for b in res_batches:
                    tl = title_line(b)
                    assert tl.startswith('.')
                    parts = tl.split('-')
                    curr_tr_toks = int(parts[2])
                    curr_part = int(parts[-1])
                    if curr_part != 1:
                        assert curr_tr_toks == tr_toks
                    else:
                        tr_toks = curr_tr_toks
                    dev_pos_accs.append(dev_pos_acc(b))
                    dev_pos_oov_accs.append(dev_pos_oov_acc(b))
                    dev_att_f1s.append(dev_mic_att_acc(b))
                    test_pos_accs.append(test_pos_acc(b))
                    test_pos_oov_accs.append(test_pos_oov_acc(b))
                    test_att_f1s.append(test_mic_att_acc(b))
                    if curr_part == 5:

                        # add accumlations to reporting
                        tr_tok_headers.append(str(tr_toks))
                        if not test:
                            pos_acc_averages[m].append("{:4f}".format(np.average(dev_pos_accs)))
                            pos_oov_acc_averages[m].append("{:4f}".format(np.average(dev_pos_oov_accs)))
                            att_f1_averages[m].append("{:4f}".format(np.average(dev_att_f1s)))
                            pos_acc_stddevs[m].append("{:4f}".format(np.std(dev_pos_accs)))
                            pos_oov_acc_stddevs[m].append("{:4f}".format(np.std(dev_pos_oov_accs)))
                            att_f1_stddevs[m].append("{:4f}".format(np.std(dev_att_f1s)))
                        else:
                            pos_acc_averages[m].append("{:4f}".format(np.average(test_pos_accs)))
                            pos_oov_acc_averages[m].append("{:4f}".format(np.average(test_pos_oov_accs)))
                            att_f1_averages[m].append("{:4f}".format(np.average(test_att_f1s)))
                            pos_acc_stddevs[m].append("{:4f}".format(np.std(test_pos_accs)))
                            pos_oov_acc_stddevs[m].append("{:4f}".format(np.std(test_pos_oov_accs)))
                            att_f1_stddevs[m].append("{:4f}".format(np.std(test_att_f1s)))

                        # init per-setup vars
                        dev_pos_accs = []
                        dev_pos_oov_accs = []
                        dev_att_f1s = []
                        test_pos_accs = []
                        test_pos_oov_accs = []
                        test_att_f1s = []

        # write all reporting in output file
        sum_file.write("{}-polyglot:\n".format(lg))
        if oov:
            fields = zip([pos_oov_acc_averages], ['POS-acc-oov-avg'])
        else:
            fields = zip([pos_acc_averages, att_f1_averages, pos_acc_stddevs, att_f1_stddevs], ['POS-acc-avg','Att-F1-avg','POS-acc-std','Att-F1-std'])
        for dict, name in fields:
            sum_file.write(name + '\t' + '\t'.join(tr_tok_headers) + '\n')
            for m in models:
                sum_file.write(m + '\t' + '\t'.join(dict[m]) + '\n')

