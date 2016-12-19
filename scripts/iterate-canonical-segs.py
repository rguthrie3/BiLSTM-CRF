import argparse

'''
Creates lowest-level segmentations from hierarchical file
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", required=True, help="Output filename", dest="output_filename")
    options = parser.parse_args()

    segs = {}
    with open(options.input, "r") as f:
        for l in f.read().splitlines():
            w_seg = l.split()
            if len(w_seg) < 4: continue # ignore single-morph words
            segs[w_seg[0]] = w_seg[2:] # ignore "canonical form" in w_seg[1]
    
    final_segs = {}
    added = True
    its = 0
    while added:
        added = False
        segs_to_rem = []
        for w in segs:
            if segs[w] == []: continue
            is_made_of_raw = len([s for s in segs[w] if s in segs]) == 0
            if is_made_of_raw:
                added = True
                final_segs[w] = list(segs[w])
                segs_to_rem.append(w) # wait till end of iteration
            else:
                new_segs = []
                for seg in segs[w]:
                    if seg in final_segs:
                        new_segs.extend(final_segs[seg])
                        added = True
                    else: # it should come around later
                        new_segs.append(seg)
                segs[w] = new_segs
        for wtr in segs_to_rem:
            segs[wtr] = []
        its += 1
        # print its, ": ", final_segs
    
    with open(options.output_filename, "w") as f:
        for k, vals in final_segs.items():
            f.write("{} {}\n".format(k, " ".join(vals)))
    
    unassigned = [s for s in segs if len(segs[s]) > 0]
    print "After {} iterations, {} unassigned words left:\n{}".format(its, len(unassigned), unassigned)
    