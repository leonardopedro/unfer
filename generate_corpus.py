#!/usr/bin/env python3
"""Generate L0 seed corpus (200 grammatical) and ungrammatical corpus (50)."""
import json, os

NUMERALS = {"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,
            "six":6,"seven":7,"eight":8,"nine":9,"ten":10}
PROPER = ["John","Mary","Bob","Alice"]
DETS = ["the","a"]
NOUNS = ["cat","dog","number"]
TRANS_V = ["loves","sees","likes","eats"]
INTRANS_V = ["sleeps","runs"]
DITRANS_V = ["adds","multiplies","subtracts"]
COP_V = ["is","equals"]
COMP_V = ["greater","less"]
ADJ = ["big","small","red","blue"]

def L(word, cat):
    return {"type":"Leaf","word":word,"category":cat}

def FA(l, r, res):
    return {"type":"Application","direction":"forward","result_category":res,"left":l,"right":r}

def BA(l, r, res):
    return {"type":"Application","direction":"backward","result_category":res,"left":l,"right":r}

def FC(l, r, res):
    return {"type":"Composition","direction":"forward","result_category":res,"left":l,"right":r}

def BC(l, r, res):
    return {"type":"Composition","direction":"backward","result_category":res,"left":l,"right":r}

def mk_proper(name):
    return L(name, "NP")

def mk_numeral(name):
    return L(name, "NP")

def mk_det_n(det, noun):
    return FA(L(det,"NP/N"), L(noun,"N"), "NP")

def mk_det_adj_n(det, adj, noun):
    return FA(L(det,"NP/N"), FA(L(adj,"N/N"), L(noun,"N"), "N"), "NP")

def mk_det_n_rel(det, noun, rel_pron, vp_deriv):
    np_part = FA(L(det,"NP/N"), L(noun,"N"), "NP")
    rel_mod = FA(L(rel_pron,"(NP\\NP)/(S\\NP)"), vp_deriv, "NP\\NP")
    return BA(np_part, rel_mod, "NP")

def mk_np_conj(np1, np2):
    return FA(FA(np1, L("and","conj"), "NP/conj"), np2, "NP")

def mk_trans_vp(v, obj):
    return FA(L(v,"(S\\NP)/NP"), obj, "S\\NP")

def mk_intrans_vp(v):
    return L(v, "S\\NP")

def mk_neg_vp(vp):
    return BA(L("not","(S\\NP)\\(S\\NP)"), vp, "S\\NP")

def mk_ditrans_vp(v, obj1, obj2):
    step1 = FA(L(v,"((S\\NP)/NP)/NP"), obj1, "(S\\NP)/NP")
    return FA(step1, obj2, "S\\NP")

def mk_copula(vp, obj):
    return FA(vp, obj, "S\\NP")

def mk_sentence(np_deriv, vp_deriv):
    return BA(np_deriv, vp_deriv, "S")

def mk_coord_vp(vp1, vp2):
    conj_vp = FA(vp1, L("and","conj"), "(S\\NP)/conj")
    return FA(conj_vp, vp2, "S\\NP")

def compute_arith(v, a, b):
    if v == "adds": return a + b
    if v == "multiplies": return a * b
    if v == "subtracts": return a - b
    return None

def compute_comp(v, a, b):
    if v == "greater": return a > b
    if v == "less": return a < b
    if v in ("equals", "is"): return a == b
    return None

sentences = []
sid = 0

def add(s, d, r):
    global sid
    sid += 1
    sentences.append({"id": sid, "sentence": s, "derivation": d, "result": r})

def cycle(lst, idx):
    return lst[idx % len(lst)]

# ============================================================
# 1. Simple transitive (22)
# ============================================================
for subj, v, obj in [
    ("John","loves","Mary"), ("Mary","sees","Bob"), ("Bob","likes","Alice"),
    ("Alice","loves","John"), ("John","sees","Alice"), ("Mary","likes","John"),
    ("Bob","eats","cat"), ("Alice","sees","dog"), ("John","likes","Bob"),
    ("Mary","loves","Alice"), ("Bob","sees","John"), ("Alice","likes","Mary"),
    ("John","eats","number"), ("Mary","eats","cat"), ("Bob","loves","dog"),
    ("Alice","sees","number"), ("John","likes","dog"), ("Mary","sees","cat"),
    ("Bob","likes","Alice"), ("Alice","loves","Bob"), ("John","sees","Mary"),
    ("Mary","likes","Alice"),
]:
    d = mk_sentence(mk_proper(subj), mk_trans_vp(v, mk_proper(obj)))
    add(f"{subj} {v} {obj}.", d, f"{v}({subj},{obj})")

# ============================================================
# 2. Simple intransitive (12)
# ============================================================
for subj, v in [
    ("John","sleeps"), ("Mary","runs"), ("Bob","sleeps"), ("Alice","runs"),
    ("John","runs"), ("Mary","sleeps"), ("Bob","runs"), ("Alice","sleeps"),
    ("John","sleeps"), ("Mary","runs"), ("Bob","sleeps"), ("Alice","runs"),
]:
    d = mk_sentence(mk_proper(subj), mk_intrans_vp(v))
    add(f"{subj} {v}.", d, f"{v}({subj})")

# ============================================================
# 3. Ditransitive (12)
# ============================================================
for subj, v, o1, o2, exp in [
    ("John","adds","two","three",5), ("Mary","multiplies","two","four",8),
    ("Bob","subtracts","five","one",4), ("Alice","adds","zero","ten",10),
    ("John","multiplies","three","three",9), ("Mary","adds","one","two",3),
    ("Bob","multiplies","two","five",10), ("Alice","subtracts","ten","three",7),
    ("John","adds","four","four",8), ("Mary","subtracts","seven","two",5),
    ("Bob","adds","six","one",7), ("Alice","multiplies","two","three",6),
]:
    d = mk_sentence(mk_proper(subj), mk_ditrans_vp(v, mk_numeral(o1), mk_numeral(o2)))
    add(f"{subj} {v} {o1} {o2}.", d, str(exp))

# ============================================================
# 4. Copula (12)
# ============================================================
for subj, v, obj in [
    ("John","is","Mary"), ("Bob","is","Alice"), ("zero","equals","zero"),
    ("one","equals","one"), ("two","equals","two"), ("three","equals","three"),
    ("John","is","John"), ("Mary","is","Mary"), ("four","equals","four"),
    ("five","equals","five"), ("Bob","is","Bob"), ("Alice","is","Alice"),
]:
    ns = mk_numeral(subj) if subj in NUMERALS else mk_proper(subj)
    no = mk_numeral(obj) if obj in NUMERALS else mk_proper(obj)
    d = mk_sentence(ns, mk_copula(L(v,"(S\\NP)/NP"), no))
    a, b = NUMERALS.get(subj, subj), NUMERALS.get(obj, obj)
    add(f"{subj} {v} {obj}.", d, str(compute_comp(v, a, b)))

# ============================================================
# 5. NP with Det+N (22)
# ============================================================
for det, noun, v in [
    ("the","cat","sleeps"), ("the","dog","runs"), ("a","cat","sleeps"),
    ("a","dog","runs"), ("the","number","sleeps"), ("a","number","runs"),
    ("the","cat","runs"), ("the","dog","sleeps"), ("a","cat","runs"),
    ("the","number","runs"), ("a","dog","sleeps"), ("the","cat","sleeps"),
    ("a","number","sleeps"), ("the","dog","runs"), ("a","cat","sleeps"),
    ("the","cat","runs"), ("a","dog","runs"), ("the","number","sleeps"),
    ("a","cat","sleeps"), ("the","dog","runs"),
]:
    d = mk_sentence(mk_det_n(det, noun), mk_intrans_vp(v))
    add(f"{det} {noun} {v}.", d, f"{v}({det} {noun})")

# ============================================================
# 6. NP with adjective (12)
# ============================================================
for i, (det, adj, noun) in enumerate([
    ("the","big","cat"), ("the","small","dog"), ("a","red","cat"),
    ("the","blue","dog"), ("a","big","number"), ("the","small","cat"),
    ("a","red","dog"), ("the","blue","cat"), ("a","big","dog"),
    ("the","small","number"), ("a","blue","cat"), ("the","red","dog"),
]):
    v = cycle(TRANS_V, i)
    obj = cycle(PROPER, i+1)
    d = mk_sentence(mk_det_adj_n(det, adj, noun), mk_trans_vp(v, mk_proper(obj)))
    add(f"{det} {adj} {noun} {v} {obj}.", d, f"{v}({det} {adj} {noun},{obj})")

# ============================================================
# 7. Relative clause (10)
# ============================================================
for det, noun, rp, v_inner, v_outer, obj in [
    ("the","cat","that","sleeps","loves","John"),
    ("the","dog","that","runs","sees","Mary"),
    ("a","cat","that","sleeps","likes","Bob"),
    ("the","dog","that","sleeps","eats","Alice"),
    ("a","number","that","runs","sees","John"),
    ("the","cat","that","runs","loves","Mary"),
    ("a","dog","that","sleeps","likes","Alice"),
    ("the","number","that","runs","sees","Bob"),
    ("a","cat","that","runs","eats","John"),
    ("the","dog","that","sleeps","likes","Mary"),
]:
    np = mk_det_n_rel(det, noun, rp, mk_intrans_vp(v_inner))
    d = mk_sentence(np, mk_trans_vp(v_outer, mk_proper(obj)))
    add(f"{det} {noun} {rp} {v_inner} {v_outer} {obj}.", d,
        f"{v_outer}({det} {noun} {rp} {v_inner},{obj})")

# ============================================================
# 8. Coordination NP (12)
# ============================================================
for p1, p2, v, obj in [
    ("John","Mary","loves","Bob"), ("Bob","Alice","sees","John"),
    ("John","Bob","likes","Alice"), ("Mary","Alice","loves","John"),
    ("John","Alice","sees","Mary"), ("Mary","Bob","eats","Alice"),
    ("John","Mary","likes","Bob"), ("Bob","Alice","loves","John"),
    ("John","Bob","eats","Alice"), ("Mary","Alice","sees","Bob"),
    ("John","Alice","likes","Mary"), ("Mary","Bob","loves","Alice"),
]:
    np = mk_np_conj(mk_proper(p1), mk_proper(p2))
    d = mk_sentence(np, mk_trans_vp(v, mk_proper(obj)))
    add(f"{p1} and {p2} {v} {obj}.", d, f"{v}({p1} and {p2},{obj})")

# ============================================================
# 9. Coordination VP (8)
# ============================================================
for subj, v1, v2 in [
    ("John","sleeps","runs"), ("Mary","runs","sleeps"),
    ("Bob","sleeps","runs"), ("Alice","runs","sleeps"),
    ("John","runs","sleeps"), ("Mary","sleeps","runs"),
    ("Bob","runs","sleeps"), ("Alice","sleeps","runs"),
]:
    d = mk_sentence(mk_proper(subj), mk_coord_vp(mk_intrans_vp(v1), mk_intrans_vp(v2)))
    add(f"{subj} {v1} and {v2}.", d, f"({v1} and {v2})({subj})")

# ============================================================
# 10. Numeral literals (15)
# ============================================================
for s, v, o in [
    ("zero","equals","zero"), ("one","equals","one"), ("two","equals","two"),
    ("three","equals","three"), ("four","equals","four"), ("five","equals","five"),
    ("six","equals","six"), ("seven","equals","seven"), ("eight","equals","eight"),
    ("nine","equals","nine"), ("ten","equals","ten"),
    ("zero","is","zero"), ("one","is","one"), ("two","is","two"),
    ("three","is","three"),
]:
    d = mk_sentence(mk_numeral(s), mk_copula(L(v,"(S\\NP)/NP"), mk_numeral(o)))
    add(f"{s} {v} {o}.", d, str(compute_comp(v, NUMERALS[s], NUMERALS[o])))

# ============================================================
# 11. Negation (8)
# ============================================================
for subj, v in [
    ("John","sleeps"), ("Mary","runs"), ("Bob","sleeps"),
    ("Alice","runs"), ("John","runs"), ("Mary","sleeps"),
    ("Bob","runs"), ("Alice","sleeps"),
]:
    d = mk_sentence(mk_proper(subj), mk_neg_vp(mk_intrans_vp(v)))
    add(f"{subj} {v} not.", d, f"not({v}({subj}))")

# ============================================================
# 12. Compound (12) — det adj noun that VP verb det adj noun
# ============================================================
for d0,a0,n0,rp,v0,v1, np2_fn, obj_str in [
    ("the","big","cat","that","runs","sees",
     lambda: mk_det_adj_n("the","small","dog"), "the small dog"),
    ("a","big","cat","that","runs","eats",
     lambda: mk_det_adj_n("the","small","dog"), "the small dog"),
    ("the","small","dog","that","sleeps","loves",
     lambda: mk_proper("John"), "John"),
    ("a","blue","cat","that","runs","sees",
     lambda: mk_proper("Mary"), "Mary"),
    ("the","big","dog","that","sleeps","eats",
     lambda: mk_det_adj_n("the","red","cat"), "the red cat"),
    ("a","small","cat","that","runs","likes",
     lambda: mk_proper("Bob"), "Bob"),
    ("the","red","number","that","runs","sees",
     lambda: mk_proper("Alice"), "Alice"),
    ("a","big","dog","that","sleeps","loves",
     lambda: mk_det_adj_n("the","small","cat"), "the small cat"),
    ("the","blue","cat","that","runs","eats",
     lambda: mk_proper("John"), "John"),
    ("a","small","dog","that","sleeps","sees",
     lambda: mk_det_adj_n("the","big","cat"), "the big cat"),
    ("the","big","cat","that","runs","likes",
     lambda: mk_det_adj_n("the","small","dog"), "the small dog"),
    ("a","red","dog","that","sleeps","eats",
     lambda: mk_proper("Mary"), "Mary"),
]:
    np1 = mk_det_n_rel(d0, n0, rp, mk_intrans_vp(v0))
    np2 = np2_fn()
    d = mk_sentence(np1, mk_trans_vp(v1, np2))
    add(f"{d0} {a0} {n0} {rp} {v0} {v1} {obj_str}.", d,
        f"{v1}({d0} {n0} {rp} {v0},{obj_str})")

# ============================================================
# 13. Arithmetic with numerals (15)
# ============================================================
for s, v, o, exp in [
    ("two","adds","three",5), ("two","multiplies","three",6),
    ("five","subtracts","two",3), ("three","adds","four",7),
    ("two","multiplies","five",10), ("seven","subtracts","three",4),
    ("one","adds","one",2), ("three","multiplies","three",9),
    ("ten","subtracts","one",9), ("four","adds","five",9),
    ("two","multiplies","four",8), ("six","subtracts","two",4),
    ("three","adds","two",5), ("five","multiplies","two",10),
    ("eight","subtracts","three",5),
]:
    d = mk_sentence(mk_numeral(s), mk_copula(L(v,"(S\\NP)/NP"), mk_numeral(o)))
    add(f"{s} {v} {o}.", d, str(exp))

# ============================================================
# 14. Mixed: transitive with det-n objects (30 to reach 200)
# ============================================================
for subj, v, det, noun in [
    ("John","loves","the","cat"), ("Mary","sees","a","dog"),
    ("Bob","likes","the","number"), ("Alice","eats","a","cat"),
    ("John","sees","the","dog"), ("Mary","likes","a","number"),
    ("Bob","loves","the","cat"), ("Alice","sees","a","dog"),
    ("John","eats","the","number"), ("Mary","loves","a","cat"),
    ("Bob","sees","the","dog"), ("Alice","likes","the","number"),
    ("John","likes","a","cat"), ("Mary","eats","the","dog"),
    ("Bob","loves","a","number"), ("Alice","eats","the","cat"),
    ("John","sees","a","number"), ("Mary","likes","the","cat"),
    ("Bob","eats","the","dog"), ("Alice","loves","a","number"),
    ("John","loves","the","number"), ("Mary","sees","the","cat"),
    ("Bob","likes","a","dog"), ("Alice","sees","the","number"),
    ("John","eats","a","dog"), ("Mary","loves","the","dog"),
    ("Bob","sees","a","cat"), ("Alice","likes","the","cat"),
    ("John","sees","the","number"), ("Mary","eats","a","number"),
]:
    d = mk_sentence(mk_proper(subj), mk_trans_vp(v, mk_det_n(det, noun)))
    add(f"{subj} {v} {det} {noun}.", d, f"{v}({subj},{det} {noun})")

print(f"Total: {len(sentences)} sentences")

# ============================================================
# Verify coverage
# ============================================================
def classify(s):
    t = s["sentence"]
    if "not." in t: return "neg"
    if "and" in t and ("sleeps" in t or "runs" in t): return "conj_vp"
    if "and" in t: return "conj_np"
    if "that" in t: return "rel"
    if "adds" in t or "multiplies" in t or "subtracts" in t:
        return "arith"
    if any(a in t for a in ADJ) and "that" not in t: return "adj_n"
    if any(v in t.split() for v in TRANS_V) and any(d in t.split() for d in DETS):
        return "mixed_det"
    if any(v in t.split() for v in TRANS_V): return "trans"
    if any(v in t.split() for v in INTRANS_V):
        if any(d in t.split() for d in DETS): return "det_n"
        return "intrans"
    if any(v in t.split() for v in COP_V + COMP_V + DITRANS_V): return "num/cop"
    return "other"

cats = {}
for s in sentences:
    c = classify(s)
    cats[c] = cats.get(c, 0) + 1

for k, v in sorted(cats.items()):
    print(f"  {k}: {v}")

# Write files
os.makedirs("logos/corpus/l0_seed", exist_ok=True)
os.makedirs("logos/corpus/l0_ungrammatical", exist_ok=True)

with open("logos/corpus/l0_seed/corpus.jsonl", "w") as f:
    for s in sentences:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

# ============================================================
# UNGRAMMATICAL (50)
# ============================================================
ungram = []
uid = 0

def add_ug(s, reason):
    global uid
    uid += 1
    ungram.append({"id": uid, "sentence": s, "reason": reason})

for v in ["loves","sees","likes","eats","sleeps"]:
    add_ug(f"{v} Mary.", "Missing subject")

for s in ["John Mary.", "Bob Alice.", "Mary John.", "Alice Bob.", "John Bob."]:
    add_ug(s, "Missing verb")

for s in ["John loves big.", "Mary sees small.", "Bob likes red.", "Alice eats blue.", "John sees big."]:
    add_ug(s, "Adjective used where NP required")

for s in ["John loves sees Mary.", "Mary sees likes Bob.", "Bob eats sleeps.", "Alice runs loves John.", "John likes sees Alice."]:
    add_ug(s, "Double verb without conjunction")

for s in ["John sleeps Mary.", "Bob runs Alice.", "Mary sleeps John.", "Alice runs Bob.", "John sleeps the cat."]:
    add_ug(s, "Intransitive verb used with object")

for s in ["John adds two.", "Mary multiplies three.", "Bob subtracts five.", "Alice adds one.", "John multiplies two."]:
    add_ug(s, "Ditransitive verb missing required argument")

for s in ["the big.", "a small.", "the red.", "a blue.", "the big."]:
    add_ug(s, "Determiner without noun")

for s in ["cat dog.", "dog cat.", "number cat.", "cat number.", "dog number."]:
    add_ug(s, "Noun phrase without verb (noun stacking)")

for s in ["John is.", "Mary is.", "Bob is.", "Alice is.", "John is."]:
    add_ug(s, "Copula/comparative missing required NP complement")

for s in ["Mary John loves.", "Alice Bob sees.", "cat the sleeps.", "dog a runs.", "John Alice likes."]:
    add_ug(s, "Ungrammatical word order")

with open("logos/corpus/l0_ungrammatical/corpus.jsonl", "w") as f:
    for u in ungram:
        f.write(json.dumps(u, ensure_ascii=False) + "\n")

print(f"Ungrammatical: {len(ungram)}")
print("Done!")
