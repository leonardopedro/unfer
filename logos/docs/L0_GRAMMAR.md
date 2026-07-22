# L0 Grammar

## Sentence Forms

```
Sentence    := NP VP
VP          := Vtrans NP | Vintrans | Vditrans NP NP | Copula NP | VP conj VP
NP          := Det N | Det N RelClause | ProperNoun | Numeral | NP conj NP | Det Adj N
RelClause   := RelPron VP
Det         := "the" | "a"
N           := "number" | "cat" | "dog"
Adj         := "big" | "small" | "red" | "blue" | "very" Adj
Vtrans      := "loves" | "sees" | "likes" | "eats"
Vintrans    := "sleeps" | "runs"
Vditrans    := "adds" | "multiplies" | "subtracts" | "give"
Copula      := "is" | "equals" | "greater" | "less"
Numeral     := "zero" | "one" | "two" | "three" | "four" | "five"
              | "six" | "seven" | "eight" | "nine" | "ten"
RelPron     := "that" | "which"
conj        := "and"
```

## CCG Categories

| Word | Category | Semantics |
|------|----------|-----------|
| Proper nouns (John, Mary, Bob, Alice) | `NP` | `Var("name")` |
| Determiners (the, a) | `NP/N` | `λn. n` |
| Nouns (number, cat, dog) | `N` | `Noun` |
| Transitive verbs (loves, sees, likes, eats) | `(S\NP)/NP` | `λy.λx. Verb(x,y)` |
| Intransitive verbs (sleeps, runs) | `S\NP` | `λx. Verb(x)` |
| Ditransitive verbs (adds, multiplies, subtracts, give) | `((S\NP)/NP)/NP` | `λz.λy.λx. Verb(x,y,z)` |
| Copulas (is, equals, greater, less) | `(S\NP)/NP` | `λy.λx. Rel(x,y)` |
| Numerals (zero..ten) | `NP` | `Lit(n)` |
| Adjectives (big, small, red, blue) | `N/N` | `λn. Adj(n)` |
| Intensifier (very) | `AP/AP` | `λadj.λx. Very(adj,x)` |
| Relative pronouns (that, which) | `(NP\NP)/S` | `λs.λn. Restrict(n,s)` |
| Coordination (and) | `conj` | `And` |
| Negation (not) | `(S\NP)\(S\NP)` | `λvp.λx. Not(vp(x))` |

## Combinator Rules

| Rule | Type | Signature |
|------|------|-----------|
| Forward Application (`>`) | `X/Y  Y  →  X` | `loves NP → S\NP` |
| Backward Application (`<`) | `Y  X\Y  →  X` | `NP S\NP → S` |
| Forward Composition (`>B`) | `X/Y  Y/Z  →  X/Z` | `(S\NP)/NP  NP/N → (S\NP)/N` |
| Backward Composition (`<B`) | `Y\Z  X\Y  →  X\Z` | — |

## Example Derivations

### "John loves Mary"

```
John    NP
loves   (S\NP)/NP
Mary    NP

loves Mary    (S\NP)/NP  NP  →  S\NP    [>]
John (loves Mary)    NP  S\NP  →  S      [<]
```

### "the cat sleeps"

```
the     NP/N
cat     N

the cat    NP/N  N  →  NP    [>]
sleeps     S\NP

(the cat) sleeps    NP  S\NP  →  S    [<]
```

### "John adds two and three"

```
John    NP
adds    ((S\NP)/NP)/NP
two     NP
three   NP

adds two    ((S\NP)/NP)/NP  NP  →  (S\NP)/NP    [>]
(adds two) three    (S\NP)/NP  NP  →  S\NP    [>]
John (adds two three)    NP  S\NP  →  S    [<]
```

### "the big cat that sleeps sees Mary"

```
big      N/N
cat      N

big cat    N/N  N  →  NP    [>]

the     NP/N
the (big cat)    NP/N  N  →  NP    [>]

sleeps    S\NP
that      (NP\NP)/S
that sleeps    (NP\NP)/S  S\NP  →  NP\NP    [>]

(the big cat) (that sleeps)    NP  NP\NP  →  NP    [<]

sees     (S\NP)/NP
sees Mary    (S\NP)/NP  NP  →  S\NP    [>]

(the big cat that sleeps) (sees Mary)    NP  S\NP  →  S    [<]
```
