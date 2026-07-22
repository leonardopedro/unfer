use super::types::*;
use crate::lexicon::Lexicon;

#[derive(Debug, Clone)]
pub struct ChartEntry {
    pub category: CCGCategory,
    pub derivation: DerivationTree,
    pub span: (usize, usize),
}

#[derive(Debug, Default, Clone)]
pub struct ChartCell {
    pub entries: Vec<ChartEntry>,
}

pub fn parse_sentence(tokens: &[String], lexicon: &Lexicon) -> Vec<DerivationTree> {
    let n = tokens.len();
    let mut chart: Vec<Vec<ChartCell>> = vec![vec![ChartCell::default(); n + 1]; n + 1];

    for (i, tok) in tokens.iter().enumerate() {
        for entry in lexicon.lookup(tok) {
            let category = CCGCategory::parse(&entry.category).unwrap_or_else(|_| {
                eprintln!("warning: skipping unrecognized category '{}' for '{}'", entry.category, tok);
                CCGCategory::S
            });
            chart[i][i + 1].entries.push(ChartEntry {
                category: category.clone(),
                derivation: DerivationTree::Leaf {
                    word: tok.clone(),
                    category,
                },
                span: (i, i + 1),
            });
        }
    }

    for span in 2..=n {
        for i in 0..=(n - span) {
            let j = i + span;
            let mut results: Vec<ChartEntry> = Vec::new();

            for k in (i + 1)..j {
                for left in &chart[i][k].entries {
                    for right in &chart[k][j].entries {
                        apply_combinators(left, right, &mut results);
                    }
                }
            }

            chart[i][j].entries.extend(results);
        }
    }

    chart[0][n]
        .entries
        .iter()
        .filter(|e| e.category.is_sentence())
        .map(|e| e.derivation.clone())
        .collect()
}

fn apply_combinators(left: &ChartEntry, right: &ChartEntry, results: &mut Vec<ChartEntry>) {
    // Forward Application: X/Y + Y → X
    if let CCGCategory::Slash { forward: true, result, argument } = &left.category {
        if **argument == right.category {
            results.push(ChartEntry {
                category: (**result).clone(),
                derivation: DerivationTree::Application {
                    direction: Direction::Forward,
                    result_category: (**result).clone(),
                    left: Box::new(left.derivation.clone()),
                    right: Box::new(right.derivation.clone()),
                },
                span: (left.span.0, right.span.1),
            });
        }
    }

    // Backward Application: Y + X\Y → X
    if let CCGCategory::Slash { forward: false, result, argument } = &right.category {
        if **argument == left.category {
            results.push(ChartEntry {
                category: (**result).clone(),
                derivation: DerivationTree::Application {
                    direction: Direction::Backward,
                    result_category: (**result).clone(),
                    left: Box::new(left.derivation.clone()),
                    right: Box::new(right.derivation.clone()),
                },
                span: (left.span.0, right.span.1),
            });
        }
    }

    // Forward Composition: X/Y + Y/Z → X/Z
    if let CCGCategory::Slash { forward: true, result: x, argument: y1 } = &left.category {
        if let CCGCategory::Slash { forward: true, result: y2, argument: z } = &right.category {
            if **y1 == **y2 {
                let new_result = CCGCategory::forward((**x).clone(), (**z).clone());
                results.push(ChartEntry {
                    category: new_result.clone(),
                    derivation: DerivationTree::Composition {
                        direction: Direction::Forward,
                        result_category: new_result,
                        left: Box::new(left.derivation.clone()),
                        right: Box::new(right.derivation.clone()),
                    },
                    span: (left.span.0, right.span.1),
                });
            }
        }
    }

    // Backward Composition: Y\Z + X\Y → X\Z
    if let CCGCategory::Slash { forward: false, result: y1, argument: z } = &left.category {
        if let CCGCategory::Slash { forward: false, result: x, argument: y2 } = &right.category {
            if **y1 == **y2 {
                let new_result = CCGCategory::backward((**x).clone(), (**z).clone());
                results.push(ChartEntry {
                    category: new_result.clone(),
                    derivation: DerivationTree::Composition {
                        direction: Direction::Backward,
                        result_category: new_result,
                        left: Box::new(left.derivation.clone()),
                        right: Box::new(right.derivation.clone()),
                    },
                    span: (left.span.0, right.span.1),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexicon::Lexicon;

    fn test_lexicon() -> Lexicon {
        let tsv = "John\tNP\tVar(\"john\")\nMary\tNP\tVar(\"mary\")\nloves\t(S\\NP)/NP\tLam(\"y\", Lam(\"x\", Con(\"Love\", [Var(\"x\"), Var(\"y\")])))\nthe\tNP/N\tLam(\"n\", Var(\"n\"))\ncat\tN\tCon(\"Cat\", [])\nsleeps\tS\\NP\tLam(\"x\", Con(\"Sleep\", [Var(\"x\")]))\n";
        Lexicon::parse(tsv).unwrap()
    }

    #[test]
    fn test_parse_john_loves_mary() {
        let lex = test_lexicon();
        let tokens: Vec<String> = "John loves Mary".split_whitespace().map(String::from).collect();
        let trees = parse_sentence(&tokens, &lex);
        assert!(!trees.is_empty(), "should parse 'John loves Mary'");
        assert!(trees.iter().all(|t| t.is_sentence()));
    }

    #[test]
    fn test_parse_the_cat_sleeps() {
        let lex = test_lexicon();
        let tokens: Vec<String> = "the cat sleeps".split_whitespace().map(String::from).collect();
        let trees = parse_sentence(&tokens, &lex);
        assert!(!trees.is_empty(), "should parse 'the cat sleeps'");
    }
}
