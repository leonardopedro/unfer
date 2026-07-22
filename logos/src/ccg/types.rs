use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CCGCategory {
    S,
    NP,
    N,
    Adj,
    Conj,
    Slash {
        forward: bool,
        result: Box<CCGCategory>,
        argument: Box<CCGCategory>,
    },
}

impl CCGCategory {
    pub fn forward(result: CCGCategory, argument: CCGCategory) -> Self {
        CCGCategory::Slash {
            forward: true,
            result: Box::new(result),
            argument: Box::new(argument),
        }
    }

    pub fn backward(result: CCGCategory, argument: CCGCategory) -> Self {
        CCGCategory::Slash {
            forward: false,
            result: Box::new(result),
            argument: Box::new(argument),
        }
    }

    pub fn is_sentence(&self) -> bool {
        matches!(self, CCGCategory::S)
    }
}

impl fmt::Display for CCGCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CCGCategory::S => write!(f, "S"),
            CCGCategory::NP => write!(f, "NP"),
            CCGCategory::N => write!(f, "N"),
            CCGCategory::Adj => write!(f, "AP"),
            CCGCategory::Conj => write!(f, "conj"),
            CCGCategory::Slash { forward, result, argument } => {
                let slash = if *forward { '/' } else { '\\' };
                write!(f, "({}{}{})", result, slash, argument)
            }
        }
    }
}

impl CCGCategory {
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();

        // Try simple names first
        match s {
            "S" => return Ok(CCGCategory::S),
            "NP" => return Ok(CCGCategory::NP),
            "N" => return Ok(CCGCategory::N),
            "AP" | "Adj" => return Ok(CCGCategory::Adj),
            "conj" => return Ok(CCGCategory::Conj),
            _ => {}
        }

        // Strip matched outer parens
        let s = if s.starts_with('(') && s.ends_with(')') && is_matched_parens(s) {
            &s[1..s.len() - 1]
        } else {
            s
        };

        // Find top-level slash
        if let Some(idx) = find_slash(s) {
            let left = &s[..idx];
            let forward = s.as_bytes()[idx] == b'/';
            let right = &s[idx + 1..];
            let result = CCGCategory::parse(left)?;
            let argument = CCGCategory::parse(right)?;
            return Ok(CCGCategory::Slash {
                forward,
                result: Box::new(result),
                argument: Box::new(argument),
            });
        }

        Err(format!("unrecognized CCG category: {}", s))
    }
}

fn find_slash(s: &str) -> Option<usize> {
    let mut depth = 0;
    for (i, ch) in s.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth -= 1,
            _ if depth == 0 && (ch == '/' || ch == '\\') => return Some(i),
            _ => {}
        }
    }
    None
}

fn is_matched_parens(s: &str) -> bool {
    let bytes = s.as_bytes();
    if bytes[0] != b'(' || bytes[bytes.len() - 1] != b')' {
        return false;
    }
    let mut depth = 0;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 && i < bytes.len() - 1 {
                    return false;
                }
            }
            _ => {}
        }
    }
    depth == 0
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DerivationTree {
    Leaf {
        word: String,
        category: CCGCategory,
    },
    Application {
        direction: Direction,
        result_category: CCGCategory,
        left: Box<DerivationTree>,
        right: Box<DerivationTree>,
    },
    Composition {
        direction: Direction,
        result_category: CCGCategory,
        left: Box<DerivationTree>,
        right: Box<DerivationTree>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Direction {
    Forward,
    Backward,
}

impl DerivationTree {
    pub fn result_category(&self) -> &CCGCategory {
        match self {
            DerivationTree::Leaf { category, .. } => category,
            DerivationTree::Application { result_category, .. } => result_category,
            DerivationTree::Composition { result_category, .. } => result_category,
        }
    }

    pub fn is_sentence(&self) -> bool {
        self.result_category().is_sentence()
    }

    pub fn leaves(&self) -> Vec<&str> {
        match self {
            DerivationTree::Leaf { word, .. } => vec![word.as_str()],
            DerivationTree::Application { left, right, .. }
            | DerivationTree::Composition { left, right, .. } => {
                let mut words = left.leaves();
                words.extend(right.leaves());
                words
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_category_parse() {
        assert_eq!(CCGCategory::parse("S").unwrap(), CCGCategory::S);
        assert_eq!(CCGCategory::parse("NP").unwrap(), CCGCategory::NP);
        assert_eq!(
            CCGCategory::parse("(S\\NP)").unwrap(),
            CCGCategory::backward(CCGCategory::S, CCGCategory::NP)
        );
        assert_eq!(
            CCGCategory::parse("((S\\NP)/NP)").unwrap(),
            CCGCategory::forward(
                CCGCategory::backward(CCGCategory::S, CCGCategory::NP),
                CCGCategory::NP
            )
        );
    }
}
