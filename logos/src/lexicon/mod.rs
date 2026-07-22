use std::collections::HashMap;
use std::fs;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LexiconError {
    #[error("failed to read lexicon file: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid TSV format at line {line}: {reason}")]
    Format { line: usize, reason: String },
    #[error("missing required column: {0}")]
    MissingColumn(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum SemExpr {
    Var(String),
    Lit(Literal),
    Con(String, Vec<SemExpr>),
    Lam(String, Box<SemExpr>),
    App(Box<SemExpr>, Box<SemExpr>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Int64(i64),
    Bool(bool),
}

impl Literal {
    pub fn to_string_value(&self) -> String {
        match self {
            Literal::Int64(n) => format!("{}", n),
            Literal::Bool(b) => format!("{}", b),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LexEntry {
    pub word: String,
    pub category: String,
    pub template: SemExpr,
}

#[derive(Debug, Clone, Default)]
pub struct Lexicon {
    entries: Vec<LexEntry>,
    by_word: HashMap<String, Vec<usize>>,
}

impl Lexicon {
    pub fn load(path: &Path) -> Result<Self, LexiconError> {
        let content = fs::read_to_string(path)?;
        Self::parse(&content)
    }

    pub fn parse(content: &str) -> Result<Self, LexiconError> {
        let mut entries = Vec::new();
        let mut by_word: HashMap<String, Vec<usize>> = HashMap::new();

        for (i, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() < 3 {
                return Err(LexiconError::Format {
                    line: i + 1,
                    reason: format!("expected 3 tab-separated columns, got {}", parts.len()),
                });
            }
            let word = parts[0].to_string();
            let category = parts[1].to_string();
            let template = parse_sem_template(parts[2])
                .map_err(|e| LexiconError::Format { line: i + 1, reason: e })?;

            let idx = entries.len();
            by_word.entry(word.clone()).or_default().push(idx);
            entries.push(LexEntry { word, category, template });
        }

        Ok(Lexicon { entries, by_word })
    }

    pub fn lookup(&self, word: &str) -> Vec<&LexEntry> {
        self.by_word
            .get(word)
            .map(|idxs| idxs.iter().map(|&i| &self.entries[i]).collect())
            .unwrap_or_default()
    }

    pub fn semantic_template(&self, word: &str) -> Option<&SemExpr> {
        self.lookup(word).first().map(|e| &e.template)
    }

    pub fn entries(&self) -> &[LexEntry] {
        &self.entries
    }

    pub fn word_count(&self) -> usize {
        self.entries.len()
    }
}

fn parse_sem_template(s: &str) -> Result<SemExpr, String> {
    let s = s.trim();
    if let Some(var) = s.strip_prefix("Var(\"").and_then(|r| r.strip_suffix("\")")) {
        return Ok(SemExpr::Var(var.to_string()));
    }
    if let Some(n) = s.strip_prefix("Lit(Int64(").and_then(|r| r.strip_suffix("))")) {
        let val: i64 = n.parse().map_err(|e| format!("invalid int: {}", e))?;
        return Ok(SemExpr::Lit(Literal::Int64(val)));
    }
    if let Some(b) = s.strip_prefix("Lit(Bool(").and_then(|r| r.strip_suffix("))")) {
        let val: bool = b.parse().map_err(|e| format!("invalid bool: {}", e))?;
        return Ok(SemExpr::Lit(Literal::Bool(val)));
    }
    if let Some(rest) = s.strip_prefix("Con(\"").and_then(|r| r.strip_suffix(')')) {
        let (tag, args_str) = rest.split_once("\", [").ok_or("Con missing args")?;
        let args_str = args_str.trim_end_matches(']');
        if args_str.is_empty() {
            return Ok(SemExpr::Con(tag.to_string(), vec![]));
        }
        let args = parse_comma_separated(args_str)?;
        return Ok(SemExpr::Con(tag.to_string(), args));
    }
    if let Some(rest) = s.strip_prefix("Lam(\"").and_then(|r| r.strip_suffix(')')) {
        let (var, body_str) = rest.split_once("\", ").ok_or("Lam missing body")?;
        let body = parse_sem_template(body_str)?;
        return Ok(SemExpr::Lam(var.to_string(), Box::new(body)));
    }
    Err(format!("unrecognized template: {}", s))
}

fn parse_comma_separated(s: &str) -> Result<Vec<SemExpr>, String> {
    let mut result = Vec::new();
    let mut depth = 0;
    let mut current = String::new();
    for ch in s.chars() {
        match ch {
            '(' | '[' => {
                depth += 1;
                current.push(ch);
            }
            ')' | ']' => {
                depth -= 1;
                current.push(ch);
            }
            ',' if depth == 0 => {
                result.push(parse_sem_template(&current)?);
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    if !current.trim().is_empty() {
        result.push(parse_sem_template(&current)?);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parse_lexicon() {
        let tsv = "John\tNP\tVar(\"john\")\nloves\t(S\\NP)/NP\tLam(\"y\", Lam(\"x\", Con(\"Love\", [Var(\"x\"), Var(\"y\")])))\n";
        let lex = Lexicon::parse(tsv).unwrap();
        assert_eq!(lex.word_count(), 2);
        assert_eq!(lex.lookup("John").len(), 1);
        assert_eq!(lex.lookup("loves").len(), 1);
    }

    #[test]
    fn test_load_from_file() {
        let tsv = "zero\tNP\tLit(Int64(0))\none\tNP\tLit(Int64(1))\n";
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(tsv.as_bytes()).unwrap();
        let lex = Lexicon::load(f.path()).unwrap();
        assert_eq!(lex.word_count(), 2);
    }
}
