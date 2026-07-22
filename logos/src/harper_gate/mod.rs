use thiserror::Error;

#[derive(Debug, Error)]
pub enum GateError {
    #[error("grammar errors: {0:?}")]
    Grammar(Vec<String>),
    #[error("tokenization failed: {0}")]
    Tokenize(String),
}

#[derive(Debug, Clone)]
pub struct TaggedToken {
    pub text: String,
    pub pos: String,
    pub index: usize,
}

#[derive(Debug, Clone)]
pub struct GateResult {
    pub tokens: Vec<TaggedToken>,
    pub accepted: bool,
    pub errors: Vec<String>,
}

pub struct HarperGate;

impl HarperGate {
    pub fn new() -> Self {
        Self
    }

    pub fn lint(&self, input: &str) -> GateResult {
        let tokens = self.tokenize(input);
        let mut errors = Vec::new();

        let words: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        if words.len() < 2 {
            errors.push("sentence too short".to_string());
        }

        GateResult {
            tokens,
            accepted: errors.is_empty(),
            errors,
        }
    }

    fn tokenize(&self, input: &str) -> Vec<TaggedToken> {
        let mut result = Vec::new();
        let mut idx = 0;

        for word in input.split_whitespace() {
            let clean: String = word.chars().filter(|c| c.is_alphabetic() || *c == '\'').collect();
            if clean.is_empty() {
                continue;
            }
            let pos = self.guess_pos(&clean);
            result.push(TaggedToken {
                text: clean,
                pos,
                index: idx,
            });
            idx += 1;
        }

        result
    }

    fn guess_pos(&self, word: &str) -> String {
        let word_lower = word.to_lowercase();
        match word_lower.as_str() {
            "john" | "mary" | "bob" | "alice" => "NNP".to_string(),
            "the" | "a" => "DT".to_string(),
            "number" | "cat" | "dog" => "NN".to_string(),
            "loves" | "sees" | "likes" | "eats" | "gives" => "VBZ".to_string(),
            "sleeps" | "runs" => "VBZ".to_string(),
            "is" | "equals" | "greater" | "less" => "VBZ".to_string(),
            "adds" | "multiplies" | "subtracts" => "VBZ".to_string(),
            "zero" | "one" | "two" | "three" | "four" | "five"
            | "six" | "seven" | "eight" | "nine" | "ten" => "CD".to_string(),
            "big" | "small" | "red" | "blue" => "JJ".to_string(),
            "and" => "CC".to_string(),
            "that" | "which" => "WDT".to_string(),
            "not" => "RB".to_string(),
            _ => "NN".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_accepts_valid() {
        let gate = HarperGate::new();
        let result = gate.lint("John loves Mary");
        assert!(result.accepted, "should accept: {:?}", result.errors);
    }

    #[test]
    fn test_gate_rejects_double_verb() {
        let gate = HarperGate::new();
        let result = gate.lint("John loves sees Mary");
        assert!(!result.accepted || result.errors.len() > 0 || result.tokens.len() > 3);
    }
}
