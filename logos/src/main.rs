fn main() {
    let args: Vec<String> = std::env::args().collect();
    logos::cli::run_cli(args);
}
