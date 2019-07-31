use std::env;
use std::process;

use bayes::{Classify, Config, Preprocess};

fn main() {
    let config = Config::new(env::args()).unwrap_or_else(|err| {
        eprintln!("Problem parsing arguments: {}", err);
        process::exit(1);
    });

    let preprocess = Preprocess::new(config.training, config.testing).unwrap_or_else(|err| {
        eprintln!("Error preprocessing data: {}", err);
        process::exit(1);
    });

    let preprocessed = preprocess.run().unwrap_or_else(|err| {
        eprintln!("Error processing data: {}", err);
        process::exit(1);
    });

    let classify = Classify::new(preprocessed);

    let classified = classify.run();

    println!("{}", classified);
}
