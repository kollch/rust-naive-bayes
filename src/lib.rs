use std::env;
use std::error::Error;
use std::fmt;
use std::fs;

#[derive(Debug)]
pub struct Config {
    pub training: String,
    pub testing: String,
}

impl Config {
    pub fn new(mut args: env::Args) -> Result<Self, &'static str> {
        args.next();

        let training = match args.next() {
            Some(arg) => arg,
            None => return Err("Didn't get training data"),
        };

        let testing = match args.next() {
            Some(arg) => arg,
            None => return Err("Didn't get testing data"),
        };

        Ok(Config { training, testing })
    }
}

pub struct Classify {
    training: Data,
    testing: Data,
    prob_pos: f64,
    prob_neg: f64,
    prob_w_pos: Vec<f64>,
    prob_w_neg: Vec<f64>,
    accuracy: f64,
}

impl Classify {
    pub fn new(preprocessed: Preprocess) -> Self {
        Classify {
            training: preprocessed.training,
            testing: preprocessed.testing,
            prob_pos: 0.0,
            prob_neg: 0.0,
            prob_w_pos: Vec::new(),
            prob_w_neg: Vec::new(),
            accuracy: 0.0,
        }
    }

    fn calculate_accuracy(&self) -> f64 {
        let mut accuracy = 0.0;

        for (i, feature) in self.testing.features.iter().enumerate() {
            if feature.classlabel == self.testing.classifications[i] {
                accuracy += 1.0;
            }
        }
        accuracy /= self.testing.classifications.len() as f64;

        accuracy
    }

    fn classify_data(&mut self) {
        let prob_pos = self.prob_pos / self.prob_neg;
        let ln_prob_pos = prob_pos.ln();

        for feature in &mut self.testing.features {
            let mut ln_sum = 0.0;
            for (i, word) in feature.vector.iter().enumerate() {
                if word == &1 {
                    let sum = self.prob_w_pos[i] / self.prob_w_neg[i];
                    ln_sum += sum.ln();
                }
            }
            if ln_prob_pos + ln_sum > 0.0 {
                feature.classlabel = 1;
            } else {
                feature.classlabel = 0;
            }
        }
    }

    fn learn_params(&mut self) {
        self.prob_pos = self.pos_stat();
        self.prob_neg = 1.0 - &self.prob_pos;
        self.prob_w_pos = self.set_stats(1);
        self.prob_w_neg = self.set_stats(0);
    }

    fn pos_stat(&self) -> f64 {
        (self
            .training
            .features
            .iter()
            .filter(|x| x.classlabel == 1)
            .count() as f64)
            / (self.training.features.len() as f64)
    }

    pub fn run(mut self) -> Self {
        self.learn_params();
        self.classify_data();
        self.accuracy = self.calculate_accuracy();

        self
    }

    fn set_stats(&self, classif: i32) -> Vec<f64> {
        let mut result = vec![0.0; self.training.features[0].length];
        let mut num_word_type = 0;
        for feature in self
            .training
            .features
            .iter()
            .filter(|x| x.classlabel == classif)
        {
            num_word_type += 1;
            for (i, _) in feature.vector.iter().enumerate().filter(|(_, &x)| x == 1) {
                result[i] += 1.0;
            }
        }
        let result_clone = result.clone();
        for (i, val) in result_clone.iter().enumerate() {
            // Uniform Dirichlet prior
            // Could realistically compare against 0.0 exactly since we've hardcoded it above, but
            // as a general rule strict comparison of floating types is a bad idea.
            if val < &0.5 {
                result[i] = 1.0 / (result.len() as f64);
            } else {
                result[i] /= f64::from(num_word_type);
            }
        }
        result
    }
}

impl fmt::Display for Classify {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let border = "-------------------------------------";

        write!(
            f,
            "{}\nTraining File: {}\nTesting File: {}\nAccuracy: {:.16}\n{0}",
            border, self.training.filename, self.testing.filename, self.accuracy
        )
    }
}

type WordSet = Vec<Box<Vec<String>>>;

pub struct Data {
    filename: String,
    data: WordSet,
    classifications: Vec<i32>,
    features: Vec<Feature>,
}

impl Data {
    pub fn new(filename: String) -> Result<Self, Box<dyn Error>> {
        let filestream = fs::read_to_string(&filename)?;

        let (data, classifications) = Self::clean_data(filestream)?;

        Ok(Data {
            filename,
            data,
            classifications,
            features: Vec::new(),
        })
    }

    fn clean_data(raw: String) -> Result<(WordSet, Vec<i32>), Box<dyn Error>> {
        let raw = raw.to_lowercase();

        let mut data: WordSet = Vec::new();
        let mut classifications: Vec<i32> = Vec::new();

        for line in raw.lines() {
            classifications.push(
                line.rsplitn(2, '\t')
                    .next()
                    .ok_or("Error: no tab delimiting sentiment classification")?
                    .trim()
                    .parse::<i32>()?,
            );
            data.push(Box::new(
                line.chars()
                    .filter(|&c| match c {
                        'a'...'z' => true,
                        ' ' => true,
                        _ => false,
                    })
                    .collect::<String>()
                    .split_whitespace()
                    .map(|item| item.to_string())
                    .collect(),
            ));
        }
        Ok((data, classifications))
    }

    fn make_features(&mut self, vocab: &[String]) {
        for line in &self.data {
            let mut new_feature = Feature::new(vocab.len());
            for word in line.iter() {
                new_feature.add(vocab, word);
            }
            self.features.push(new_feature);
        }
    }
}

pub struct Feature {
    length: usize,
    vector: Vec<i32>,
    classlabel: i32,
}

impl Feature {
    pub fn new(length: usize) -> Self {
        Feature {
            length,
            vector: vec![0; length],
            classlabel: -1,
        }
    }

    fn add(&mut self, vocab: &[String], word: &str) {
        if let Some(pos) = vocab.iter().position(|x| x == word) {
            self.vector[pos] = 1;
        }
    }
}

impl fmt::Display for Feature {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{},{}",
            self.vector
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<String>>()
                .join(","),
            self.classlabel.to_string()
        )
    }
}

pub struct Preprocess {
    training: Data,
    testing: Data,
    vocab: Vec<String>,
}

impl Preprocess {
    pub fn new(training: String, testing: String) -> Result<Self, Box<dyn Error>> {
        let training = Data::new(training)?;
        let testing = Data::new(testing)?;

        Ok(Preprocess {
            training,
            testing,
            vocab: Vec::new(),
        })
    }

    fn make_vocab(data: &WordSet) -> Vec<String> {
        let mut vocab = data
            .iter()
            .flat_map(|line| line.iter())
            .cloned()
            .collect::<Vec<String>>();

        vocab.sort_unstable();
        vocab.dedup();
        vocab
    }

    fn output_features(&self, data: &Data, filename: &str) -> Result<(), Box<dyn Error>> {
        let mut output = self.vocab.join(",");
        output.push_str(",classlabel\n");
        for feature in data.features.iter() {
            output.push_str(&feature.to_string());
            output.push('\n');
        }
        fs::write(filename, output)?;
        Ok(())
    }

    pub fn run(mut self) -> Result<Self, Box<dyn Error>> {
        self.vocab = Self::make_vocab(&self.training.data);
        self.training.make_features(&self.vocab);
        for (i, feature) in self.training.features.iter_mut().enumerate() {
            feature.classlabel = self.training.classifications[i];
        }
        self.testing.make_features(&self.vocab);
        self.output_features(&self.training, "preprocessed_train.txt")?;
        self.output_features(&self.testing, "preprocessed_test.txt")?;

        Ok(self)
    }
}

#[cfg(test)]
mod tests {}
