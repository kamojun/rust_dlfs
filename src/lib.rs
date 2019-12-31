#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        let (contexts, target) = util::create_contexts_target(&vec![0, 1, 2, 3, 4, 1, 5, 6], 1);
        let corpus = util::convert_one_hot_2(&contexts, 7);
        let target = util::convert_one_hot_1(&target, 7);
        println!(
            "contexts={:?}, corpus={:?}, target={:?}",
            contexts, corpus, target
        );
    }
}

#[macro_use]
pub mod macros;

pub mod functions;
pub mod io;
pub mod layers;
pub mod math;
pub mod model;
pub mod optimizer;
pub mod params;
pub mod train;
pub mod trainer;
pub mod types;
pub mod util;
// pub mod ui;
