#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {}
}

#[macro_use]
pub mod macros;

pub mod array_util;
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
