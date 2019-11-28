macro_rules! puts {
    ($( $x:expr), *) => {
        $(
            print!("{:?} ", $x);
        )*
    };
}

macro_rules! putsd {
    ($( $x:expr), *) => {
        $(
            print!("{}= {:?}; ", stringify!($x), $x);
        )*
    };
}

macro_rules! putsl {
    ($( $x:expr), *) => {
        $(
            print!("{}= {:?}\n", stringify!($x), $x);
        )*
    };
}
