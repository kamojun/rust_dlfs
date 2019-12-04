macro_rules! puts {
    ($( $x:expr), *) => {
        $(
            print!("{:?} ", $x);
        )*
        println!();
    };
}

macro_rules! putsd {
    ($( $x:expr), *) => {
        $(
            print!("{}= {:?}; ", stringify!($x), $x);
        )*        println!();

    };
}

macro_rules! putsl {
    ($( $x:expr), *) => {
        $(
            println!("{}= {:?}", stringify!($x), $x);
        )*
    };
}
