use crate::io::*;
use crate::layers::negativ_sampling_layer::*;
use crate::layers::*;
use crate::model::{Model2, CBOW};
use crate::optimizer::{AdaGrad, Adam, Optimizer, SGD};
use crate::trainer::{Trainer, Trainer2};
use crate::util::*;
use std::collections::HashMap;

pub fn train(test_name: &str) {
    const WINDOW_SIZE: usize = 5;
    const HIDDEN_SIZE: usize = 100;
    const BATCH_SIZE: usize = 100;
    const MAX_EPOCH: usize = 10;
    const SAMPLE_SIZE: usize = 5;
    const DISTRIBUTION_POWER: f32 = 0.75;

    let text = "The executive Power shall be vested in a President of the United States of America. He shall hold his Office during the Term of four Years, and, together with the Vice President, chosen for the same Term, be elected, as follows: Each State shall appoint, in such Manner as the Legislature thereof may direct, a Number of Electors, equal to the whole Number of Senators and Representatives to which the State may be entitled in the Congress: but no Senator or Representative, or Person holding an Office of Trust or Profit under the United States, shall be appointed an Elector. The Electors shall meet in their respective States, and vote by Ballot for two Persons, of whom one at least shall not be an Inhabitant of the same State with themselves. And they shall make a List of all the Persons voted for, and of the Number of Votes for each; which List they shall sign and certify, and transmit sealed to the Seat of the Government of the United States, directed to the President of the Senate. The President of the Senate shall, in the Presence of the Senate and House of Representatives, open all the Certificates, and the Votes shall then be counted. The Person having the greatest Number of Votes shall be the President, if such Number be a Majority of the whole Number of Electors appointed; and if there be more than one who have such Majority, and have an equal Number of Votes, then the House of Representatives shall immediately chuse by Ballot one of them for President; and if no Person have a Majority, then from the five highest on the List the said House shall in like Manner chuse the President. But in chusing the President, the Votes shall be taken by States, the Representation from each State having one Vote; a quorum for this Purpose shall consist of a Member or Members from two thirds of the States, and a Majority of all the States shall be necessary to a Choice. In every Case, after the Choice of the President, the Person having the greatest Number of Votes of the Electors shall be the Vice President. But if there should remain two or more who have equal Votes, the Senate shall chuse from them by Ballot the Vice President. The Congress may determine the Time of chusing the Electors, and the Day on which they shall give their Votes; which Day shall be the same throughout the United States.No Person except a natural born Citizen, or a Citizen of the United States, at the time of the Adoption of this Constitution, shall be eligible to the Office of President; neither shall any person be eligible to that Office who shall not have attained to the Age of thirty five Years, and been fourteen Years a Resident within the United States.In Case of the Removal of the President from Office, or of his Death, Resignation, or Inability to discharge the Powers and Duties of the said Office, the Same shall devolve on the Vice President, and the Congress may by Law provide for the Case of Removal, Death, Resignation or Inability, both of the President and Vice President, declaring what Officer shall then act as President, and such Officer shall act accordingly, until the Disability be removed, or a President shall be elected.The President shall, at stated Times, receive for his Services, a Compensation, which shall neither be encreased nor diminished during the Period for which he shall have been elected, and he shall not receive within that Period any other Emolument from the United States, or any of them.";
    // corpus ... 文章を、その単語のID列で表したもの
    // word_to_id ... 単語に対し、そのID(一応出現順に割り振る)を対応させた辞書
    // id_to_word ... 逆にしたもの
    let corpus = read_csv::<usize>("./data/ptb/corpus.csv").expect("error in reading corpus");
    let id_to_word: HashMap<usize, String> = read_csv("./data/ptb/id.csv")
        .expect("error in reading id")
        .into_iter()
        .collect();
    // let (corpus, word_to_id, id_to_word) = preprocess(text);
    // let word_to_id: HashMap<String, usize> = id_to_word.iter().map(|(i,w)| (w.clone(), *i)).collect();
    let vocab_size = id_to_word.len(); // 単語数
    putsl!(vocab_size);
    let (contexts, target) = create_contexts_target_arr(&corpus, WINDOW_SIZE);
    putsl!(contexts, target);
    // コンテキストは、ここから対応する行を取り出して和を取る
    let w_in = randarr2d(vocab_size, HIDDEN_SIZE);
    // ターゲットをここから取り出し、内積を取る
    let w_out = randarr2d(vocab_size, HIDDEN_SIZE);
    let distribution = get_distribution(&corpus, Some(DISTRIBUTION_POWER));
    let model = CBOW::new(&[w_in, w_out], SAMPLE_SIZE, distribution);
    // let mut optimizer = SGD { lr: 1.0 };
    let mut optimizer = AdaGrad::new(1.0);
    // let mut optimizer = Adam::new(0.001, 0.9, 0.999);
    let mut trainer = Trainer2::new(model, optimizer);
    trainer.fit(contexts, target, MAX_EPOCH, BATCH_SIZE, Some(100));
    trainer.show_loss();
    trainer
        .model
        .save_as_csv("trained/CBOW_model")
        .expect("error!");
    // for (id, word_vec) in trainer.model.word_vecs().outer_iter().enumerate() {
    //     puts!(
    //         format!("{:<8}", id_to_word[&id]),
    //         word_vec.iter().cloned().collect::<Vec<f32>>()
    //     );
    // }
}

#[test]
fn ch04_test() {
    train("testname");
}
