// Run with cargo run --example xor_test_run
use NueralArenaRS::*;

const LAYERS: usize = 3;

const INPUTS: usize = 4;
const HIDDEN: usize = 8;
const OUTPUT: usize = 1;

fn main() {
    let model: Model<Layer, LAYERS> = Model {
        learning_rate: 0.1,
        memory_layout: [
            Layer { x: INPUTS },
            Layer { x: HIDDEN },
            Layer { x: OUTPUT },
        ],
        activation_layout: [ActFn::Sigmoid, ActFn::Sigmoid, ActFn::Sigmoid],
    };
    let mut arena_model = model.allocate();
    arena_model.init(InitFunc::Zero, &model);
}
