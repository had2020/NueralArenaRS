use NueralArenaRS::*;

const LAYERS: usize = 3;
const INIT_FN: u8 = 0;

const X0: usize = 4;
const X1: usize = 8;
const X2: usize = 1;
fn main() {
    let model: Model<Layer, LAYERS, INIT_FN> = Model {
        learning_rate: 0.1,
        memory_layout: [Layer { x: X0 }, Layer { x: X1 }, Layer { x: X2 }],
        activation_layout: [ActFn::Sigmoid, ActFn::Sigmoid, ActFn::Sigmoid],
    };
}
