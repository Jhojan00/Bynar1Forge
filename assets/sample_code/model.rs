use ndarray as nd;
use ndarray_rand::RandomExt;
use rand_distr::Normal;
use std::collections::HashMap;

#[derive(Clone, Copy)]
enum ActivationFunc {
    Tanh,
    Sigmoid,
}

pub struct BinaryNeuronalNetwork {
    X: nd::Array2<f64>,
    Y: nd::Array2<f64>,
    learning_rate: f64,
    cycles: usize,
    neurons_like: Vec<usize>,
    m: usize,
    l: usize,
    cache: HashMap<String, HashMap<u16, nd::Array2<f64>>>,
}

impl BinaryNeuronalNetwork {
    pub fn new(
        X: nd::Array2<f64>,
        Y: nd::Array2<f64>,
        learning_rate: f64,
        cycles: usize,
        neurons_like: Vec<usize>,
    ) -> Self {
        let m = X.shape()[1];
        let mut neurons = vec![X.shape()[0]];
        neurons.extend(neurons_like.clone());
        let l = neurons.len() - 1;

        let mut cache: HashMap<String, HashMap<u16, nd::Array2<f64>>> = HashMap::new();
        let mut W = HashMap::new();
        let mut B = HashMap::new();
        let mut A = HashMap::new();

        A.insert(0, X.clone());

        for layer in 1..=l {
            let scale = (1.0 / neurons[layer - 1] as f64).sqrt();
            W.insert(
                layer as u16,
                nd::Array2::random(
                    (neurons[layer], neurons[layer - 1]),
                    Normal::new(0.0, scale).unwrap(),
                ),
            );
            B.insert(layer as u16, nd::Array2::zeros((neurons[layer], 1)));
        }

        cache.insert("W".to_string(), W);
        cache.insert("B".to_string(), B);
        cache.insert("A".to_string(), A);
        cache.insert("Z".to_string(), HashMap::new());

        Self {
            X,
            Y,
            learning_rate,
            cycles,
            neurons_like: neurons,
            m,
            l,
            cache,
        }
    }

    fn activation(&self, z: &nd::Array2<f64>, last_layer: bool) -> nd::Array2<f64> {
        if last_layer {
            z.mapv(|x| 1.0 / (1.0 + (-x).exp()))
        } else {
            z.mapv(f64::tanh)
        }
    }

    fn activation_derivative(&self, a: &nd::Array2<f64>) -> nd::Array2<f64> {
        a.mapv(|x| 1.0 - x.powi(2))
    }

    fn binary_cost(&self, A: &nd::Array2<f64>) -> f64 {
        let eps = 1e-8;
        let cost = -(&self.Y * &(A + eps).mapv(f64::ln)
            + &(1.0 - &self.Y) * &(1.0 - A + eps).mapv(f64::ln))
            .sum()
            / self.m as f64;
        cost
    }

    pub fn train(&mut self) {
        for c in 0..self.cycles {
            for layer in 1..=self.l {
                let A_prev = self.cache["A"].get(&(layer as u16 - 1)).unwrap();
                let W = self.cache["W"].get(&(layer as u16)).unwrap();
                let B = self.cache["B"].get(&(layer as u16)).unwrap();
                let Z = W.dot(A_prev) + B;

                let A = self.activation(&Z, layer == self.l);

                self.cache.get_mut("Z").unwrap().insert(layer as u16, Z);
                self.cache.get_mut("A").unwrap().insert(layer as u16, A);
            }

            let mut DA: Option<nd::Array2<f64>> = None;
            for layer in (1..=self.l).rev() {
                let A = self.cache["A"].get(&(layer as u16)).unwrap();
                let A_prev = self.cache["A"].get(&(layer as u16 - 1)).unwrap();
                let Z = self.cache["Z"].get(&(layer as u16)).unwrap();
                let W = self.cache["W"].get(&(layer as u16)).unwrap();

                let DZ = if layer == self.l {
                    A - &self.Y
                } else {
                    let da = DA.unwrap();
                    &da * &self.activation_derivative(A)
                };

                let DW = DZ.dot(&A_prev.t()) / self.m as f64;
                let DB = DZ.sum_axis(nd::Axis(1)).insert_axis(nd::Axis(1)) / self.m as f64;

                let updated_W = W - &(DW.mapv(|x| x * self.learning_rate));
                let updated_B = self.cache["B"].get(&(layer as u16)).unwrap()
                    - &DB.mapv(|x| x * self.learning_rate);

                self.cache
                    .get_mut("W")
                    .unwrap()
                    .insert(layer as u16, updated_W);
                self.cache
                    .get_mut("B")
                    .unwrap()
                    .insert(layer as u16, updated_B);

                DA = Some(W.t().dot(&DZ));
            }

            if c % 100 == 0 {
                let output = self.cache["A"].get(&(self.l as u16)).unwrap();
                println!("Cycle {} - Cost: {:.6}", c, self.binary_cost(output));
            }
        }
    }

    pub fn predict(&self, X: nd::Array2<f64>) -> nd::Array2<u8> {
        let mut A = X;
        for layer in 1..=self.l {
            let W = self.cache["W"].get(&(layer as u16)).unwrap();
            let B = self.cache["B"].get(&(layer as u16)).unwrap();
            let Z = W.dot(&A) + B;
            A = self.activation(&Z, layer == self.l);
        }
        A.mapv(|x| if x > 0.5 { 1 } else { 0 })
    }
}
