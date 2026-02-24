use core::num;
use std::collections::VecDeque;

use burn::{Tensor, module::{Ignored, Module}, nn::{Linear, LinearConfig, PaddingConfig2d, conv::{Conv2d, Conv2dConfig}, loss::CrossEntropyLossConfig, pool::{AvgPool2d, AvgPool2dConfig, MaxPool2d, MaxPool2dConfig}}, prelude::Backend, tensor::{Device, backend::AutodiffBackend}, train::{TrainOutput, TrainStep}};

use crate::dtos::NodeDtoJSON;

#[derive(Clone, Debug)]
pub enum DynamicTensor<B: Backend> {
    Dim2(Tensor<B, 2>),
    Dim4(Tensor<B, 4>),
}

#[derive(Module, Debug)]
pub enum Layer<B:Backend> {
    Conv2d(Conv2d<B>),
    Dense(Linear<B>),
    MaxPool2D(MaxPool2d),
    AvgPool2D(AvgPool2d),
}

#[derive(Clone, Debug)]
pub enum Operation {
    Input(usize),
    Dense{layer_idx: usize, activation: String},
    Conv2D{layer_idx : usize},
    MaxPool {layer_idx: usize},
    AvgPool {layer_idx: usize},
    Flatten,
    Add,
    Concat,
    Output(usize),
}

#[derive(Clone, Debug)]
pub struct Instruction {
    pub node_id: usize,
    pub op: Operation,
    pub input_ids: Vec<usize>,
}

#[derive(Module, Debug)]
pub struct GraphModel<B: Backend> {
    pub layers: Vec<Layer<B>>,

    pub execution_plan: Ignored<Vec<Instruction>>,
    pub use_counts: Vec<usize>,
    pub num_inputs: usize,
    pub num_outputs: usize,
}

impl<B: Backend> GraphModel<B> {
    pub fn build(raw_data: &str, device: &B::Device) -> Self {
        let mut configs = Vec::new();
        let mut edges = Vec::new();
        let mut parsing_connections = false;

        for line in raw_data.lines().map(|l| l.trim()).filter(|l| !l.is_empty()) {
            if line == "CONNECTIONS" {
                parsing_connections = true;
                continue;
            }
            if parsing_connections {
                let parts: Vec<usize> = line.split_whitespace().map(|s| s.parse().unwrap()).collect();
                edges.push((parts[0], parts[1]));
            } else {
                let config: NodeDtoJSON = serde_json::from_str(line).expect("Ошибка парсинга");
                configs.push(config);
            }
        }

        let num_nodes = configs.len();

        let mut in_degrees = vec![0; num_nodes];
        let mut adj_list = vec![vec![]; num_nodes];
        let mut node_inputs = vec![vec![]; num_nodes];
        let mut use_counts = vec![0; num_nodes];

        for &(from, to) in &edges {
            adj_list[from].push(to);
            in_degrees[to] += 1;
            node_inputs[to].push(from);
            use_counts[from] += 1;
        }

        let mut queue: VecDeque<usize> = in_degrees.iter()
            .enumerate()
            .filter(|(_, &deg)| deg == 0)
            .map(|(i, _)|i)
            .collect();

        let mut topo_order = Vec::new();
        while let Some(node) = queue.pop_front() {
            topo_order.push(node);
            for &next_node in &adj_list[node] {
                in_degrees[next_node] -= 1;
                if in_degrees[next_node] == 0 {
                    queue.push_back(next_node);
                }
            }
        }

        let mut layers = Vec::new();
        let mut executuib_plan = Vec::new();
        let mut shape_cache = vec![vec![]; num_nodes];

        let mut num_inputs = 0;
        let mut num_outputs = 0;

        for &node_id in &topo_order {
            let config = &configs[node_id];
            let inputs_for_node = node_inputs[node_id].clone();

            let (op, out_shape) = match config {
                NodeDtoJSON::Conv2D { filters, kernel_size, stride, padding, dilation, use_bias } => {
                    let prev_shape = &shape_cache[inputs_for_node[0]];
                    let d_input = prev_shape[0];
                    
                    let k = if kernel_size.len() == 1 { [kernel_size[0] as usize; 2] }
                        else {[kernel_size[0] as usize, kernel_size[1] as usize]};

                    let conv = Conv2dConfig::new([d_input, *filters as usize], k)
                        .with_stride([*stride as usize; 2])
                        .with_padding(PaddingConfig2d::Explicit(*padding as usize, *padding as usize))
                        .with_dilation([*dilation as usize; 2])
                        .with_bias(*use_bias)
                        .init(device);

                    layers.push(Layer::Conv2d(conv));

                    let h_in = prev_shape[1];
                    let w_in = prev_shape[2];
                    let h_out = (h_in + 2 * (*padding as usize) - (*dilation as usize) * (k[0] - 1) - 1) / (*stride as usize) + 1;
                    let w_out = (w_in + 2 * (*padding as usize) - (*dilation as usize) * (k[1] - 1) - 1) / (*stride as usize) + 1;
                    
                    (Operation::Conv2D { layer_idx: layers.len() - 1 }, vec![*filters as usize, h_out, w_out])
                },
                NodeDtoJSON::Dense { units, activation, use_bias } => {
                    let prev_shape = &shape_cache[inputs_for_node[0]];
                    let d_input = prev_shape[0];

                    let linear = LinearConfig::new(d_input, *units as usize)
                        .with_bias(*use_bias)
                        .init(device);
                    layers.push(Layer::Dense(linear));
                    (Operation::Dense { layer_idx: layers.len() - 1, activation: activation.clone() }, vec![*units as usize])
                },
                NodeDtoJSON::Flatten {} => {
                    let prev_shape = &shape_cache[inputs_for_node[0]];
                    let flat_size = prev_shape[0] * prev_shape[1] * prev_shape[2];
                    (Operation::Flatten, vec![flat_size])
                },
                NodeDtoJSON::Input { output_shape } => {
                    let op = Operation::Input(num_inputs);
                    num_inputs += 1;
                    let shape: Vec<usize> = output_shape.iter().map(|&v| v as usize).collect();
                    (op, shape)
                },
                NodeDtoJSON::Output { input_shape } => {
                    let op = Operation::Output(num_outputs);
                    num_outputs += 1;
                    (op, input_shape.iter().map(|&v| v as usize).collect())
                },
                NodeDtoJSON::Pooling { pool_type, kernel_size, stride, padding } => {
                    let prev_shape = &shape_cache[inputs_for_node[0]];
                    let k = if kernel_size.len() == 1 {[kernel_size[0] as usize; 2]}
                        else {[kernel_size[0] as usize, kernel_size[1] as usize]};

                    let h_out = (prev_shape[1] + 2 * (*padding as usize) - k[0]) / (*stride as usize) + 1;
                    let w_out = (prev_shape[2] + 2 * (*padding as usize) - k[1]) / (*stride as usize) + 1;
                    let out_shape = vec![prev_shape[0], h_out, w_out];

                    let layer_idx = layers.len();
                    if pool_type == "max" {
                        layers.push(Layer::MaxPool2D(MaxPool2dConfig::new(k)
                            .with_strides([*stride as usize; 2])
                            .with_padding(PaddingConfig2d::Explicit(*padding as usize, *padding as usize))
                            .init()
                        ));
                        (Operation::MaxPool { layer_idx }, out_shape)
                    } else {
                        layers.push(Layer::AvgPool2D(
                            AvgPool2dConfig::new(k)
                            .with_strides([*stride as usize; 2])
                            .with_padding(PaddingConfig2d::Explicit(*padding as usize, *padding as usize))
                            .init()
                        ));
                        (Operation::AvgPool { layer_idx }, out_shape)
                    }
                },
                NodeDtoJSON::Add {}=> {
                    (Operation::Add, shape_cache[inputs_for_node[0]].clone())
                },
                NodeDtoJSON::Concat {}=> {
                    (Operation::Add, shape_cache[inputs_for_node[0]].clone())
                }
            };

            shape_cache[node_id] = out_shape;
            executuib_plan.push(Instruction{node_id, op, input_ids: inputs_for_node});

        }

        Self {
            layers,
            execution_plan: Ignored(executuib_plan),
            use_counts: use_counts,
            num_inputs: num_inputs,
            num_outputs: num_outputs
        }
    }

    pub fn forward(&self, inputs: &[DynamicTensor<B>]) -> Vec<DynamicTensor<B>> {
        assert_eq!(inputs.len(), self.num_inputs);
        let mut memory: Vec<Option<DynamicTensor<B>>> = vec![None; self.use_counts.len() as usize];
        let mut remaining_uses = self.use_counts.clone();
        let mut final_outputs = vec![None; self.num_outputs];

        macro_rules! consume {
            ($id:expr) => {{
                remaining_uses[$id] -= 1;
                if remaining_uses[$id] == 0 {memory[$id].take().unwrap()}
                else {memory[$id].as_ref().unwrap().clone()}   
            }};
        }

        for instr in &self.execution_plan.0 {
            if let Operation::Output(out_idx) = &instr.op {
                final_outputs[*out_idx] = Some(consume!(instr.input_ids[0]));
                continue;
            }

            let out_tensor = match &instr.op {
                Operation::Input(idx) => inputs[*idx].clone(),
                Operation::Dense{layer_idx, activation} => {
                    if let (DynamicTensor::Dim2(x), Layer::Dense(linear)) = (consume!(instr.input_ids[0]), &self.layers[*layer_idx]) {
                        let mut out = linear.forward(x);
                        out = match activation.as_str() {
                            "relu" => burn::tensor::activation::relu(out),
                            "leaky_relu" => burn::tensor::activation::leaky_relu(out, 0.01),
                            "softmax" => burn::tensor::activation::softmax(out, 1),
                            _ => {unreachable!("Неверная функция активации")}
                        }; 
                        DynamicTensor::Dim2(out)
                    } else {unreachable!("Dense требует 2D тензор")}
                },
                Operation::Conv2D { layer_idx } => {
                    if let (DynamicTensor::Dim4(x), Layer::Conv2d(conv)) = (consume!(instr.input_ids[0]), &self.layers[*layer_idx]) {
                        DynamicTensor::Dim4(conv.forward(x))
                    } else {unreachable!("Conv2D требует 4D тензор")}
                },
                Operation::MaxPool { layer_idx } => {
                    if let (DynamicTensor::Dim4(x), Layer::MaxPool2D(pool)) = (consume!(instr.input_ids[0]), &self.layers[*layer_idx]) {
                        DynamicTensor::Dim4(pool.forward(x)) 
                    } else {unreachable!()}
                },
                Operation::AvgPool { layer_idx } => {
                    if let (DynamicTensor::Dim4(x), Layer::AvgPool2D(pool)) = (consume!(instr.input_ids[0]), &self.layers[*layer_idx]) {
                        DynamicTensor::Dim4(pool.forward(x))
                    } else {unreachable!()}
                },
                Operation::Flatten => {
                    if let DynamicTensor::Dim4(x) = consume!(!instr.input_ids[0]) {
                        DynamicTensor::Dim2(x.flatten::<2>(1, 3))
                    } else {unreachable!()}
                },
                Operation::Add => {
                    match consume!(instr.input_ids[0]) {
                        DynamicTensor::Dim2(mut sum) => {
                            for &in_id in instr.input_ids.iter().skip(1) {
                                if let DynamicTensor::Dim2(t2) = consume!(in_id) {sum = sum + t2; }
                            }
                            DynamicTensor::Dim2(sum)
                        },
                        DynamicTensor::Dim4(mut sum) => {
                            for &in_id in instr.input_ids.iter().skip(1) {
                                if let DynamicTensor::Dim4(t4) = consume!(in_id) {sum = sum + t4; }
                            }
                            DynamicTensor::Dim4(sum)
                        }
                    }
                },
                Operation::Concat => {
                    let mut tensors = vec![consume!(instr.input_ids[0])];
                    for &in_id in instr.input_ids.iter().skip(1) {
                        tensors.push(consume!(in_id));
                    }

                    match  tensors[0].clone() {
                        DynamicTensor::Dim2(_) => {
                            let t2_list: Vec<_> = tensors.into_iter().map(|t| if let DynamicTensor::Dim2(x) = t {x} 
                                else {unreachable!()}).collect();
                            DynamicTensor::Dim2(Tensor::cat(t2_list, 1))
                        },
                        DynamicTensor::Dim4(_) => {
                            let t4_list: Vec<_> = tensors.into_iter().map(|t| if let DynamicTensor::Dim4(x) = t {x} 
                                else {unreachable!()}).collect();
                            DynamicTensor::Dim4(Tensor::cat(t4_list, 1))
                        }
                    }
                },
                _ => unreachable!(),
            };

            if remaining_uses[instr.node_id] > 0 {memory[instr.node_id] = Some(out_tensor);}
        }

        final_outputs.into_iter().map(|o| o.unwrap()).collect()
    }
}

pub struct DynamicBatch<B: Backend> {
    pub inputs: Vec<DynamicTensor<B>>,
    pub targets: Vec<DynamicTensor<B>>,
}

#[derive(Clone, Debug)]
pub struct MyTrainMetrics<B:Backend> {
    pub loss: Tensor<B, 1>
}

impl<B: AutodiffBackend> TrainStep<DynamicBatch<B>, MyTrainMetrics<B>> for GraphModel<B> {
    
    fn step(&self, batch: DynamicBatch<B>) -> TrainOutput<MyTrainMetrics<B>> {
        
        // 1. FORWARD PASS: Прогоняем входы через нашу динамическую сеть
        // self.forward() вернет Vec<DynamicTensor<B>> (результаты всех выходных нод)
        let predictions = self.forward(&batch.inputs);
        
        // 2. LOSS CALCULATION: Считаем ошибку
        // В динамической сети юзер сам выбирает, какую функцию потерь применять к какому выходу.
        // Для примера предположим, что у нас одна выходная нода и это классификация (Dense слой)
        
        let loss = if let (DynamicTensor::Dim2(pred), DynamicTensor::Dim2(target)) = (&predictions[0], &batch.targets[0]) {
            // Pred: [Batch, NumClasses], Target: [Batch, NumClasses] (One-hot)
            // Применяем CrossEntropyLoss
            let loss_func = CrossEntropyLossConfig::new().init(&pred.device());
            
            // Внимание: Burn ожидает таргеты как индексы (Tensor<B, 1, Int>), 
            // так что здесь понадобится конвертация или правильный парсинг в Batcher-е.
            // Для упрощения показываю абстрактно:
            loss_func.forward(pred.clone(), target_as_int) 
            
        } else {
            panic!("Несовпадение размерностей для вычисления Loss!");
        };

        // 3. BACKWARD PASS: Магия автодифференцирования
        // Метод backward() вычисляет градиенты для всех весов модели (Conv2D, Dense и т.д.),
        // которые участвовали в получении этого loss.
        let gradients = loss.backward();

        // 4. ВОЗВРАТ РЕЗУЛЬТАТА
        // Собираем метрики (чтобы Learner мог их залогировать)
        let metrics = MyTrainMetrics { loss: loss.clone() };
        
        // Передаем модель, градиенты и метрики фреймворку
        TrainOutput::new(self, gradients, metrics)
    }
}