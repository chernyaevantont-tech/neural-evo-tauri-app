use std::collections::VecDeque;
use std::sync::Arc;

use burn::{
    Tensor,
    data::{
        dataloader::{DataLoader, DataLoaderBuilder, batcher::Batcher},
        dataset::InMemDataset,
    },
    lr_scheduler::constant::ConstantLr,
    module::{Ignored, Module},
    nn::{
        Linear, LinearConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
        loss::{CrossEntropyLossConfig, MseLoss},
        pool::{AvgPool2d, AvgPool2dConfig, MaxPool2d, MaxPool2dConfig},
    },
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::Backend,
    tensor::{Int, backend::AutodiffBackend},
    train::{
        InferenceStep, ItemLazy, Learner, SupervisedTraining, TrainOutput, TrainStep,
        metric::{Adaptor, LossInput, LossMetric},
    },
};

use crate::dtos::NodeDtoJSON;

// ---------------------------------------------------------------------------
// Тензорные типы
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum DynamicTensor<B: Backend> {
    Dim2(Tensor<B, 2>),
    Dim4(Tensor<B, 4>),
}

pub fn concat_dynamic_tensors<B: Backend>(tensors: Vec<DynamicTensor<B>>) -> DynamicTensor<B> {
    if tensors.is_empty() {
        panic!("Cannot concatenate empty tensor list");
    }
    match &tensors[0] {
        DynamicTensor::Dim2(_) => {
            let t2_list: Vec<Tensor<B, 2>> = tensors
                .into_iter()
                .map(|t| match t {
                    DynamicTensor::Dim2(t2) => t2,
                    _ => panic!("Mixed tensor dimensions in concat!"),
                })
                .collect();
            DynamicTensor::Dim2(Tensor::cat(t2_list, 0))
        }
        DynamicTensor::Dim4(_) => {
            let t4_list: Vec<Tensor<B, 4>> = tensors
                .into_iter()
                .map(|t| match t {
                    DynamicTensor::Dim4(t4) => t4,
                    _ => panic!("Mixed tensor dimensions in concat!"),
                })
                .collect();
            DynamicTensor::Dim4(Tensor::cat(t4_list, 0))
        }
    }
}

// ---------------------------------------------------------------------------
// Слои (обёртки для burn-слоёв)
// ---------------------------------------------------------------------------

// Removed Layer enum to avoid hiding parameters from Burn

// ---------------------------------------------------------------------------
// Описание операции одной ноды
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum Operation {
    Input(usize),
    Dense {
        dense_idx: usize,
        activation: String,
    },
    Conv2D {
        conv2d_idx: usize,
        activation: String,
    },
    MaxPool {
        pool_idx: usize,
    },
    AvgPool {
        pool_idx: usize,
    },
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

// ---------------------------------------------------------------------------
// Основная модель-граф
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct GraphModel<B: Backend> {
    pub conv2ds: Vec<Conv2d<B>>,
    pub denses: Vec<Linear<B>>,
    pub max_pools: Vec<MaxPool2d>,
    pub avg_pools: Vec<AvgPool2d>,

    pub execution_plan: Ignored<Vec<Instruction>>,
    pub use_counts: Vec<usize>,
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub input_shapes: Vec<Vec<usize>>,
    pub output_shapes: Vec<Vec<usize>>,
}

// ---------------------------------------------------------------------------
// Batch и выходные структуры
// ---------------------------------------------------------------------------

/// Батч данных для обучения / инференса.
#[derive(Clone, Debug)]
pub struct DynamicBatch<B: Backend> {
    pub inputs: Vec<DynamicTensor<B>>,
    pub targets: Vec<DynamicTensor<B>>,
}

/// Выход одного шага (содержит loss для метрики).
#[derive(Clone, Debug)]
pub struct DynamicOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
}

// ---------------------------------------------------------------------------
// ItemLazy — нужен burn‑train для асинхронной обработки метрик
// ---------------------------------------------------------------------------

/// «Синхронная» версия `DynamicOutput` (после sync).
#[derive(Clone, Debug)]
pub struct DynamicOutputSync {
    pub loss_value: f32,
}

impl<B: Backend> ItemLazy for DynamicOutput<B> {
    type ItemSync = DynamicOutputSync;

    fn sync(self) -> Self::ItemSync {
        let loss_value = self.loss.into_data().to_vec::<f32>().unwrap()[0];
        DynamicOutputSync { loss_value }
    }
}

// ---------------------------------------------------------------------------
// Adaptor<LossInput<B>> — мост от DynamicOutputSync к LossMetric
// ---------------------------------------------------------------------------

impl<B: Backend> Adaptor<LossInput<B>> for DynamicOutputSync {
    fn adapt(&self) -> LossInput<B> {
        let device = B::Device::default();
        let loss_tensor = Tensor::<B, 1>::from_floats([self.loss_value], &device);
        LossInput::new(loss_tensor)
    }
}

// ---------------------------------------------------------------------------
// Конструктор модели
// ---------------------------------------------------------------------------

impl<B: Backend> GraphModel<B> {
    /// Строит `GraphModel` из текстового описания генома.
    ///
    /// Формат:
    /// ```text
    /// <json-строка для ноды 0>
    /// <json-строка для ноды 1>
    /// ...
    /// CONNECTIONS
    /// 0 1
    /// 1 2
    /// ...
    /// ```
    pub fn build(
        raw_data: &str,
        device: &B::Device,
        input_shape_overrides: Option<&[Vec<usize>]>,
        output_shape_overrides: Option<&[Vec<usize>]>,
    ) -> Self {
        let mut configs = Vec::new();
        let mut edges = Vec::new();
        let mut parsing_connections = false;

        for line in raw_data.lines().map(|l| l.trim()).filter(|l| !l.is_empty()) {
            if line == "CONNECTIONS" {
                parsing_connections = true;
                continue;
            }
            if parsing_connections {
                let parts: Vec<usize> = line
                    .split_whitespace()
                    .map(|s| s.parse().unwrap())
                    .collect();
                edges.push((parts[0], parts[1]));
            } else {
                let config: NodeDtoJSON = serde_json::from_str(line).expect("Ошибка парсинга ноды");
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

        // Топологическая сортировка (BFS / Кан)
        let mut queue: VecDeque<usize> = in_degrees
            .iter()
            .enumerate()
            .filter(|&(_, &deg)| deg == 0)
            .map(|(i, _)| i)
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

        let mut conv2ds = Vec::new();
        let mut denses = Vec::new();
        let mut max_pools = Vec::new();
        let mut avg_pools = Vec::new();
        let mut execution_plan: Vec<Instruction> = Vec::new();
        let mut shape_cache = vec![vec![]; num_nodes];

        let mut num_inputs = 0;
        let mut num_outputs = 0;

        let mut input_shapes = Vec::new();
        let mut output_shapes = Vec::new();

        for &node_id in &topo_order {
            let config = &configs[node_id];
            let inputs_for_node = node_inputs[node_id].clone();

            let (op, out_shape) = match config {
                NodeDtoJSON::Conv2D {
                    filters,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    use_bias,
                    activation,
                } => {
                    let prev_shape = &shape_cache[inputs_for_node[0]];
                    let d_input = prev_shape[0];

                    let k = [kernel_size.h as usize, kernel_size.w as usize];

                    let conv = Conv2dConfig::new([d_input, *filters as usize], k)
                        .with_stride([*stride as usize; 2])
                        .with_padding(PaddingConfig2d::Explicit(
                            *padding as usize,
                            *padding as usize,
                        ))
                        .with_dilation([*dilation as usize; 2])
                        .with_bias(*use_bias)
                        .with_initializer(burn::nn::Initializer::KaimingNormal {
                            gain: 1.0,
                            fan_out_only: false,
                        })
                        .init(device);

                    conv2ds.push(conv);

                    let h_in = prev_shape[1];
                    let w_in = prev_shape[2];
                    let h_out = (h_in + 2 * (*padding as usize))
                        .saturating_sub((*dilation as usize) * (k[0] - 1) + 1)
                        / (*stride as usize)
                        + 1;
                    let w_out = (w_in + 2 * (*padding as usize))
                        .saturating_sub((*dilation as usize) * (k[1] - 1) + 1)
                        / (*stride as usize)
                        + 1;

                    (
                        Operation::Conv2D {
                            conv2d_idx: conv2ds.len() - 1,
                            activation: activation.clone(),
                        },
                        vec![*filters as usize, h_out, w_out],
                    )
                }

                NodeDtoJSON::Dense {
                    units,
                    activation,
                    use_bias,
                } => {
                    let prev_shape = &shape_cache[inputs_for_node[0]];
                    let d_input = prev_shape[0];

                    // Guard: prevent absurdly large weight matrices (e.g., 254016 × 6272 = 1.6B params)
                    const MAX_DENSE_PARAMS: usize = 50_000_000; // 50M params ~= 200MB
                    let param_count = d_input * (*units as usize);
                    if param_count > MAX_DENSE_PARAMS {
                        panic!(
                            "Dense layer too large: {}×{} = {} params (max {}). Architecture is infeasible.",
                            d_input, units, param_count, MAX_DENSE_PARAMS
                        );
                    }

                    let linear = LinearConfig::new(d_input, *units as usize)
                        .with_bias(*use_bias)
                        .with_initializer(burn::nn::Initializer::KaimingNormal {
                            gain: 1.0,
                            fan_out_only: false,
                        })
                        .init(device);
                    denses.push(linear);
                    (
                        Operation::Dense {
                            dense_idx: denses.len() - 1,
                            activation: activation.clone(),
                        },
                        vec![*units as usize],
                    )
                }

                NodeDtoJSON::Flatten {} => {
                    let prev_shape = &shape_cache[inputs_for_node[0]];
                    let flat_size = prev_shape[0] * prev_shape[1] * prev_shape[2];
                    (Operation::Flatten, vec![flat_size])
                }

                NodeDtoJSON::Input { output_shape } => {
                    let op = Operation::Input(num_inputs);
                    let input_idx = num_inputs;
                    num_inputs += 1;

                    // Use override if provided, otherwise fall back to genome JSON
                    let internal_shape = if let Some(overrides) = input_shape_overrides {
                        if input_idx < overrides.len() {
                            let ov = &overrides[input_idx];
                            input_shapes.push(ov.clone());
                            ov.clone() // Already in [C, H, W] format from caller
                        } else {
                            let shape: Vec<usize> =
                                output_shape.iter().map(|&v| v as usize).collect();
                            input_shapes.push(shape.clone());
                            if shape.len() == 3 {
                                vec![shape[2], shape[0], shape[1]]
                            } else {
                                shape
                            }
                        }
                    } else {
                        let shape: Vec<usize> = output_shape.iter().map(|&v| v as usize).collect();
                        input_shapes.push(shape.clone());
                        if shape.len() == 3 {
                            vec![shape[2], shape[0], shape[1]]
                        } else {
                            shape
                        }
                    };

                    (op, internal_shape)
                }

                NodeDtoJSON::Output { input_shape } => {
                    let op = Operation::Output(num_outputs);
                    let output_idx = num_outputs;
                    num_outputs += 1;

                    let shape = if let Some(overrides) = output_shape_overrides {
                        if output_idx < overrides.len() {
                            let ov = overrides[output_idx].clone();

                            // Format the predecessor Dense layer to match new output requirements
                            if inputs_for_node.len() == 1 {
                                let pred_id = inputs_for_node[0];
                                let pred_shape = &shape_cache[pred_id];

                                // Find the Dense layer for the predecessor
                                if let Some(instr) =
                                    execution_plan.iter().find(|i| i.node_id == pred_id)
                                {
                                    if let Operation::Dense { dense_idx, .. } = &instr.op {
                                        let layer_idx_val = *dense_idx;
                                        let pred_inputs = &node_inputs[pred_id];

                                        // 1. Rebuild linear layer if units don't match dataset classes
                                        if pred_shape.len() == 1
                                            && ov.len() == 1
                                            && pred_shape[0] != ov[0]
                                        {
                                            let d_input = if !pred_inputs.is_empty() {
                                                shape_cache[pred_inputs[0]][0]
                                            } else {
                                                pred_shape[0] // Fallback
                                            };
                                            let new_linear =
                                                burn::nn::LinearConfig::new(d_input, ov[0])
                                                    .with_initializer(
                                                        burn::nn::Initializer::KaimingNormal {
                                                            gain: 1.0,
                                                            fan_out_only: false,
                                                        },
                                                    )
                                                    .init(device);
                                            denses[layer_idx_val] = new_linear;
                                            shape_cache[pred_id] = ov.clone();

                                            eprintln!(
                                                ">>> Overrode predecessor Dense (node {}) to {} units",
                                                pred_id, ov[0]
                                            );
                                        }

                                        // 2. ALWAYS force "linear" activation on pre-Output Dense
                                        // CrossEntropyLoss applies log_softmax internally,
                                        // so the last Dense must output raw logits.
                                        if let Some(instr_mut) =
                                            execution_plan.iter_mut().find(|i| i.node_id == pred_id)
                                        {
                                            if let Operation::Dense { activation, .. } =
                                                &mut instr_mut.op
                                            {
                                                if activation != "linear" {
                                                    eprintln!(
                                                        ">>> Forcing pre-Output Dense (node {}) activation from '{}' to 'linear'",
                                                        pred_id, activation
                                                    );
                                                    *activation = "linear".to_string();
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            output_shapes.push(ov.clone());
                            ov
                        } else {
                            let s: Vec<usize> = input_shape.iter().map(|&v| v as usize).collect();
                            output_shapes.push(s.clone());
                            s
                        }
                    } else {
                        let s: Vec<usize> = input_shape.iter().map(|&v| v as usize).collect();
                        output_shapes.push(s.clone());
                        s
                    };
                    (op, shape)
                }

                NodeDtoJSON::Pooling {
                    pool_type,
                    kernel_size,
                    stride,
                    padding,
                } => {
                    let prev_shape = &shape_cache[inputs_for_node[0]];

                    let k = [kernel_size.h as usize, kernel_size.w as usize];

                    let h_out = (prev_shape[1] + 2 * (*padding as usize)).saturating_sub(k[0])
                        / (*stride as usize)
                        + 1;
                    let w_out = (prev_shape[2] + 2 * (*padding as usize)).saturating_sub(k[1])
                        / (*stride as usize)
                        + 1;
                    let out_shape = vec![prev_shape[0], h_out, w_out];

                    if pool_type == "max" {
                        max_pools.push(
                            MaxPool2dConfig::new(k)
                                .with_strides([*stride as usize; 2])
                                .with_padding(PaddingConfig2d::Explicit(
                                    *padding as usize,
                                    *padding as usize,
                                ))
                                .init(),
                        );
                        (
                            Operation::MaxPool {
                                pool_idx: max_pools.len() - 1,
                            },
                            out_shape,
                        )
                    } else {
                        avg_pools.push(
                            AvgPool2dConfig::new(k)
                                .with_strides([*stride as usize; 2])
                                .with_padding(PaddingConfig2d::Explicit(
                                    *padding as usize,
                                    *padding as usize,
                                ))
                                .init(),
                        );
                        (
                            Operation::AvgPool {
                                pool_idx: avg_pools.len() - 1,
                            },
                            out_shape,
                        )
                    }
                }

                NodeDtoJSON::Add {} => {
                    let first_id = inputs_for_node[0];
                    let expected_shape = &shape_cache[first_id];

                    for &in_id in inputs_for_node.iter().skip(1) {
                        let actual_shape = &shape_cache[in_id];
                        if expected_shape != actual_shape {
                            panic!(
                                "AddNode shape mismatch: Node {} shape {:?} vs Node {} shape {:?}",
                                first_id, expected_shape, in_id, actual_shape
                            );
                        }
                    }

                    (Operation::Add, expected_shape.clone())
                }

                NodeDtoJSON::Concat {} => {
                    let first_id = inputs_for_node[0];
                    let first_shape = &shape_cache[first_id];

                    let mut total_channels = first_shape[0];
                    for &in_id in inputs_for_node.iter().skip(1) {
                        let actual_shape = &shape_cache[in_id];

                        // Concat validation (assuming concatenation on channel dimension 0)
                        if first_shape.len() != actual_shape.len() {
                            panic!(
                                "ConcatNode dimension mismatch: Node {} ({:?}) vs Node {} ({:?})",
                                first_id, first_shape, in_id, actual_shape
                            );
                        }

                        if first_shape.len() == 3 {
                            // Spatial dims (H, W) must match
                            if first_shape[1] != actual_shape[1]
                                || first_shape[2] != actual_shape[2]
                            {
                                panic!(
                                    "ConcatNode spatial mismatch: Node {} ({:?}) vs Node {} ({:?})",
                                    first_id, first_shape, in_id, actual_shape
                                );
                            }
                        }

                        total_channels += actual_shape[0];
                    }

                    let out_shape = if first_shape.len() == 3 {
                        vec![total_channels, first_shape[1], first_shape[2]]
                    } else {
                        vec![total_channels]
                    };
                    (Operation::Concat, out_shape)
                }
            };

            // Polymorphic spatial validation: catch any dimension collapsing to 0
            if out_shape.iter().any(|&d| d == 0) {
                eprintln!(
                    ">>> WARNING: Node {} ({:?}) produced a zero-dimension shape {:?}. Skipping genome.",
                    node_id, op, out_shape
                );
                panic!(
                    "Spatial dimension collapsed to 0 at node {} ({:?}), shape: {:?}",
                    node_id, op, out_shape
                );
            }

            println!(
                "Compiled Node {node_id}: {:?} -> Out Shape: {:?}",
                op, out_shape
            );
            shape_cache[node_id] = out_shape;
            execution_plan.push(Instruction {
                node_id,
                op,
                input_ids: inputs_for_node,
            });
        }

        Self {
            conv2ds,
            denses,
            max_pools,
            avg_pools,
            execution_plan: Ignored(execution_plan),
            use_counts,
            num_inputs,
            num_outputs,
            input_shapes,
            output_shapes,
        }
    }

    // -----------------------------------------------------------------------
    // Прямой проход
    // -----------------------------------------------------------------------

    pub fn forward(&self, inputs: &[DynamicTensor<B>]) -> Vec<DynamicTensor<B>> {
        self.forward_internal(inputs, false)
    }

    pub fn forward_internal(
        &self,
        inputs: &[DynamicTensor<B>],
        debug: bool,
    ) -> Vec<DynamicTensor<B>> {
        assert_eq!(inputs.len(), self.num_inputs);

        let mut memory: Vec<Option<DynamicTensor<B>>> = vec![None; self.use_counts.len()];
        let mut remaining_uses = self.use_counts.clone();
        let mut final_outputs = vec![None; self.num_outputs];

        macro_rules! consume {
            ($id:expr) => {{
                remaining_uses[$id] -= 1;
                if remaining_uses[$id] == 0 {
                    memory[$id].take().unwrap()
                } else {
                    memory[$id].as_ref().unwrap().clone()
                }
            }};
        }

        for instr in &self.execution_plan.0 {
            if let Operation::Output(out_idx) = &instr.op {
                final_outputs[*out_idx] = Some(consume!(instr.input_ids[0]));
                continue;
            }

            let out_tensor = match &instr.op {
                Operation::Input(idx) => inputs[*idx].clone(),

                Operation::Dense {
                    dense_idx,
                    activation,
                } => {
                    if let DynamicTensor::Dim2(x) = consume!(instr.input_ids[0]) {
                        let linear = &self.denses[*dense_idx];
                        let mut out = linear.forward(x);
                        out = match activation.as_str() {
                            "relu" => burn::tensor::activation::relu(out),
                            "leaky_relu" => burn::tensor::activation::leaky_relu(out, 0.01),
                            "softmax" => burn::tensor::activation::softmax(out, 1),
                            "sigmoid" => burn::tensor::activation::sigmoid(out),
                            "linear" | "none" => out,
                            _ => out,
                        };
                        DynamicTensor::Dim2(out)
                    } else {
                        unreachable!("Dense требует 2D тензор")
                    }
                }

                Operation::Conv2D {
                    conv2d_idx,
                    activation,
                } => {
                    if let DynamicTensor::Dim4(x) = consume!(instr.input_ids[0]) {
                        let conv = &self.conv2ds[*conv2d_idx];
                        let mut out = conv.forward(x);
                        out = match activation.as_str() {
                            "relu" => burn::tensor::activation::relu(out),
                            "leaky_relu" => burn::tensor::activation::leaky_relu(out, 0.01),
                            "sigmoid" => burn::tensor::activation::sigmoid(out),
                            "linear" | "none" => out,
                            _ => burn::tensor::activation::relu(out), // default to relu
                        };
                        DynamicTensor::Dim4(out)
                    } else {
                        unreachable!("Conv2D требует 4D тензор")
                    }
                }

                Operation::MaxPool { pool_idx } => {
                    if let DynamicTensor::Dim4(x) = consume!(instr.input_ids[0]) {
                        let pool = &self.max_pools[*pool_idx];
                        DynamicTensor::Dim4(pool.forward(x))
                    } else {
                        unreachable!()
                    }
                }

                Operation::AvgPool { pool_idx } => {
                    if let DynamicTensor::Dim4(x) = consume!(instr.input_ids[0]) {
                        let pool = &self.avg_pools[*pool_idx];
                        DynamicTensor::Dim4(pool.forward(x))
                    } else {
                        unreachable!()
                    }
                }

                Operation::Flatten => {
                    if let DynamicTensor::Dim4(x) = consume!(instr.input_ids[0]) {
                        DynamicTensor::Dim2(x.flatten::<2>(1, 3))
                    } else {
                        unreachable!()
                    }
                }

                Operation::Add => match consume!(instr.input_ids[0]) {
                    DynamicTensor::Dim2(mut sum) => {
                        for &in_id in instr.input_ids.iter().skip(1) {
                            if let DynamicTensor::Dim2(t2) = consume!(in_id) {
                                sum = sum + t2;
                            }
                        }
                        DynamicTensor::Dim2(sum)
                    }
                    DynamicTensor::Dim4(mut sum) => {
                        for &in_id in instr.input_ids.iter().skip(1) {
                            if let DynamicTensor::Dim4(t4) = consume!(in_id) {
                                sum = sum + t4;
                            }
                        }
                        DynamicTensor::Dim4(sum)
                    }
                },

                Operation::Concat => {
                    let mut tensors = vec![consume!(instr.input_ids[0])];
                    for &in_id in instr.input_ids.iter().skip(1) {
                        tensors.push(consume!(in_id));
                    }

                    match tensors[0].clone() {
                        DynamicTensor::Dim2(_) => {
                            let t2_list: Vec<_> = tensors
                                .into_iter()
                                .map(|t| {
                                    if let DynamicTensor::Dim2(x) = t {
                                        x
                                    } else {
                                        unreachable!()
                                    }
                                })
                                .collect();
                            DynamicTensor::Dim2(Tensor::cat(t2_list, 1))
                        }
                        DynamicTensor::Dim4(_) => {
                            let t4_list: Vec<_> = tensors
                                .into_iter()
                                .map(|t| {
                                    if let DynamicTensor::Dim4(x) = t {
                                        x
                                    } else {
                                        unreachable!()
                                    }
                                })
                                .collect();
                            DynamicTensor::Dim4(Tensor::cat(t4_list, 1))
                        }
                    }
                }

                _ => unreachable!(),
            };

            if debug {
                let shape_str = match &out_tensor {
                    DynamicTensor::Dim2(t) => format!("{:?}", t.dims()),
                    DynamicTensor::Dim4(t) => format!("{:?}", t.dims()),
                };
                println!(
                    "[DEBUG Step] Node {} | Op: {:?} | Generated Output Shape: {}",
                    instr.node_id, instr.op, shape_str
                );
            }

            if remaining_uses[instr.node_id] > 0 {
                memory[instr.node_id] = Some(out_tensor);
            }
        }

        final_outputs.into_iter().map(|o| o.unwrap()).collect()
    }

    // -----------------------------------------------------------------------
    // Вычисление функции потерь
    // -----------------------------------------------------------------------

    pub fn compute_loss(
        &self,
        predictions: &[DynamicTensor<B>],
        targets: &[DynamicTensor<B>],
    ) -> Tensor<B, 1> {
        assert_eq!(
            predictions.len(),
            targets.len(),
            "Количество выходов не совпадает с количеством таргетов!"
        );

        let mut total_loss: Option<Tensor<B, 1>> = None;

        for i in 0..predictions.len() {
            let current_loss = match (&predictions[i], &targets[i]) {
                (DynamicTensor::Dim2(p), DynamicTensor::Dim2(t)) => {
                    let batch_size = p.dims()[0];
                    let out_features = p.dims()[1];

                    if out_features == 1 {
                        // For a final node with 1 unit (binary classification or regression),
                        // interpret floats and use MSE
                        let loss_func = MseLoss::new();
                        loss_func.forward(p.clone(), t.clone(), burn::nn::loss::Reduction::Mean)
                    } else {
                        // Multi-class classification via CrossEntropy (expects `out_features` > 1)
                        let t_int: Tensor<B, 1, Int> = t.clone().reshape([batch_size]).int();
                        let loss_func = CrossEntropyLossConfig::new().init(&p.device());
                        loss_func.forward(p.clone(), t_int)
                    }
                }
                // Регрессия / Image-to-Image (Dim4)
                (DynamicTensor::Dim4(p), DynamicTensor::Dim4(t)) => {
                    let loss_func = MseLoss::new();
                    loss_func.forward(p.clone(), t.clone(), burn::nn::loss::Reduction::Mean)
                }
                _ => panic!("Неподдерживаемая комбинация размерностей для выхода {i}"),
            };

            total_loss = match total_loss {
                Some(acc) => Some(acc.add(current_loss)),
                None => Some(current_loss),
            };
        }

        total_loss.expect("Граф модели должен иметь как минимум одну Output ноду!")
    }
}

// ---------------------------------------------------------------------------
// TrainStep — шаг обучения (с градиентами)
// ---------------------------------------------------------------------------

impl<B: AutodiffBackend> TrainStep for GraphModel<B> {
    type Input = DynamicBatch<B>;
    type Output = DynamicOutput<B>;

    fn step(&self, batch: DynamicBatch<B>) -> TrainOutput<DynamicOutput<B>> {
        // 1. Прямой проход
        let predictions = self.forward(&batch.inputs);

        // 2. Вычисление ошибки
        let loss = self.compute_loss(&predictions, &batch.targets);

        // 3. Обратный проход
        let grads = loss.backward();

        // 4. Формируем результат
        let item = DynamicOutput { loss: loss.clone() };

        TrainOutput::new(self, grads, item)
    }
}

// ---------------------------------------------------------------------------
// InferenceStep — шаг валидации (без градиентов)
// ---------------------------------------------------------------------------

impl<B: Backend> InferenceStep for GraphModel<B> {
    type Input = DynamicBatch<B>;
    type Output = DynamicOutput<B>;

    fn step(&self, batch: DynamicBatch<B>) -> DynamicOutput<B> {
        let predictions = self.forward(&batch.inputs);
        let loss = self.compute_loss(&predictions, &batch.targets);
        DynamicOutput { loss }
    }
}

// ---------------------------------------------------------------------------
// Batcher — превращает вектор батчей в один DynamicBatch (для DataLoader)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct DynamicBatcher<B: Backend> {
    pub device: B::Device,
}

impl<B: Backend> DynamicBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<B, DynamicBatch<B>, DynamicBatch<B>> for DynamicBatcher<B> {
    fn batch(&self, items: Vec<DynamicBatch<B>>, _device: &B::Device) -> DynamicBatch<B> {
        // В данной реализации каждый «элемент» уже является готовым батчем,
        // поэтому просто берём первый. Если нужна полноценная пакетизация
        // из отдельных семплов — расширьте эту логику.
        assert!(
            !items.is_empty(),
            "DataLoader вернул пустой набор элементов"
        );
        items.into_iter().next().unwrap()
    }
}

// ---------------------------------------------------------------------------
// Конфигурация тренировки
// ---------------------------------------------------------------------------

/// Настройки тренировки, передаваемые с фронтенда.
#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub learning_rate: f64,
    pub num_epochs: usize,
    pub checkpoint_dir: String,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            num_epochs: 10,
            checkpoint_dir: "./checkpoints".into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Функция запуска тренировки
// ---------------------------------------------------------------------------

/// Запускает полный цикл обучения модели.
///
/// # Аргументы
/// * `model` — собранная `GraphModel`
/// * `train_data` — вектор батчей для обучения
/// * `valid_data` — вектор батчей для валидации
/// * `config` — настройки обучения
///
/// # Возвращает
/// Обученную модель (inner module, без autodiff обёртки).
pub fn train<B: AutodiffBackend>(
    model: GraphModel<B>,
    train_data: Vec<DynamicBatch<B>>,
    valid_data: Vec<DynamicBatch<B::InnerBackend>>,
    config: TrainConfig,
    device: &B::Device,
) -> GraphModel<B::InnerBackend> {
    // 1. Оптимизатор
    let optimizer = AdamConfig::new().init();

    // 2. Learning Rate Scheduler (постоянный LR)
    let lr_scheduler = ConstantLr::new(config.learning_rate);

    // 3. DataLoader для train
    let batcher_train = DynamicBatcher::<B>::new(device.clone());
    let dataloader_train: Arc<dyn DataLoader<B, DynamicBatch<B>>> =
        DataLoaderBuilder::new(batcher_train)
            .batch_size(1) // каждый элемент уже батч
            .num_workers(1)
            .build(InMemDataset::new(train_data));

    // 4. DataLoader для validation
    let inner_device = <<B as AutodiffBackend>::InnerBackend as Backend>::Device::default();
    let batcher_valid = DynamicBatcher::<B::InnerBackend>::new(inner_device);
    let dataloader_valid: Arc<dyn DataLoader<B::InnerBackend, DynamicBatch<B::InnerBackend>>> =
        DataLoaderBuilder::new(batcher_valid)
            .batch_size(1)
            .num_workers(1)
            .build(InMemDataset::new(valid_data));

    // 5. Learner (модель + оптимизатор + LR scheduler)
    let learner = Learner::new(model, optimizer, lr_scheduler);

    // 6. Запуск тренировки через SupervisedTraining
    let result =
        SupervisedTraining::new(&config.checkpoint_dir, dataloader_train, dataloader_valid)
            .num_epochs(config.num_epochs)
            .metric_train_numeric(LossMetric::<B>::new())
            .metric_valid_numeric(LossMetric::<B::InnerBackend>::new())
            .summary()
            .launch(learner);

    result.model
}

// ---------------------------------------------------------------------------
// Простой цикл обучения (без burn Learner, для прототипирования)
// ---------------------------------------------------------------------------

fn compute_accuracy<B: AutodiffBackend>(
    predictions: &[DynamicTensor<B>],
    targets: &[DynamicTensor<B>],
    is_classification: bool,
) -> (usize, usize) {
    if predictions.is_empty() || !is_classification {
        return (0, 0);
    }

    // Берем только первый выход для подсчета (обычно один output слой)
    if let (DynamicTensor::Dim2(p), DynamicTensor::Dim2(t)) = (&predictions[0], &targets[0]) {
        let (batch_size, out_features) = (p.dims()[0], p.dims()[1]);
        if out_features > 1 {
            // multiclass, use argmax
            let preds_max = p.clone().argmax(1).into_data().to_vec::<i32>().unwrap();
            let targets_int = t
                .clone()
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .into_iter()
                .map(|f| f as i32)
                .collect::<Vec<_>>();
            let mut correct = 0;
            for i in 0..batch_size {
                if preds_max[i] == targets_int[i] {
                    correct += 1;
                }
            }
            return (correct, batch_size);
        } else {
            // binary or regression map to 0.5 threshold
            let preds_vals = p.clone().into_data().to_vec::<f32>().unwrap();
            let targets_vals = t.clone().into_data().to_vec::<f32>().unwrap();
            let mut correct = 0;
            for i in 0..batch_size {
                let p_class = if preds_vals[i] > 0.5 { 1.0 } else { 0.0 };
                if (p_class - targets_vals[i]).abs() < 0.1 {
                    // Match!
                    correct += 1;
                }
            }
            return (correct, batch_size);
        }
    }
    (0, 0)
}

/// Простой тренировочный цикл «вручную» — без SupervisedTraining.
///
/// Полезен для быстрого прототипирования и когда не нужны все
/// возможности burn Learner (метрики, чекпоинты и т.д.).
pub fn train_simple<B: AutodiffBackend>(
    mut model: GraphModel<B>,
    train_batches: &[DynamicBatch<B>],
    valid_batches: &[DynamicBatch<B>],
    test_batches: &[DynamicBatch<B>],
    num_epochs: usize,
    learning_rate: f64,
) -> GraphModel<B> {
    let mut optim = AdamConfig::new().init::<B, GraphModel<B>>();

    for epoch in 0..num_epochs {
        // --- TRAIN PHASE ---
        let mut train_loss_sum = 0.0f32;
        let mut train_correct = 0;
        let mut train_total = 0;

        for (batch_idx, batch) in train_batches.iter().enumerate() {
            let is_first_step = epoch == 0 && batch_idx == 0;
            if is_first_step {
                println!("=== Starting first forward pass with debug logs ===");
            }
            // Прямой проход
            let predictions = model.forward_internal(&batch.inputs, is_first_step);

            // Вычисление loss
            let loss = model.compute_loss(&predictions, &batch.targets);
            if is_first_step {
                let loss_val = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
                println!("=== First pass complete. Loss: {:.6} ===", loss_val);
            }

            // Обратный проход
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &model);

            // Обновление весов
            model = optim.step(learning_rate, model, grads_params);

            train_loss_sum += loss.into_data().to_vec::<f32>().unwrap()[0];

            // Расчет Accuracy (если классификация, 1D вектор out_features > 1)
            let (batch_correct, batch_total) = compute_accuracy(&predictions, &batch.targets, true);
            train_correct += batch_correct;
            train_total += batch_total;
        }

        let train_avg_loss = train_loss_sum / train_batches.len().max(1) as f32;
        let train_acc = if train_total > 0 {
            (train_correct as f32 / train_total as f32) * 100.0
        } else {
            0.0
        };

        // --- VALIDATION PHASE ---
        let mut valid_loss_sum = 0.0f32;
        let mut valid_correct = 0;
        let mut valid_total = 0;

        for batch in valid_batches {
            let predictions = model.forward_internal(&batch.inputs, false);
            let loss = model.compute_loss(&predictions, &batch.targets);
            valid_loss_sum += loss.into_data().to_vec::<f32>().unwrap()[0];

            let (batch_correct, batch_total) = compute_accuracy(&predictions, &batch.targets, true);
            valid_correct += batch_correct;
            valid_total += batch_total;
        }

        let valid_avg_loss = valid_loss_sum / valid_batches.len().max(1) as f32;
        let valid_acc = if valid_total > 0 {
            (valid_correct as f32 / valid_total as f32) * 100.0
        } else {
            0.0
        };

        println!(
            "Epoch {:>3}/{}: Train Loss: {:.4}, Train Acc: {:>6.2}% | Valid Loss: {:.4}, Valid Acc: {:>6.2}%",
            epoch + 1,
            num_epochs,
            train_avg_loss,
            train_acc,
            valid_avg_loss,
            valid_acc
        );
    }

    // --- TEST PHASE ---
    if !test_batches.is_empty() {
        let mut test_correct = 0;
        let mut test_total = 0;
        for batch in test_batches {
            let predictions = model.forward_internal(&batch.inputs, false);
            let (batch_correct, batch_total) = compute_accuracy(&predictions, &batch.targets, true);
            test_correct += batch_correct;
            test_total += batch_total;
        }
        let test_acc = if test_total > 0 {
            (test_correct as f32 / test_total as f32) * 100.0
        } else {
            0.0
        };
        println!("========================================");
        println!("Test Evaluation Complete. Test Acc: {:.2}%", test_acc);
        println!("========================================");
    }

    model
}

#[derive(serde::Serialize, Clone, Debug)]
pub struct BatchMetrics {
    pub epoch: usize,
    pub batch: usize,
    pub total_batches: usize,
    pub loss: f32,
    pub accuracy: f32,
}

#[allow(clippy::type_complexity)]
pub fn run_eval_pass<B: AutodiffBackend>(
    app_handle: &tauri::AppHandle,
    model: &mut GraphModel<B>,
    batches: &[DynamicBatch<B>],
    num_epochs: usize,
    learning_rate: f64,
    is_classification: bool,
    session_counter: &std::sync::atomic::AtomicU64,
    session_snapshot: u64,
) -> (f32, f32) {
    let mut optim = burn::optim::AdamConfig::new().init::<B, GraphModel<B>>();

    let mut final_loss = 999.0;
    let mut final_acc = 0.0;

    let total_batches = batches.len();
    let log_interval = (total_batches / 10).max(1);
    let mut prev_acc = 0.0;

    for epoch in 0..num_epochs {
        // Check cancellation at the start of each epoch
        if session_counter.load(std::sync::atomic::Ordering::SeqCst) != session_snapshot {
            println!("  >>> Training cancelled at epoch {}.", epoch + 1);
            break;
        }

        let mut train_loss_sum = 0.0f32;
        let mut train_correct = 0;
        let mut train_total = 0;

        for (batch_idx, batch) in batches.iter().enumerate() {
            // Check cancellation every 50 batches
            if batch_idx % 50 == 0
                && session_counter.load(std::sync::atomic::Ordering::SeqCst) != session_snapshot
            {
                println!(
                    "  >>> Training cancelled at epoch {} batch {}.",
                    epoch + 1,
                    batch_idx + 1
                );
                return (final_loss, final_acc);
            }

            // Clone inputs/targets so the autodiff graph is fresh each pass
            let cloned_inputs: Vec<DynamicTensor<B>> =
                batch.inputs.iter().map(|t| t.clone()).collect();
            let cloned_targets: Vec<DynamicTensor<B>> =
                batch.targets.iter().map(|t| t.clone()).collect();

            let predictions = model.forward_internal(&cloned_inputs, false);
            let loss = model.compute_loss(&predictions, &cloned_targets);

            let grads = loss.backward();
            let grads_params = burn::optim::GradientsParams::from_grads(grads, model);

            *model = optim.step(learning_rate, model.clone(), grads_params);

            let loss_val = loss.into_data().to_vec::<f32>().unwrap()[0];
            train_loss_sum += loss_val;

            let (batch_correct, batch_total) =
                compute_accuracy(&predictions, &cloned_targets, is_classification);
            train_correct += batch_correct;
            train_total += batch_total;

            if batch_idx % log_interval == 0 || batch_idx == total_batches - 1 {
                let current_avg_loss = train_loss_sum / (batch_idx + 1) as f32;
                let current_acc = if train_total > 0 {
                    (train_correct as f32 / train_total as f32) * 100.0
                } else {
                    0.0
                };
                println!(
                    "  [Epoch {}/{}] Batch {}/{}: Loss = {:.4}, Acc = {:>5.2}%",
                    epoch + 1,
                    num_epochs,
                    batch_idx + 1,
                    total_batches,
                    current_avg_loss,
                    current_acc
                );

                use tauri::Emitter;
                let _ = app_handle.emit(
                    "evaluating-batch-metrics",
                    BatchMetrics {
                        epoch: epoch + 1,
                        batch: batch_idx + 1,
                        total_batches,
                        loss: current_avg_loss,
                        accuracy: current_acc,
                    },
                );
            }
        }

        final_loss = train_loss_sum / total_batches.max(1) as f32;
        final_acc = if train_total > 0 {
            (train_correct as f32 / train_total as f32) * 100.0
        } else {
            0.0
        };

        if is_classification {
            if epoch > 0 {
                let improvement = final_acc - prev_acc;
                if improvement < 1.0 && final_acc < 75.0 {
                    println!(
                        "  >>> Early stopping triggered: Accuracy stalled at {:.2}% (improvement: {:.2}%).",
                        final_acc, improvement
                    );
                    break;
                }
            }
            prev_acc = final_acc;
        }
    }

    (final_loss, final_acc)
}

/// Inference-only pass on validation/test batches. No gradients, no weight updates.
#[allow(clippy::type_complexity)]
pub fn run_validation_pass<B: AutodiffBackend>(
    model: &GraphModel<B>,
    batches: &[DynamicBatch<B>],
    split_name: &str,
    is_classification: bool,
) -> (f32, f32) {
    let mut val_loss_sum = 0.0f32;
    let mut val_correct = 0usize;
    let mut val_total = 0usize;

    for batch in batches {
        let cloned_inputs: Vec<DynamicTensor<B>> = batch.inputs.iter().map(|t| t.clone()).collect();
        let cloned_targets: Vec<DynamicTensor<B>> =
            batch.targets.iter().map(|t| t.clone()).collect();

        let predictions = model.forward_internal(&cloned_inputs, false);
        let loss = model.compute_loss(&predictions, &cloned_targets);

        val_loss_sum += loss.into_data().to_vec::<f32>().unwrap()[0];

        let (batch_correct, batch_total) =
            compute_accuracy(&predictions, &cloned_targets, is_classification);
        val_correct += batch_correct;
        val_total += batch_total;
    }

    let avg_loss = val_loss_sum / batches.len().max(1) as f32;
    let acc = if val_total > 0 {
        (val_correct as f32 / val_total as f32) * 100.0
    } else {
        0.0
    };

    eprintln!(
        "  [{}] Loss = {:.4}, Acc = {:.2}%",
        split_name, avg_loss, acc
    );

    (avg_loss, acc)
}
