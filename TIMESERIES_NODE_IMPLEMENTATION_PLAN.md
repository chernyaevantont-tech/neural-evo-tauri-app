# Реализация нод для обработки временных рядов

## 1. Анализ текущей архитектуры

### 1.1 Полиморфизм на фронтенде

#### Иерархия классов
```
BaseNode (абстрактный класс)
├── Input/Output (специальные ноды)
├── Слои обработки:
│   ├── DenseNode (полносвязная сеть)
│   ├── Conv2DNode (свёртка 2D)
│   ├── FlattenNode (развёртывание)
│   ├── PoolingNode (pooling)
│   └── [другие слои]
├── Нормализация:
│   ├── BatchNormNode
│   ├── LayerNormNode
│   └── [другие]
└── Слияние:
    ├── AddNode
    └── ConcatNode
```

#### Интерфейс BaseNode (обязательные методы)
```typescript
abstract class BaseNode {
    // Абстрактные методы, которые ДОЛЖНЫ быть реализованы
    protected abstract CalculateOutputShape(): void;
    abstract GetInfo(): string;
    abstract GetResources(dtype: number): ResourceCriteria;
    protected abstract Mutate(mutation_options: Map<string, number>): void;
    public abstract GetExpectedInputDimensions(): number | "any";
    public abstract GetOutputDimensions(): number | "any";
    public abstract GetNodeType(): string;
    protected abstract _CloneImpl(): BaseNode;
    public abstract GetIsMerging(): boolean;

    // Общие методы для всех нод
    public CanAcceptConnectionFrom(node: BaseNode): boolean { ... }
    public PropagateShapeUpdate(): void { ... }
    public AddNext(node: BaseNode): void { ... }
}
```

#### Как работает сериализация
```typescript
// Нода преобразуется в JSON:
{
    "node": "LayerType",  // Дискриминатор типа (результат GetNodeType())
    "params": {           // Все параметры конкретной ноды
        "param1": value1,
        "param2": value2,
        ...
    }
}

// Пример для DenseNode:
{
    "node": "Dense",
    "params": {
        "units": 256,
        "activation": "relu",
        "use_bias": true
    }
}
```

---

### 1.2 Полиморфизм на бекенде (Burn)

#### DTO десериализация
```rust
// dtos.rs - Rust enum с дискриминатором
#[derive(Deserialize, Clone, Debug)]
#[serde(tag = "node", content = "params")]  // "node" = key, "params" = value
pub enum NodeDtoJSON {
    Dense { units: u64, activation: String, use_bias: bool },
    Conv2D { filters: u64, kernel_size: KernelSizeDto, ... },
    Pooling { pool_type: String, kernel_size: KernelSizeDto, ... },
    // ... остальные ноды
    // НУЖНО ДОБАВИТЬ:
    // LSTM { ... },
    // GRU { ... },
    // RNN { ... },
}
```

#### Компиляция нод в Burn модули
```rust
// entities.rs::build() - преобразование JSON в Burn операции

match &config {
    NodeDtoJSON::Dense { units, activation, use_bias } => {
        // Создаём Linear слой (эквивалент Dense в Burn)
        let linear = LinearConfig::new(input_dim, *units)
            .with_bias(*use_bias)
            .init(device);
        denses.push(linear);
        (Operation::Dense { dense_idx, activation }, output_shape)
    }
    NodeDtoJSON::Conv2D { filters, kernel_size, ... } => {
        // Создаём Conv2d слой
        let conv = Conv2dConfig::new([in_channels, actual_filters], [k_h, k_w])
            .init(device);
        conv2ds.push(conv);
        (Operation::Conv2D { conv2d_idx, activation }, output_shape)
    }
    // ... остальные ноды
    // НУЖНО ДОБАВИТЬ:
    // NodeDtoJSON::LSTM { ... } => { ... }
    // NodeDtoJSON::GRU { ... } => { ... }
    // NodeDtoJSON::RNN { ... } => { ... }
}
```

#### Выполнение операций в forward pass
```rust
// entities.rs::forward() - применение операций

impl<B: AutodiffBackend> GraphModel<B> {
    pub fn forward(&self, mut tensors: Vec<DynamicTensor<B>>) -> Vec<DynamicTensor<B>> {
        for instr in &self.execution_plan {
            match &instr.op {
                Operation::Dense { dense_idx, activation } => {
                    let x = &tensors[instr.input_ids[0]];
                    // Применяем Linear слой + активация
                    let out = self.denses[*dense_idx].forward(x.to_2d());
                    tensors[instr.node_id] = self.apply_activation(&out, activation);
                }
                Operation::Conv2D { conv2d_idx, activation } => {
                    let x = &tensors[instr.input_ids[0]];
                    let out = self.conv2ds[*conv2d_idx].forward(x.to_4d());
                    tensors[instr.node_id] = self.apply_activation(&out, activation);
                }
                // ... остальные операции
                // НУЖНО ДОБАВИТЬ:
                // Operation::LSTM { ... } => { ... }
                // Operation::GRU { ... } => { ... }
                // Operation::RNN { ... } => { ... }
            }
        }
        output
    }
}
```

---

## 2. Ноды для обработки временных рядов

### 2.1 Какие ноды нужны

#### RNN (Recurrent Neural Network)
- **Назначение**: Базовая рекуррентная архитектура для последовательностей
- **Входная размерность**: 2D `[sequence_length, features]` или 3D `[batch_size, sequence_length, features]`
- **Выходная размерность**: 2D `[hidden_units]` (последний временной шаг) или 3D `[batch_size, sequence_length, hidden_units]` (все временные шаги)
- **Параметры**:
  - `hidden_units` (степени двойки: 16, 32, 64, 128, ...)
  - `return_sequences` (bool) - выход для каждого временного шага или только последний
  - `activation` ("tanh", "relu", "sigmoid")

#### LSTM (Long Short-Term Memory)
- **Назначение**: Обучается долгим зависимостям, предотвращает vanishing gradients
- **Входная размерность**: 2D или 3D последовательность
- **Выходная размерность**: Как RNN
- **Параметры**:
  - `hidden_units`
  - `return_sequences`
  - `activation` ("tanh" по умолчанию)
  - `recurrent_activation` ("sigmoid" для гейтов)

#### GRU (Gated Recurrent Unit)
- **Назначение**: Упрощённая LSTM, меньше параметров, часто такая же точность
- **Параметры**: Аналогично LSTM, но без отдельной cell state

#### Attention Layer (опционально)
- **Назначение**: Взвешивает важность разных временных шагов
- **Входная размерность**: Последовательность
- **Выходная размерность**: Взвешенная последовательность

#### 1D Convolution (временная свёртка)
- **Назначение**: Локальные паттерны во времени (может быть быстрее RNN на длинных последовательностях)
- **Входная размерность**: 2D `[sequence_length, features]`
- **Выходная размерность**: 2D `[output_length, filters]`

### 2.2 Приоритет реализации

**Фаза 1 (ГОТОВО В BURN):**
1. **LSTM** ✅ - `Lstm<B>` и `BiLstm<B>` в `burn::nn::lstm`
2. **GRU** ✅ - `Gru<B>` и `BiGru<B>` в `burn::nn::gru`
3. **RNN** ✅ - базовый RNN в `burn::nn::basic`

**Фаза 2 (реализация на фронтенде):**
1. **LSTMNode** - фронтенд обёртка над `Lstm`
2. **GRUNode** - фронтенд обёртка над `Gru`
3. **RNNNode** - базовый RNN (опционально)

**Фаза 3 (полезные):**
4. **Conv1D** - мощный для локальных паттернов
5. **BiLSTMNode** / **BiGRUNode** - двусторонние варианты

**Фаза 4 (продвинутые):**
6. **Attention** - улучшает интерпретируемость
7. **Transformer layers** - самое современное

---

## 3. Burn RNN módules - это УЖЕ реализовано!

### 3.1 Имеющиеся модули в Burn

Burn 0.20+ **уже имеет полную поддержку RNN**:

#### LSTM (`burn::nn::lstm`)
```rust
// Конфигурация
let lstm_config = LstmConfig::new(d_input: 10, d_hidden: 64, bias: true);
let lstm = lstm_config.init::<Backend>(&device);

// Forward pass возвращает (output, state)
let (output, state) = lstm.forward(
    input: Tensor<B, 3>,  // [batch, seq_len, input_size]
    initial_state: Option<LstmState<B, 3>>
);

// BiLstm для двусторонней обработки
let bilstm_config = BiLstmConfig::new(d_input, d_hidden, bias);
let bilstm = bilstm_config.init::<Backend>(&device);
let (output, state) = bilstm.forward(input, initial_state);
```

**Параметры LstmConfig:**
- `d_input: usize` - размер входа
- `d_hidden: usize` - размер скрытого состояния
- `bias: bool` - использовать bias
- `gate_activation: ActivationConfig` - активация для гейтов (по умолчанию Sigmoid)
- `cell_activation: ActivationConfig` - активация для cell gate (по умолчанию Tanh)
- `hidden_activation: ActivationConfig` - активация для вывода (по умолчанию Tanh)
- `initializer: Initializer` - инициализация весов

#### GRU (`burn::nn::gru`)
```rust
// Аналогично LSTM, но проще
let gru_config = GruConfig::new(d_input, d_hidden, bias);
let gru = gru_config.init::<Backend>(&device);

let output = gru.forward(
    input: Tensor<B, 3>,  // [batch, seq_len, input_size]
    initial_state: Option<Tensor<B, 2>>  // [batch, hidden_size]
);

// BiGru
let bigru_config = BiGruConfig::new(d_input, d_hidden, bias);
let bigru = bigru_config.init(&device);
let (output, state) = bigru.forward(input, initial_state);
```

**Параметры GruConfig:**
- `d_input`, `d_hidden`, `bias` - как в LSTM
- `reset_after: bool` - когда применять reset gate (различия с PyTorch версиями)
- `gate_activation`, `hidden_activation` - активации

#### RNN (`burn::nn::basic`)
```rust
let rnn_config = RnnConfig::new(d_input, d_hidden, bias);
let rnn = rnn_config.init::<Backend>(&device);

let output = rnn.forward(input, initial_state);
```

### 3.2 Формат состояния

**LSTM состояние:**
```rust
pub struct LstmState<B: Backend, const D: usize> {
    pub cell: Tensor<B, D>,     // Cell state [batch, hidden_size]
    pub hidden: Tensor<B, D>,   // Hidden state [batch, hidden_size]
}
```

**GRU состояние:**
- Просто `Tensor<B, 2>` с формой `[batch, hidden_size]`

---

## 4. Пошаговый план реализации

### 4.1 Фаза 1: LSTM Node (фронтенд)

**Файл**: `src/entities/canvas-genome/model/nodes/sequential/lstm_node.ts`

```typescript
import { RandomizeInteger } from "../../../../../lib/random";
import { BaseNode, ResourceCriteria } from "../base_node";
import { ActivationFunction } from "../types";

export class LSTMNode extends BaseNode {
    private static activationFunctions: ActivationFunction[] = ["tanh", "relu", "sigmoid"];
    private static recurrentActivations = ["sigmoid", "tanh"];

    private hiddenUnits: number;
    private activation: ActivationFunction;
    private recurrentActivation: string;
    private returnSequences: boolean;
    private useBias: boolean;

    constructor(
        hiddenUnits: number,
        activation: ActivationFunction = "tanh",
        recurrentActivation: string = "sigmoid",
        returnSequences: boolean = false,
        useBias: boolean = true
    ) {
        super();
        if (!(Number.isInteger(hiddenUnits) && hiddenUnits > 0)) {
            throw Error("hiddenUnits must be a positive integer");
        }
        this.hiddenUnits = hiddenUnits;
        this.activation = activation;
        this.recurrentActivation = recurrentActivation;
        this.returnSequences = returnSequences;
        this.useBias = useBias;
        this.inputShape = new Array<number>(2); // [sequence_length, features]
    }

    protected CalculateOutputShape(): void {
        // Если returnSequences=true: [sequence_length, hidden_units]
        // Если returnSequences=false: [hidden_units]
        if (this.returnSequences) {
            this.outputShape = [this.inputShape[0], this.hiddenUnits];
        } else {
            this.outputShape = [this.hiddenUnits];
        }
    }

    public GetExpectedInputDimensions(): number | "any" {
        return 2; // [sequence_length, features]
    }

    public GetOutputDimensions(): number | "any" {
        return this.returnSequences ? 2 : 1;
    }

    GetInfo(): string {
        return JSON.stringify({
            node: this.GetNodeType(),
            params: {
                hidden_units: this.hiddenUnits,
                activation: this.activation,
                recurrent_activation: this.recurrentActivation,
                return_sequences: this.returnSequences,
                use_bias: this.useBias
            }
        });
    }

    GetResources(dtype: number): ResourceCriteria {
        // LSTM имеет 4 гейта (input, forget, cell, output)
        const features = this.inputShape[1];
        const sequence_len = this.inputShape[0];
        
        // Параметры: input-to-hidden (для каждого гейта) + hidden-to-hidden + bias
        const params_per_gate = (features + this.hiddenUnits) * this.hiddenUnits;
        const total_params = 4 * params_per_gate + (this.useBias ? 4 * this.hiddenUnits : 0);
        
        const flash = total_params * dtype;
        const ram = (sequence_len * (features + this.hiddenUnits * 4)) * dtype;
        const macs = sequence_len * params_per_gate * 4;

        return { flash, ram, macs };
    }

    protected Mutate(mutation_options: Map<string, number>): void {
        if (Math.random() <= (mutation_options.get("lstm_hidden_units") || -1)) {
            this.hiddenUnits = Math.pow(2, RandomizeInteger(4, 10)); // 16 to 1024
        }

        if (Math.random() <= (mutation_options.get("lstm_activation") || -1)) {
            this.activation = LSTMNode.activationFunctions[RandomizeInteger(0, 2)];
        }

        if (Math.random() <= (mutation_options.get("lstm_recurrent_activation") || -1)) {
            this.recurrentActivation = LSTMNode.recurrentActivations[RandomizeInteger(0, 1)];
        }

        if (Math.random() <= (mutation_options.get("lstm_return_sequences") || -1)) {
            this.returnSequences = !this.returnSequences;
        }

        if (Math.random() <= (mutation_options.get("lstm_use_bias") || -1)) {
            this.useBias = !this.useBias;
        }

        this.CalculateOutputShape();
    }

    public GetNodeType = (): string => "LSTM";

    protected _CloneImpl = (): BaseNode =>
        new LSTMNode(
            this.hiddenUnits,
            this.activation,
            this.recurrentActivation,
            this.returnSequences,
            this.useBias
        );

    public GetIsMerging = (): boolean => false;
}
```

---

### 3.2 Фаза 1: Регистрация LSTMNode на бекенде

**Файл**: `src-tauri/src/dtos.rs`

Добавить в enum `NodeDtoJSON`:
```rust
#[derive(Deserialize, Clone, Debug)]
#[serde(tag = "node", content = "params")]
pub enum NodeDtoJSON {
    // ... существующие ноды ...
    
    /// LSTM Layer - Long Short-Term Memory for sequential data
    LSTM {
        hidden_units: u64,
        #[serde(default = "default_tanh")]
        activation: String,
        #[serde(default = "default_sigmoid")]
        recurrent_activation: String,
        #[serde(default)]
        return_sequences: bool,
        #[serde(default = "default_true")]
        use_bias: bool,
    },
}

fn default_tanh() -> String {
    "tanh".to_string()
}

fn default_sigmoid() -> String {
    "sigmoid".to_string()
}

fn default_true() -> bool {
    true
}
```

---

### 3.3 Фаза 1: Компиляция LSTM в Burn

**Файл**: `src-tauri/src/entities.rs`

1. Создать LSTM модуль Burn (или использовать готовый):
```rust
// Добавить в GraphModel
#[derive(Module, Debug)]
pub struct GraphModel<B: Backend> {
    // ... существующие слои ...
    pub lstms: Vec<LSTMModule<B>>,  // НЕ СУЩЕСТВУЕТ В BURN СТАНДАРТНО!
    pub grus: Vec<GRUModule<B>>,    // НЕ СУЩЕСТВУЕТ В BURN СТАНДАРТНО!
    // ...
}
```

**ВАЖНО**: Burn 0.20 НЕ имеет встроённых LSTM/GRU слоёв! Нужно реализовать их вручную или использовать другой подход.

---

## 4. АЛЬТЕРНАТИВНЫЙ ПОДХОД: Использовать Dense слои для Временных рядов

Если реализация полноценных RNN сложна из-за ограничений Burn, можно:

### 4.1 Reshape + Dense обработка
```
Input: [sequence_length, features]  →  Flatten  →  [sequence_length * features]
                                        ↓
                                      Dense(units)  →  [units]
                                        ↓
                                      Reshape  →  [seq_len, units]  (опционально)
```

### 4.2 Conv1D вместо RNN
```
Input: [sequence_length, features]  →  Conv1D(filters=64, kernel=3)  →  [sequence_length-2, 64]
                                        ↓
                                      Pooling1D  →  [reduced_seq_len, 64]
```

Это сохранит временную информацию, но без явных рекуррентных связей.

---

## 5. Шаги реализации (практический план)

### 5.1 Если Burn поддерживает RNN/LSTM:
```
1. Добавить LSTMNode.ts на фронтенде (следуя паттерну DenseNode)
   ✓ Реализовать все обязательные методы
   ✓ Добавить тесты в lstm_node.test.ts

2. Добавить NodeDtoJSON::LSTM в dtos.rs
   ✓ Правильная десериализация JSON параметров
   ✓ Параметры с defaults

3. В entities.rs::build():
   ✓ Добавить case для NodeDtoJSON::LSTM
   ✓ Создать LSTM слой Burn
   ✓ Добавить в execution_plan Operation::LSTM

4. В entities.rs::forward():
   ✓ Реализовать forward pass для LSTM
   ✓ Управление hidden state между временными шагами

5. Регистрация в UI:
   ✓ Добавить LSTMNode в фабрику нод
   ✓ UI контрол для параметров (hidden_units, activation, return_sequences)
   ✓ Иконка в палитре нод
```

### 5.2 Если Burn НЕ поддерживает:
```
Реализовать LSTM вручную в Burn используя:
- Linear слои для преобразований
- Элементарные операции: mul, add, sigmoid, tanh
- Управление состоянием вручную

ИЛИ

Использовать Conv1D как альтернативу для обработки последо
