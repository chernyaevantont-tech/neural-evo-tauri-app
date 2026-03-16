# Реализация RNN нод для обработки временных рядов

## ✅ ОТЛИЧНЫЕ НОВОСТИ: Burn 0.20+ УЖЕ ИМ ЕЕТ RNN/LSTM/GRU!

Burn имеет полную поддержку рекуррентных нейронных сетей в модуле `burn::nn`:

- **LSTM** - `Lstm<B>` и `BiLstm<B>` в `burn::nn::lstm`
- **GRU** - `Gru<B>` и `BiGru<B>` в `burn::nn::gru`
- **RNN** - базовый RNN в `burn::nn::basic`

Это означает, что реализация на нашей стороне будет **прямолинейной обёрткой** над существующими модулями Burn.

---

## 1. API Burn RNN модулей

### 1.1 LSTM (`burn::nn::lstm`)

```rust
// Конфигурация + инициализация
let lstm_config = LstmConfig::new(
    d_input: usize,              // Размер входа
    d_hidden: usize,             // Размер скрытого состояния
    bias: bool,                  // Использовать bias
);
let lstm = lstm_config.init::<Backend>(&device);

// Forward pass
let (output, final_state) = lstm.forward(
    input: Tensor<B, 3>,         // [batch, seq_len, input_size]
    initial_state: Option<LstmState<B, 3>>
);

// Параметры LstmConfig
let config = LstmConfig::new(10, 64, true)
    .with_gate_activation(ActivationConfig::Sigmoid)      // default
    .with_cell_activation(ActivationConfig::Tanh)         // default
    .with_hidden_activation(ActivationConfig::Tanh)       // default
    .with_initializer(Initializer::XavierNormal{gain:1.0});

// Bidirectional LSTM
let bilstm_config = BiLstmConfig::new(d_input, d_hidden, bias);
let bilstm = bilstm_config.init(&device);
let (output, state) = bilstm.forward(input, initial_state);

// LstmState: struct with cell и hidden fields
pub struct LstmState<B: Backend, const D: usize> {
    pub cell: Tensor<B, D>,      // [batch, hidden_size]
    pub hidden: Tensor<B, D>,    // [batch, hidden_size]
}
```

### 1.2 GRU (`burn::nn::gru`)

```rust
// Аналогично LSTM, но проще (нет cell state)
let gru_config = GruConfig::new(d_input, d_hidden, bias);
let gru = gru_config.init(&device);

let output = gru.forward(
    input: Tensor<B, 3>,         // [batch, seq_len, input_size]
    initial_state: Option<Tensor<B, 2>>  // [batch, hidden_size]
);

// Параметры GruConfig
let config = GruConfig::new(10, 64, true)
    .with_gate_activation(ActivationConfig::Sigmoid)      // For update/reset gates
    .with_hidden_activation(ActivationConfig::Tanh)       // For new/candidate gate
    .with_reset_after(true)                               // PyTorch arXiv v1 vs v3
    .with_initializer(Initializer::XavierNormal{gain:1.0});

// Bidirectional GRU
let bigru_config = BiGruConfig::new(d_input, d_hidden, bias);
let bigru = bigru_config.init(&device);
let (output, state) = bigru.forward(input, initial_state);
```

### 1.3 RNN (`burn::nn::basic`)

```rust
let rnn_config = RnnConfig::new(d_input, d_hidden, bias);
let rnn = rnn_config.init(&device);

let output = rnn.forward(
    input: Tensor<B, 3>,
    initial_state: Option<Tensor<B, 2>>
);
```

---

## 2. Архитектура ноды на фронтенде

### Интерфейс BaseNode

Все ноды должны реализовать:
```typescript
abstract class BaseNode {
    protected abstract CalculateOutputShape(): void;
    abstract GetInfo(): string;                      // JSON сериализация
    abstract GetResources(dtype: number): ResourceCriteria;
    protected abstract Mutate(mutation_options: Map<string, number>): void;
    public abstract GetExpectedInputDimensions(): number | "any";
    public abstract GetOutputDimensions(): number | "any";
    public abstract GetNodeType(): string;           // Дискриминатор ("LSTM", "GRU", и т.д.)
    protected abstract _CloneImpl(): BaseNode;        // Для кроссовера
    public abstract GetIsMerging(): boolean;         // Merge node flag
}
```

### Сериализация в JSON

Каждая нода → JSON с паттерном `{ "node": "TypeName", "params": {...} }`:

```json
{
    "node": "LSTM",
    "params": {
        "hidden_units": 64,
        "gate_activation": "sigmoid",
        "cell_activation": "tanh",
        "hidden_activation": "tanh",
        "use_bias": true
    }
}
```

---

## 3. Пошаговая реализация LSTM

### 3.1 LSTMNode на фронтенде

**Файл**: `src/entities/canvas-genome/model/nodes/sequential/lstm_node.ts`

```typescript
import { RandomizeInteger } from "../../../../../lib/random";
import { BaseNode, ResourceCriteria } from "../base_node";

export class LSTMNode extends BaseNode {
    private static gateActivationFunctions = ["sigmoid", "tanh"];
    private static cellActivationFunctions = ["tanh", "relu"];
    private static hiddenActivationFunctions = ["tanh", "sigmoid"];

    private hiddenUnits: number;
    private gateActivation: string;        // "sigmoid" (по умолчанию в Burn)
    private cellActivation: string;        // "tanh" (по умолчанию в Burn)
    private hiddenActivation: string;      // "tanh" (по умолчанию в Burn)
    private useBias: boolean;

    /**
     * @param hiddenUnits - Размер скрытого состояния
     * @param gateActivation - Активация для input/forget/output гейтов
     * @param cellActivation - Активация для cell gate
     * @param hiddenActivation - Активация перед выходом
     * @param useBias - Использовать bias
     */
    constructor(
        hiddenUnits: number = 64,
        gateActivation: string = "sigmoid",
        cellActivation: string = "tanh",
        hiddenActivation: string = "tanh",
        useBias: boolean = true
    ) {
        super();
        if (!(Number.isInteger(hiddenUnits) && hiddenUnits > 0)) {
            throw Error("hiddenUnits must be a positive integer");
        }
        this.hiddenUnits = hiddenUnits;
        this.gateActivation = gateActivation;
        this.cellActivation = cellActivation;
        this.hiddenActivation = hiddenActivation;
        this.useBias = useBias;
        this.inputShape = new Array<number>(2); // [sequence_length, features]
    }

    protected CalculateOutputShape(): void {
        // LSTM возвращает output для каждого временного шага
        // Форма: [sequence_length, hidden_units]
        this.outputShape = [this.inputShape[0], this.hiddenUnits];
    }

    public GetExpectedInputDimensions(): number | "any" {
        return 2; // [sequence_length, features]
    }

    public GetOutputDimensions(): number | "any" {
        return 2; // [sequence_length, hidden_units]
    }

    GetInfo(): string {
        return JSON.stringify({
            node: this.GetNodeType(),
            params: {
                hidden_units: this.hiddenUnits,
                gate_activation: this.gateActivation,
                cell_activation: this.cellActivation,
                hidden_activation: this.hiddenActivation,
                use_bias: this.useBias
            }
        });
    }

    GetResources(dtype: number): ResourceCriteria {
        const features = this.inputShape[1];
        const sequence_len = this.inputShape[0];
        
        // LSTM: 4 гейта × (input-to-hidden + hidden-to-hidden)
        const params_per_gate = (features + this.hiddenUnits) * this.hiddenUnits;
        const total_params = 4 * params_per_gate + (this.useBias ? 4 * this.hiddenUnits : 0);
        
        const flash = total_params * dtype;
        const ram = (sequence_len * (features + this.hiddenUnits * 4)) * dtype;
        const macs = sequence_len * 4 * params_per_gate;

        return { flash, ram, macs };
    }

    protected Mutate(mutation_options: Map<string, number>): void {
        if (Math.random() <= (mutation_options.get("lstm_hidden_units") || -1)) {
            this.hiddenUnits = Math.pow(2, RandomizeInteger(4, 10)); // 16 to 1024
        }

        if (Math.random() <= (mutation_options.get("lstm_gate_activation") || -1)) {
            this.gateActivation = LSTMNode.gateActivationFunctions[RandomizeInteger(0, 1)];
        }

        if (Math.random() <= (mutation_options.get("lstm_cell_activation") || -1)) {
            this.cellActivation = LSTMNode.cellActivationFunctions[RandomizeInteger(0, 1)];
        }

        if (Math.random() <= (mutation_options.get("lstm_hidden_activation") || -1)) {
            this.hiddenActivation = LSTMNode.hiddenActivationFunctions[RandomizeInteger(0, 1)];
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
            this.gateActivation,
            this.cellActivation,
            this.hiddenActivation,
            this.useBias
        );

    public GetIsMerging = (): boolean => false;
}
```

### 3.2 DTO в Rust

**Файл**: `src-tauri/src/dtos.rs`

```rust
#[derive(Deserialize, Clone, Debug)]
#[serde(tag = "node", content = "params")]
pub enum NodeDtoJSON {
    // ... существующие ноды ...
    
    LSTM {
        hidden_units: u64,
        #[serde(default = "default_sigmoid")]
        gate_activation: String,
        #[serde(default = "default_tanh")]
        cell_activation: String,
        #[serde(default = "default_tanh")]
        hidden_activation: String,
        #[serde(default = "default_true")]
        use_bias: bool,
    },
    
    GRU {
        hidden_units: u64,
        #[serde(default = "default_sigmoid")]
        gate_activation: String,
        #[serde(default = "default_tanh")]
        hidden_activation: String,
        #[serde(default = "default_true")]
        reset_after: bool,
        #[serde(default = "default_true")]
        use_bias: bool,
    },
}

fn default_sigmoid() -> String { "sigmoid".to_string() }
fn default_tanh() -> String { "tanh".to_string() }
fn default_true() -> bool { true }
```

### 3.3 Компиляция в Burn

**Файл**: `src-tauri/src/entities.rs`

```rust
// Добавить в GraphModel
#[derive(Module, Debug)]
pub struct GraphModel<B: Backend> {
    // ... существующие поля ...
    pub lstms: Vec<Lstm<B>>,      // ✅ Используем готовый Lstm из Burn!
    pub grus: Vec<Gru<B>>,        // ✅ Используем готовый Gru из Burn!
    // ...
}

// В method build():
match &config {
    // ... другие ноды ...
    
    NodeDtoJSON::LSTM {
        hidden_units,
        gate_activation,
        cell_activation,
        hidden_activation,
        use_bias,
    } => {
        let prev_shape = &shape_cache[inputs_for_node[0]];
        let d_input = prev_shape[1];  // [seq_len, features]
        
        // Используем LstmConfig из Burn::nn::lstm
        let lstm = LstmConfig::new(d_input, *hidden_units as usize, *use_bias)
            .with_gate_activation(ActivationConfig::from_string(gate_activation))
            .with_cell_activation(ActivationConfig::from_string(cell_activation))
            .with_hidden_activation(ActivationConfig::from_string(hidden_activation))
            .init(device);
        
        let lstm_idx = lstms.len();
        lstms.push(lstm);
        
        (
            Operation::LSTM { lstm_idx },
            vec![prev_shape[0], *hidden_units as usize],  // [seq_len, hidden_units]
        )
    }
    
    NodeDtoJSON::GRU {
        hidden_units,
        gate_activation,
        hidden_activation,
        reset_after,
        use_bias,
    } => {
        let prev_shape = &shape_cache[inputs_for_node[0]];
        let d_input = prev_shape[1];
        
        let gru = GruConfig::new(d_input, *hidden_units as usize, *use_bias)
            .with_gate_activation(ActivationConfig::from_string(gate_activation))
            .with_hidden_activation(ActivationConfig::from_string(hidden_activation))
            .with_reset_after(*reset_after)
            .init(device);
        
        let gru_idx = grus.len();
        grus.push(gru);
        
        (
            Operation::GRU { gru_idx },
            vec![prev_shape[0], *hidden_units as usize],
        )
    }
}

// Добавить в enum Operation
#[derive(Clone, Debug)]
pub enum Operation {
    // ... существующие ...
    LSTM { lstm_idx: usize },
    GRU { gru_idx: usize },
    // ...
}
```

### 3.4 Forward Pass

**Файл**: `src-tauri/src/entities.rs`

```rust
impl<B: AutodiffBackend> GraphModel<B> {
    pub fn forward(&self, mut tensors: Vec<DynamicTensor<B>>) -> Vec<DynamicTensor<B>> {
        for instr in &self.execution_plan {
            match &instr.op {
                // ... другие операции ...
                
                Operation::LSTM { lstm_idx } => {
                    let input = match &tensors[instr.input_ids[0]] {
                        DynamicTensor::Dim2(x) => x.clone(),
                        _ => panic!("LSTM input must be 2D [seq_len, features]"),
                    };
                    
                    // Reshape для Burn: [seq_len, features] → [1, seq_len, features]
                    let [seq_len, features] = input.dims();
                    let batched = input.reshape([1, seq_len, features]);
                    
                    let (output, _state) = self.lstms[*lstm_idx].forward(batched, None);
                    
                    // Reshape обратно: [1, seq_len, hidden] → [seq_len, hidden]
                    let [_, seq_out, hidden] = output.dims();
                    let squeezed = output.reshape([seq_out, hidden]);
                    
                    tensors[instr.node_id] = DynamicTensor::Dim2(squeezed);
                }
                
                Operation::GRU { gru_idx } => {
                    let input = match &tensors[instr.input_ids[0]] {
                        DynamicTensor::Dim2(x) => x.clone(),
                        _ => panic!("GRU input must be 2D"),
                    };
                    
                    let [seq_len, features] = input.dims();
                    let batched = input.reshape([1, seq_len, features]);
                    
                    let output = self.grus[*gru_idx].forward(batched, None);
                    
                    let [_, seq_out, hidden] = output.dims();
                    let squeezed = output.reshape([seq_out, hidden]);
                    
                    tensors[instr.node_id] = DynamicTensor::Dim2(squeezed);
                }
                
                // ... остальные операции ...
            }
        }
        output
    }
}
```

---

## 4. Аналогично для GRU Node

Процесс полностью аналогичен, но с параметрами `GruConfig`:
- `d_input`, `d_hidden`, `bias`
- `gate_activation` (для update/reset)
- `hidden_activation` (для new/candidate)
- `reset_after` (PyTorch v1 vs v3 несовместимость)

**Важный параметр `reset_after`:**
- `true` (default) - применять reset gate ✕ hidden перед умножением на веса
- `false` - применять reset gate после умножения (как в PyTorch v3)

---

## 5. Тестирование нод

### Unit тесты для LSTMNode

**Файл**: `src/entities/canvas-genome/model/nodes/sequential/lstm_node.test.ts`

```typescript
import { describe, it, expect } from "vitest";
import { LSTMNode } from "./lstm_node";

describe("LSTMNode", () => {
    it("calculates output shape correctly", () => {
        const node = new LSTMNode(64);
        node.SetInputShape([100, 10]); // [seq_len, features]
        
        expect(node.GetOutputShape()).toEqual([100, 64]); // [seq_len, hidden_units]
    });

    it("supports mutation of parameters", () => {
        const node = new LSTMNode(64);
        const original = JSON.parse(node.GetInfo());
        
        const mutationOptions = new Map();
        mutationOptions.set("lstm_hidden_units", 1.0);
        node.Mutate(mutationOptions);
        
        const mutated = JSON.parse(node.GetInfo());
        expect(mutated.params.hidden_units).not.toBe(original.params.hidden_units);
    });

    it("returns 2D dimensions", () => {
        const node = new LSTMNode(64);
        expect(node.GetExpectedInputDimensions()).toBe(2);
        expect(node.GetOutputDimensions()).toBe(2);
    });

    it("clones correctly", () => {
        const original = new LSTMNode(128, "tanh", "tanh", "sigmoid", true);
        const cloned = original.Clone();
        
        expect(cloned.GetNodeType()).toBe("LSTM");
        expect(cloned.GetInfo()).toBe(original.GetInfo());
        expect(cloned.id).not.toBe(original.id);
    });
});
```

---

## 6. Пример сети для временных рядов

```
Input Dataset: [samples, sequence_length=100, features=10]

Network:
┌─────────────────────────────────────────┐
│ Input [100, 10]                         │
│   ↓                                      │
│ LSTM(hidden=64) [100, 64]               │
│   ↓                                      │
│ Dense(128, activation=relu) [128]       │
│   ↓                                      │
│ Dense(10, activation=softmax) [10] ← Out│
└─────────────────────────────────────────┘
```

Как альтернатива:
```
Input [100, 10]
   ↓
GRU(hidden=32) [100, 32]
   ↓
Flatten [3200]
   ↓
Dense(64) → Dense(10, softmax) → Output
```

---

## 7. Тонкости реализации

### Форма входа/выхода

- **Входная форма**: `[sequence_length, features]` (2D)
- **Выходная форма**: `[sequence_length, hidden_units]` (2D)
- **Burn ожидает** 3D с batch: `[batch, seq_len, input_size]`
  - Решение: reshape входа перед forward, reshape выхода после

### Начальное состояние

- **LSTM**: `LstmState<B, 3>` с `cell` и `hidden`
- **GRU**: просто `Tensor<B, 2>`
- Если не передано, инициализируется нулями в Burn

### Совместимость слоёв

✅ **Работает:**
- LSTM → Dense (2D → 1D flatten)
- LSTM → LSTM (2D → 2D)
- GRU → Conv1D (если реализуем)
- GRU → Flatten → Dense

❌ **НЕ работает:**
- LSTM → Conv2D (размерность мismatch)
- GRU → Pooling (если Pooling ожидает 3D)

---

## 8. Регистрация в UI

**Местонахождение**: `src/entities/canvas-genome/model/nodes/nodeFactory.ts`

```typescript
import { LSTMNode } from "./sequential/lstm_node";
import { GRUNode } from "./sequential/gru_node";

export function createNodeByType(type: string, params?: any): BaseNode {
    switch (type) {
        // ... существующие ноды ...
        case "LSTM":
            return new LSTMNode(
                params?.hidden_units || 64,
                params?.gate_activation || "sigmoid",
                params?.cell_activation || "tanh",
                params?.hidden_activation || "tanh",
                params?.use_bias ?? true
            );
        case "GRU":
            return new GRUNode(
                params?.hidden_units || 64,
                params?.gate_activation || "sigmoid",
                params?.hidden_activation || "tanh",
                params?.reset_after ?? true,
                params?.use_bias ?? true
            );
        default:
            throw Error(`Unknown node type: ${type}`);
    }
}
```

---

## 📝 Резюме

✅ **Burn предоставляет**:
- Готовые `Lstm`, `BiLstm`, `Gru`, `BiGru` модули
- Полнофункциональное управление состоянием
- Оптимизированные реализации для разных бэкендов

✅ **Нужно реализовать**:
1. LSTMNode TypeScript class (следуя паттерну BaseNode)
2. GRUNode TypeScript class (аналогично)
3. Добавить NodeDtoJSON::LSTM и NodeDtoJSON::GRU варианты
4. Реализовать build() logic соответственно
5. Реализовать forward() logic
6. Тесты

✅ **Это относительно прямолинейно** потому что мы просто **обёртываем готовые модули Burn**.

**Начните с:** LSTMNode и compile логики в Burn, если всё работает → перейдите на GRU.
