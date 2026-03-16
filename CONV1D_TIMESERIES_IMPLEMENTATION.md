# Conv1D для обработки временных рядов — Практическое руководство

## Предпосылка

Burn 0.20 **не имеет встроённых слоёв RNN/LSTM/GRU**. Однако **Conv1D (одномерная свёртка)** — отличная альтернатива для обработки временных рядов, т.к.:

1. ✅ **Легко реализуется** через существующий Conv2D + reshape
2. ✅ **Быстро** выполняется (параллелизуется по temporal dimension)
3. ✅ **Эффективна** для локальных паттернов во времени
4. ✅ **Поддерживается** стандартной Conv2d в Burn

### Пример: Временной ряд + Conv1D
```
Input:    [sequence_length=100, features=10]
                    ↓
           Reshape → [1, 100, 10]  (batch=1, height=100, width=10)
                    ↓
        Conv1D (kernel=3, filters=32)  →  [1, 98, 32]
                    ↓
             Pooling1D (pool=2)  →  [1, 49, 32]
                    ↓
          Flatten + Dense → [output_classes]
```

---

## Реализация Conv1D Node на фронтенде

### Шаг 1: Создать Conv1DNode.ts

**Файл**: `src/entities/canvas-genome/model/nodes/sequential/conv1d_node.ts`

```typescript
import { RandomizeInteger } from "../../../../../lib/random";
import { BaseNode, ResourceCriteria } from "../base_node";

export class Conv1DNode extends BaseNode {
    private static dilationOptions = [1, 2];
    private static activationFunctions = ["relu", "tanh", "sigmoid", "linear"];

    private filters: number;              // Количество фильтров
    private kernelSize: number;           // Размер ядра свёртки (нечётное число)
    private stride: number;               // Шаг свёртки
    private padding: number;              // Padding
    private dilation: number;             // Dilation
    private useBias: boolean;
    private activation: string;

    /**
     * @param filters - Количество фильтров свёртки (обычно 16, 32, 64, etc)
     * @param kernelSize - Размер ядра (tipически 3, 5, 7)
     * @param stride - Шаг итерации (обычно 1 или 2)
     * @param padding - Padding (0 или 1 обычно)
     * @param dilation - Dilation для больших receptive fields
     * @param useBias - Использовать bias
     * @param activation - Activation function
     */
    constructor(
        filters: number = 32,
        kernelSize: number = 3,
        stride: number = 1,
        padding: number = 1,
        dilation: number = 1,
        useBias: boolean = true,
        activation: string = "relu"
    ) {
        super();

        if (!(Number.isInteger(filters) && filters > 0)) {
            throw Error("filters must be a positive integer");
        }
        if (!(Number.isInteger(kernelSize) && kernelSize > 0 && kernelSize % 2 === 1)) {
            throw Error("kernelSize must be odd positive integer (3, 5, 7, etc)");
        }

        this.filters = filters;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        this.dilation = dilation;
        this.useBias = useBias;
        this.activation = activation;

        // Входная форма: [sequence_length, features]
        this.inputShape = new Array<number>(2);
    }

    protected CalculateOutputShape(): void {
        const seqLen = this.inputShape[0];
        const features = this.inputShape[1];

        // Conv1D output length formula:
        // output_length = floor((input_length + 2*padding - dilation*(kernel-1) - 1) / stride + 1)
        const outputLen = Math.floor(
            (seqLen + 2 * this.padding - this.dilation * (this.kernelSize - 1) - 1) / this.stride + 1
        );

        // Выходная форма: [output_length, filters]
        this.outputShape = [outputLen, this.filters];
    }

    public GetExpectedInputDimensions(): number | "any" {
        return 2; // [sequence_length, features]
    }

    public GetOutputDimensions(): number | "any" {
        return 2; // [new_sequence_length, filters]
    }

    GetInfo(): string {
        return JSON.stringify({
            node: this.GetNodeType(),
            params: {
                filters: this.filters,
                kernel_size: this.kernelSize,
                stride: this.stride,
                padding: this.padding,
                dilation: this.dilation,
                use_bias: this.useBias,
                activation: this.activation
            }
        });
    }

    GetResources(dtype: number): ResourceCriteria {
        const seqLen = this.inputShape[0];
        const features = this.inputShape[1];
        const outLen = this.outputShape[0];

        // Параметры свёртки: kernel_size * features * filters + bias
        const params = this.kernelSize * features * this.filters + (this.useBias ? this.filters : 0);

        const flash = params * dtype;
        const ram = (seqLen * features + outLen * this.filters) * dtype;
        const macs = outLen * this.kernelSize * features * this.filters;

        return { flash, ram, macs };
    }

    protected Mutate(mutation_options: Map<string, number>): void {
        // Мутация: изменить количество фильтров
        if (Math.random() <= (mutation_options.get("conv1d_filters") || -1)) {
            this.filters = 4 * RandomizeInteger(4, 16); // [16, 20, 24, ..., 64]
        }

        // Мутация: изменить размер ядра
        if (Math.random() <= (mutation_options.get("conv1d_kernel_size") || -1)) {
            const kernelOptions = [3, 5, 7, 9];
            this.kernelSize = kernelOptions[RandomizeInteger(0, 3)];
        }

        // Мутация: изменить stride
        if (Math.random() <= (mutation_options.get("conv1d_stride") || -1)) {
            this.stride = RandomizeInteger(1, 2);
        }

        // Мутация: изменить padding
        if (Math.random() <= (mutation_options.get("conv1d_padding") || -1)) {
            this.padding = RandomizeInteger(0, 2);
        }

        // Мутация: изменить dilation
        if (Math.random() <= (mutation_options.get("conv1d_dilation") || -1)) {
            this.dilation = Conv1DNode.dilationOptions[RandomizeInteger(0, 1)];
        }

        // Мутация: изменить activation
        if (Math.random() <= (mutation_options.get("conv1d_activation") || -1)) {
            this.activation = Conv1DNode.activationFunctions[
                RandomizeInteger(0, Conv1DNode.activationFunctions.length - 1)
            ];
        }

        // Мутация: toggle bias
        if (Math.random() <= (mutation_options.get("conv1d_use_bias") || -1)) {
            this.useBias = !this.useBias;
        }

        this.CalculateOutputShape();
    }

    public GetNodeType = (): string => "Conv1D";

    protected _CloneImpl = (): BaseNode =>
        new Conv1DNode(
            this.filters,
            this.kernelSize,
            this.stride,
            this.padding,
            this.dilation,
            this.useBias,
            this.activation
        );

    public GetIsMerging = (): boolean => false;
}
```

---

## Регистрация Conv1D на бекенде

### Шаг 2: Добавить в dtos.rs

**Файл**: `src-tauri/src/dtos.rs`

```rust
#[derive(Deserialize, Clone, Debug)]
#[serde(tag = "node", content = "params")]
pub enum NodeDtoJSON {
    // ... существующие ноды ...
    
    /// 1D Convolution for temporal/sequential data
    Conv1D {
        filters: u64,
        kernel_size: u8,      // Нечётное число: 3, 5, 7, etc
        stride: u8,
        padding: u8,
        dilation: u8,
        use_bias: bool,
        #[serde(default = "default_relu")]
        activation: String,
    },
    
    // ... остальные ноды ...
}
```

---

### Шаг 3: Компиляция Conv1D → Burn Conv2D

**Файл**: `src-tauri/src/entities.rs`

Conv1D реализуется как Conv2D с `height=1`:

```rust
NodeDtoJSON::Conv1D {
    filters,
    kernel_size,
    stride,
    padding,
    dilation,
    use_bias,
    activation,
} => {
    let prev_shape = &shape_cache[inputs_for_node[0]];
    
    // Входная форма: [seq_len, features]
    let seq_len = prev_shape[0];
    let in_features = prev_shape[1];
    
    // Выходной размер последовательности:
    // output_len = floor((seq_len + 2*padding - dilation*(kernel-1) - 1) / stride + 1)
    let output_len = (seq_len as usize + 2 * (*padding as usize)
        - (*dilation as usize) * (*kernel_size as usize - 1) - 1)
        / (*stride as usize)
        + 1;
    
    // Создаём Conv2D с height=1 (т.е. свёртка только по временной оси)
    let conv = Conv2dConfig::new([in_features as usize, *filters as usize], [1, *kernel_size as usize])
        .with_stride([1, *stride as usize])
        .with_padding(PaddingConfig2d::Explicit(0, *padding as usize))  // padding only on width
        .with_dilation([1, *dilation as usize])
        .with_bias(*use_bias)
        .init(device);
    
    let conv1d_idx = conv2ds.len();
    conv2ds.push(conv);
    
    (
        Operation::Conv1D {
            conv1d_idx,
            activation: activation.clone(),
        },
        vec![output_len, *filters as usize],  // [output_seq_len, filters]
    )
}
```

---

### Шаг 4: Forward pass для Conv1D

**Файл**: `src-tauri/src/entities.rs` (в методе `forward()`)

```rust
// В GraphModel::forward()
pub fn forward(&self, mut tensors: Vec<DynamicTensor<B>>) -> Vec<DynamicTensor<B>> {
    for instr in &self.execution_plan {
        match &instr.op {
            // ... остальные операции ...
            
            Operation::Conv1D {
                conv1d_idx,
                activation,
            } => {
                let input_2d = match &tensors[instr.input_ids[0]] {
                    DynamicTensor::Dim2(x) => x.clone(),
                    _ => panic!("Conv1D input must be 2D [seq_len, features]"),
                };

                // Преобразуем [seq_len, features] → [1, seq_len, 1, features]
                //                                      [batch, height, width, channels]
                let [seq_len, features] = input_2d.dims();
                let reshaped = input_2d
                    .reshape([1, 1, seq_len, features])  // [1, 1, seq_len, features]
                    .permute([0, 3, 1, 2]);               // [1, features, 1, seq_len]

                // Применяем Conv2D (которая работает как Conv1D благодаря height=1)
                let output_4d = self.conv2ds[*conv1d_idx].forward(reshaped);

                // Преобразуем обратно [1, filters, 1, output_len] → [output_len, filters]
                let [_, filters, _, output_len] = output_4d.dims();
                let output_2d = output_4d
                    .squeeze::<3>(2)                  // [1, filters, output_len]
                    .permute([2, 1])                  // [output_len, filters]
                    .squeeze::<1>(0);                 // [output_len, filters]

                // Применить activation
                let activated = self.apply_activation_2d(&output_2d, activation);
                tensors[instr.node_id] = DynamicTensor::Dim2(activated);
            }

            // ... остальные операции ...
        }
    }
    output
}
```

---

## Интеграция в UI (фронтенд)

### Шаг 5: Регистрация Conv1D в фабрике нод

**Файл**: `src/entities/canvas-genome/model/genome.ts` (или в фабрике нод)

```typescript
// nodeFactory.ts или подобный файл
import { Conv1DNode } from "./nodes/sequential/conv1d_node";

export function createNodeByType(type: string, params?: any): BaseNode {
    switch (type) {
        case "Dense":
            return new DenseNode(params.units || 256, params.activation || "relu", params.use_bias ?? true);
        case "Conv2D":
            return new Conv2DNode(...);
        case "Conv1D":
            return new Conv1DNode(
                params.filters || 32,
                params.kernel_size || 3,
                params.stride || 1,
                params.padding || 1,
                params.dilation || 1,
                params.use_bias ?? true,
                params.activation || "relu"
            );
        // ... остальные ноды ...
        default:
            throw Error(`Unknown node type: ${type}`);
    }
}
```

---

## Тестирование

### Шаг 6: Тесты для Conv1D

**Файл**: `src/entities/canvas-genome/model/nodes/sequential/conv1d_node.test.ts`

```typescript
import { describe, it, expect } from "vitest";
import { Conv1DNode } from "./conv1d_node";

describe("Conv1DNode", () => {
    it("calculates output shape correctly", () => {
        const node = new Conv1DNode(32, 3, 1, 1, 1);
        node.SetInputShape([100, 10]); // [seq_len=100, features=10]
        
        const expected = [100, 32]; // [seq_len=100 (unchanged with stride=1, kernel=3, padding=1), filters=32]
        expect(node.GetOutputShape()).toEqual(expected);
    });

    it("calculates output shape with stride=2", () => {
        const node = new Conv1DNode(64, 5, 2, 2, 1);
        node.SetInputShape([200, 20]);
        
        // output_len = floor((200 + 2*2 - 1*(5-1) - 1) / 2 + 1)
        //            = floor((204 - 4 - 1) / 2 + 1)
        //            = floor(99.5 + 1) = 100
        const expected = [100, 64];
        expect(node.GetOutputShape()).toEqual(expected);
    });

    it("supports mutation of parameters", () => {
        const node = new Conv1DNode(32, 3, 1, 1);
        const original = JSON.parse(node.GetInfo());
        
        const mutationOptions = new Map();
        mutationOptions.set("conv1d_filters", 1.0); // 100% probability
        node.Mutate(mutationOptions);
        
        const mutated = JSON.parse(node.GetInfo());
        expect(mutated.params.filters).not.toBe(original.params.filters);
    });

    it("returns 2D expected and output dimensions", () => {
        const node = new Conv1DNode(32, 3, 1, 1);
        expect(node.GetExpectedInputDimensions()).toBe(2);
        expect(node.GetOutputDimensions()).toBe(2);
    });

    it("clones correctly", () => {
        const original = new Conv1DNode(64, 5, 1, 2);
        const cloned = original.Clone();
        
        expect(cloned.GetNodeType()).toBe("Conv1D");
        expect(cloned.GetInfo()).toBe(original.GetInfo());
        expect(cloned.id).not.toBe(original.id); // Different UUID
    });
});
```

---

## Пример использования в приложении

### Сценарий: Классификация временного ряда

```
Input Dataset: [samples, sequence_length=100, features=10]

Network Architecture:
┌─────────────────────────────────────────────────┐
│ Input [100, 10]                                 │
│   ↓                                              │
│ Conv1D(filters=32, kernel=5, stride=1) [100, 32]│
│   ↓                                              │
│ Pooling1D(kernel=2, stride=2) [50, 32]          │
│   ↓                                              │
│ Conv1D(filters=64, kernel=3, stride=1) [50, 64] │
│   ↓                                              │
│ Flatten [50*64=3200]                            │
│   ↓                                              │
│ Dense(256, activation=relu) [256]               │
│   ↓                                              │
│ Dense(10, activation=softmax) [10]  ← Classes   │
└─────────────────────────────────────────────────┘
```

---

## Полиморфизм в действии

### Как это работает вместе:

1. **Пользователь рисует граф** в UI, добавляя Conv1DNode
2. **Фронтенд сохраняет** JSON:
   ```json
   {
       "node": "Conv1D",
       "params": {
           "filters": 32,
           "kernel_size": 5,
           "stride": 1,
           "padding": 2,
           "dilation": 1,
           "use_bias": true,
           "activation": "relu"
       }
   }
   ```

3. **Бекенд десериализует** Json в `NodeDtoJSON::Conv1D{...}` enum variant

4. **Компилирует** в Burn операцию:
   ```rust
   Operation::Conv1D {
       conv1d_idx: 0,
       activation: "relu".to_string(),
   }
   ```

5. **Выполняет** в forward pass, преобразуя 2D tensor в 4D, применяя Conv2D, затем обратно в 2D

6. **Поддерживает** мутации, кроссовер, эволюцию как обычные ноды ✅

---

## Возможные расширения

### Conv1D + Pooling1D (комбо)
```typescript
export class Pooling1DNode extends BaseNode { ... }
// Для downsampling временных рядов
```

### Bidirectional Conv1D
```typescript
export class BidirectionalConv1DNode extends BaseNode { ... }
// Обрабатывает последовательность в обе стороны
```

### Attention Layer
```typescript
export class AttentionNode extends BaseNode { ... }
// Взвешивает важность разных временных шагов
```

---

## Резюме

✅ **Conv1D** — это практичный способ добавить обработку временных рядов без реализации RNN  
✅ Использует **существующую Conv2D** в Burn через reshape  
✅ Следует **существующему полиморфизму** BaseNode  
✅ Поддерживает **все стандартные операции**:  
   - Shape propagation  
   - Compatibility checks  
   - Mutations & crossover  
   - Serialization/deserialization  

**Шаги для обновления copilot-instructions.md:**
1. Добавить Conv1D в таблицу поддерживаемых слоёв
2. Описать входную/выходную размерность
3. Указать параметры (filters, kernel_size, stride, padding)
4. Дать пример сети для временных рядов
