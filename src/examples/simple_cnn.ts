// Пример создания простой архитектуры вручную через код
import { InputNode } from '../evo/nodes/layers/input_node';
import { Conv2DNode } from '../evo/nodes/layers/conv_node';
import { PoolingNode } from '../evo/nodes/layers/pooling_node';
import { DenseNode } from '../evo/nodes/layers/dense_node';
import { Genome } from '../evo/genome';

// Создаем простую CNN архитектуру: Input -> Conv2D -> Pooling -> Dense

// 1. Входной слой 28x28x1 (как MNIST)
const input = new InputNode(28, 28, 1);

// 2. Сверточный слой: 32 фильтра, ядро 3x3, stride=1, padding=1
const conv1 = new Conv2DNode(32, { h: 3, w: 3 }, 1, 1, 1, true);

// 3. Max pooling: 2x2, stride=2
const pool1 = new PoolingNode('max', { h: 2, w: 2 }, 2, 0);

// 4. Сверточный слой: 64 фильтра
const conv2 = new Conv2DNode(64, { h: 3, w: 3 }, 1, 1, 1, true);

// 5. Max pooling
const pool2 = new PoolingNode('max', { h: 2, w: 2 }, 2, 0);

// 6. Полносвязный слой: 128 нейронов
const dense1 = new DenseNode(128, 'relu', true);

// 7. Выходной слой: 10 классов (для MNIST)
const output = new DenseNode(10, 'softmax', true);

// Соединяем слои
input.AddNext(conv1);
conv1.AddNext(pool1);
pool1.AddNext(conv2);
conv2.AddNext(pool1);
pool2.AddNext(dense1);
dense1.AddNext(output);

// Создаем геном
const genome = new Genome([input], [output]);

// Выводим информацию о каждом слое
console.log('=== Архитектура сети ===');
console.log('Input:', input.GetInfo());
console.log('Conv1:', conv1.GetInfo());
console.log('Pool1:', pool1.GetInfo());
console.log('Conv2:', conv2.GetInfo());
console.log('Pool2:', pool2.GetInfo());
console.log('Dense1:', dense1.GetInfo());
console.log('Output:', output.GetInfo());

// Получаем ресурсы (пример для float32 = 4 байта)
const dtype = 4;
console.log('\n=== Ресурсы сети ===');
console.log('Conv1 resources:', conv1.GetResources(dtype));
console.log('Dense1 resources:', dense1.GetResources(dtype));

export { genome };
