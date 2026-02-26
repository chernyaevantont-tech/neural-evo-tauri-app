import { BaseNode } from "./nodes/base_node"
import { Conv2DNode } from "./nodes/layers/conv_node"
import { DenseNode } from "./nodes/layers/dense_node"
import { FlattenNode } from "./nodes/layers/flatten_node"
import { PoolingNode } from "./nodes/layers/pooling_node"

export class Genome {
    public inputNodes: BaseNode[]
    public outputNodes: BaseNode[]

    constructor(
        inputNodes: BaseNode[],
        outputNodes: BaseNode[]
    ) {
        this.inputNodes = inputNodes;
        this.outputNodes = outputNodes;
    }

    private _getRandomSubgenome(): BaseNode[] {
        // Шаг 1: Находим все линейные участки (цепочки нод с 1 входом и 1 выходом)
        const visited = new Set<string>();
        const linearChains: BaseNode[][] = [];

        // Функция для построения линейной цепочки начиная с узла
        const buildLinearChain = (startNode: BaseNode): BaseNode[] => {
            const chain: BaseNode[] = [];
            let current: BaseNode | null = startNode;

            // Идём назад до начала цепочки
            while (current && current.previous.length === 1 && current.next.length === 1) {
                const prev: BaseNode = current.previous[0];
                if (visited.has(prev.id) || prev.previous.length !== 1 || prev.next.length !== 1) {
                    break;
                }
                current = prev;
            }

            // Теперь идём вперёд, собирая всю цепочку
            while (current && current.previous.length === 1 && current.next.length === 1 && !visited.has(current.id)) {
                visited.add(current.id);
                chain.push(current);
                current = current.next[0];
            }

            return chain;
        };

        // Собираем все узлы графа через BFS
        const allNodes = new Set<BaseNode>();
        const queue: BaseNode[] = [...this.inputNodes];
        const visitedBFS = new Set<string>();

        this.inputNodes.forEach(node => visitedBFS.add(node.id));

        while (queue.length > 0) {
            const node = queue.shift()!;
            allNodes.add(node);

            node.next.forEach(nextNode => {
                if (!visitedBFS.has(nextNode.id)) {
                    visitedBFS.add(nextNode.id);
                    queue.push(nextNode);
                }
            });
        }

        // Ищем линейные цепочки
        allNodes.forEach(node => {
            if (!visited.has(node.id) && node.previous.length === 1 && node.next.length === 1) {
                const chain = buildLinearChain(node);
                if (chain.length >= 2) { // Минимум 2 ноды в цепочке
                    linearChains.push(chain);
                }
            }
        });

        // Шаг 2: Выбираем случайный линейный участок
        if (linearChains.length === 0) {
            throw new Error("No linear chains found for subgenome extraction");
        }

        const selectedChain = linearChains[Math.floor(Math.random() * linearChains.length)];

        // Шаг 3: Выбираем случайный связанный подучасток из выбранной цепочки
        // Выбираем случайную начальную позицию
        const maxStartIndex = selectedChain.length - 1;
        const startIndex = Math.floor(Math.random() * maxStartIndex);

        // Выбираем случайную длину подучастка (минимум 1, максимум до конца цепочки)
        const maxLength = selectedChain.length - startIndex;
        const length = Math.floor(Math.random() * maxLength) + 1;

        const subchain = selectedChain.slice(startIndex, startIndex + length);

        return subchain
    }

    public GetRandomSubgenomeNodeIds(): string[] {
        return this._getRandomSubgenome().map(node => node.id);
    }

    public GetRandomSubgenome(): BaseNode[] {
        const subgenome = this._getRandomSubgenome();
        if (subgenome.length == 0) return [];
        const newSubgenome: BaseNode[] = [subgenome[0].Clone()];
        for (let i = 1; i < subgenome.length; i++) {
            newSubgenome.push(subgenome[i].Clone());
            newSubgenome[i - 1].AddNext(newSubgenome[i]);
        }
        return newSubgenome;
    }

    public FindInsertionPoint(
        subgenomeInputNode: BaseNode,
        subgenomeOutputNode: BaseNode
    ): {
        cutFromNodeId: string;
        cutToNodeId: string;
        inputAdapterNodes: BaseNode[];
        outputAdapterNodes: BaseNode[];
    } | null {
        // Собираем все пары последовательных нод в графе (потенциальные точки разреза)
        const allEdges: { from: BaseNode; to: BaseNode }[] = [];
        const visitedBFS = new Set<string>();
        const queue: BaseNode[] = [...this.inputNodes];

        this.inputNodes.forEach(node => visitedBFS.add(node.id));

        while (queue.length > 0) {
            const node = queue.shift()!;

            node.next.forEach(nextNode => {
                // Сохраняем ребро как потенциальную точку разреза
                allEdges.push({ from: node, to: nextNode });

                if (!visitedBFS.has(nextNode.id)) {
                    visitedBFS.add(nextNode.id);
                    queue.push(nextNode);
                }
            });
        }

        // Структура для хранения возможных точек вставки
        const validInsertionPoints: {
            cutFromNodeId: string;
            cutToNodeId: string;
            inputAdapterNodes: BaseNode[];
            outputAdapterNodes: BaseNode[];
        }[] = [];

        // Проверяем каждое ребро как потенциальную точку вставки
        for (const edge of allEdges) {
            const fromNode = edge.from;
            const toNode = edge.to;

            // Проверяем совместимость: fromNode -> subgenomeInputNode
            let inputAdapters: BaseNode[] = [];
            const inputCompatible = fromNode.CheckCompability(subgenomeInputNode);

            if (!inputCompatible) {
                // Пытаемся создать адаптер
                const adapter = this.createAdapter(fromNode.GetOutputShape(), subgenomeInputNode.GetOutputShape());
                if (adapter) {
                    inputAdapters = adapter;
                } else {
                    continue; // Невозможно создать адаптер, пропускаем эту точку
                }
            }

            // Проверяем совместимость: subgenomeOutputNode -> toNode
            let outputAdapters: BaseNode[] = [];
            const outputCompatible = subgenomeOutputNode.CheckCompability(toNode);

            if (!outputCompatible) {
                // Пытаемся создать адаптер
                const adapter = this.createAdapter(subgenomeOutputNode.GetOutputShape(), toNode.GetOutputShape());
                if (adapter) {
                    outputAdapters = adapter;
                } else {
                    continue; // Невозможно создать адаптер, пропускаем эту точку
                }
            }

            // Если оба соединения возможны, добавляем эту точку в список
            validInsertionPoints.push({
                cutFromNodeId: fromNode.id,
                cutToNodeId: toNode.id,
                inputAdapterNodes: inputAdapters,
                outputAdapterNodes: outputAdapters
            });
        }

        // Если нет возможных точек вставки, возвращаем null
        if (validInsertionPoints.length === 0) {
            return null;
        }

        // Случайно выбираем одну из возможных точек вставки
        const selectedPoint = validInsertionPoints[Math.floor(Math.random() * validInsertionPoints.length)];

        return selectedPoint;
    }

    // Вспомогательная функция для создания адаптеров между несовместимыми формами
    private createAdapter(fromShape: number[], toShape: number[]): BaseNode[] | null {
        const adapters: BaseNode[] = [];

        // Случай 1: 3D -> 3D (различные пространственные размеры или каналы)
        if (fromShape.length === 3 && toShape.length === 3) {
            const [fromH, fromW, fromC] = fromShape;
            const [toH, toW, toC] = toShape;

            // Если нужно изменить пространственные размеры
            if (fromH !== toH || fromW !== toW) {
                // Определяем, нужно уменьшать или увеличивать
                const needDownsample = fromH > toH || fromW > toW;

                if (needDownsample) {
                    // Используем pooling для уменьшения
                    const hRatio = fromH / toH;
                    const wRatio = fromW / toW;

                    // Выбираем stride чтобы приблизиться к целевому размеру
                    const stride = Math.max(2, Math.floor(Math.min(hRatio, wRatio)));
                    const kernelSize = { h: stride, w: stride };

                    adapters.push(new PoolingNode('max', kernelSize, stride, 0));

                    // После pooling пересчитываем размер
                    const newH = Math.floor((fromH - kernelSize.h) / stride + 1);
                    const newW = Math.floor((fromW - kernelSize.w) / stride + 1);

                    // Если всё ещё не совпадает, пробуем conv2d с подходящим stride
                    if (newH !== toH || newW !== toW) {
                        const remainingHRatio = newH / toH;
                        const remainingWRatio = newW / toW;
                        const convStride = Math.max(1, Math.floor(Math.min(remainingHRatio, remainingWRatio)));

                        adapters.push(new Conv2DNode(
                            fromC,
                            { h: 3, w: 3 },
                            convStride,
                            1,
                            1,
                            true
                        ));
                    }
                } else {
                    // Для upsampling используем conv2d с padding
                    // (в реальности нужна транспонированная свёртка, но упростим)
                    return null; // Upsampling пока не поддерживается
                }
            }

            // Если нужно изменить количество каналов
            if (fromC !== toC) {
                // Используем Conv2D 1x1 для изменения количества каналов
                adapters.push(new Conv2DNode(
                    toC,
                    { h: 1, w: 1 },
                    1,
                    0,
                    1,
                    true
                ));
            }

            return adapters.length > 0 ? adapters : null;
        }

        // Случай 2: 3D -> 1D (нужен Flatten)
        // После Flatten форма становится 1D, а любой 1D узел (Dense) автоматически примет этот размер
        // как свой input_shape. Дополнительный Dense слой в качестве адаптера не нужен.
        if (fromShape.length === 3 && toShape.length === 1) {
            adapters.push(new FlattenNode());
            return adapters;
        }

        // Случай 3: 1D -> 1D (различное количество нейронов или другие параметры)
        // Для Dense слоёв количество нейронов предыдущего слоя автоматически становится 
        // input_shape следующего слоя при сборке. Адаптер здесь НЕ нужен.
        if (fromShape.length === 1 && toShape.length === 1) {
            return null; // Адаптер не требуется, просто соединяем напрямую
        }

        // Случай 4: 1D -> 3D (очень сложно, не поддерживается)
        if (fromShape.length === 1 && toShape.length === 3) {
            return null; // Reshape из 1D в 3D пока не поддерживается
        }

        // Неизвестный случай
        return null;
    }

    public Breed(genome: Genome, maxNodes?: number): { genome: Genome, nodes: BaseNode[], isValid: boolean } | null {
        const fromSubgenome = genome.GetRandomSubgenome();
        if (fromSubgenome.length == 0) return null;
        const insertion = this.FindInsertionPoint(fromSubgenome[0], fromSubgenome[fromSubgenome.length - 1]);
        if (insertion == null) return null;

        const nodesToCheck = [...this.inputNodes];
        const nodesChecked = new Set<BaseNode>();
        const oldNodes: BaseNode[] = [];
        const newNodes: BaseNode[] = [];
        const oldNewNode = new Map<BaseNode, BaseNode>();
        const newInputNodes: BaseNode[] = [];
        const newOutputNodes: BaseNode[] = [];
        let cutToNode: BaseNode | null = null;
        let isValidFlag = true;

        while (nodesToCheck.length > 0) {
            const currentNode = nodesToCheck.shift()!;
            if (nodesChecked.has(currentNode)) continue;
            nodesChecked.add(currentNode);
            const newNode = currentNode.Clone();
            newNodes.push(newNode);
            if (currentNode.next.length == 0) {
                if (currentNode.GetNodeType() != "Output") isValidFlag = false;
                newOutputNodes.push(newNode)
            }
            if (currentNode.previous.length == 0) {
                if (currentNode.GetNodeType() != "Input") isValidFlag = false;
                newInputNodes.push(newNode)
            }
            if (currentNode.id == insertion.cutToNodeId) {
                cutToNode = newNode;
            }
            oldNodes.push(currentNode);
            oldNewNode.set(currentNode, newNode);
            nodesToCheck.push(...currentNode.next)
        }

        for (let i = 0; i < newNodes.length; i++) {
            if (oldNodes[i].id == insertion.cutFromNodeId) {
                if (insertion.inputAdapterNodes.length > 0) {
                    newNodes[i].AddNext(insertion.inputAdapterNodes[0]);
                    for (let j = 1; j < insertion.inputAdapterNodes.length; j++) {
                        insertion.inputAdapterNodes[j - 1].AddNext(insertion.inputAdapterNodes[j]);
                    }

                    insertion.inputAdapterNodes[insertion.inputAdapterNodes.length - 1].AddNext(fromSubgenome[0]);
                } else {
                    newNodes[i].AddNext(fromSubgenome[0]);
                }

                if (insertion.outputAdapterNodes.length > 0) {
                    fromSubgenome[fromSubgenome.length - 1].AddNext(insertion.outputAdapterNodes[0]);
                    for (let j = 1; j < insertion.outputAdapterNodes.length; j++) {
                        insertion.outputAdapterNodes[j - 1].AddNext(insertion.outputAdapterNodes[j]);
                    }
                    insertion.outputAdapterNodes[insertion.outputAdapterNodes.length - 1].AddNext(cutToNode!)
                } else {
                    fromSubgenome[fromSubgenome.length - 1].AddNext(cutToNode!);
                }

                for (let oldNextNode of oldNodes[i].next) {
                    const newNextNode = oldNewNode.get(oldNextNode)!;
                    if (newNextNode == cutToNode) continue;
                    newNodes[i].AddNext(newNextNode);
                }
            }
            else {
                for (let oldNextNode of oldNodes[i].next) {
                    newNodes[i].AddNext(oldNewNode.get(oldNextNode)!);
                }
            }
        }

        newNodes.push(
            ...fromSubgenome,
            ...insertion.inputAdapterNodes,
            ...insertion.outputAdapterNodes
        );

        if (maxNodes !== undefined && newNodes.length > maxNodes) {
            return null;
        }

        return {
            genome: new Genome(newInputNodes, newOutputNodes),
            nodes: newNodes,
            isValid: isValidFlag
        }
    }

    public MutateRemoveNode(): { genome: Genome, nodes: BaseNode[], isValid: boolean } | null {
        // 1. Gather all nodes via BFS to ensure we have a flat list of node objects
        const allNodes: BaseNode[] = [];
        const visitedBFS = new Set<string>();
        const queue: BaseNode[] = [...this.inputNodes];

        this.inputNodes.forEach(node => visitedBFS.add(node.id));

        while (queue.length > 0) {
            const node = queue.shift()!;
            allNodes.push(node);

            node.next.forEach(nextNode => {
                if (!visitedBFS.has(nextNode.id)) {
                    visitedBFS.add(nextNode.id);
                    queue.push(nextNode);
                }
            });
        }

        // 2. Find candidates for removal (1 in, 1 out, not Input, not Output, not Flatten)
        // Flatten is dangerous to remove alone because it changes dims drastically.
        const candidates = allNodes.filter(n =>
            n.previous.length === 1 &&
            n.next.length === 1 &&
            n.GetNodeType() !== "Input" &&
            n.GetNodeType() !== "Output"
        );

        if (candidates.length === 0) return null;

        // 3. Pick a random node to remove
        const nodeToRemove = candidates[Math.floor(Math.random() * candidates.length)];
        const prevNodeOriginal = nodeToRemove.previous[0];
        const nextNodeOriginal = nodeToRemove.next[0];

        // 4. Test compatibility between prev and next
        let adapters: BaseNode[] = [];
        const isCompatible = prevNodeOriginal.CheckCompability(nextNodeOriginal);

        if (!isCompatible) {
            const newAdapters = this.createAdapter(prevNodeOriginal.GetOutputShape(), nextNodeOriginal.GetInputShape());
            if (newAdapters) {
                adapters = newAdapters;
            } else {
                return null; // Cannot create valid adapter to close the gap
            }
        }

        // 5. Build the new Genome by cloning the graph but omitting nodeToRemove
        // and inserting adapters if any.
        const nodesToCheck = [...this.inputNodes];
        const nodesChecked = new Set<BaseNode>();
        const oldNodes: BaseNode[] = [];
        const newNodes: BaseNode[] = [];
        const oldNewNode = new Map<BaseNode, BaseNode>();

        const newInputNodes: BaseNode[] = [];
        const newOutputNodes: BaseNode[] = [];
        let isValidFlag = true;

        // First pass: Clone all nodes EXCEPT the one being removed
        while (nodesToCheck.length > 0) {
            const currentNode = nodesToCheck.shift()!;
            if (nodesChecked.has(currentNode)) continue;
            nodesChecked.add(currentNode);

            if (currentNode.id !== nodeToRemove.id) {
                const newNode = currentNode.Clone();
                newNodes.push(newNode);

                if (currentNode.next.length === 0 || (currentNode.next.length === 1 && currentNode.next[0].id === nodeToRemove.id && nodeToRemove.next.length === 0)) {
                    // It's an output node or becomes an output node
                    if (currentNode.GetNodeType() !== "Output") isValidFlag = false;
                    newOutputNodes.push(newNode);
                }

                if (currentNode.previous.length === 0) {
                    if (currentNode.GetNodeType() !== "Input") isValidFlag = false;
                    newInputNodes.push(newNode);
                }

                oldNodes.push(currentNode);
                oldNewNode.set(currentNode, newNode);
            }

            nodesToCheck.push(...currentNode.next);
        }

        // Second pass: Restore connections
        for (let i = 0; i < newNodes.length; i++) {
            const oldNode = oldNodes[i];
            const newNode = newNodes[i];

            if (oldNode.id === prevNodeOriginal.id) {
                // This is the node BEFORE the removed node. 
                // We need to route it to the adapters, OR directly to nextNode.

                // Add regular connections that aren't the removed node
                for (let oldNext of oldNode.next) {
                    if (oldNext.id !== nodeToRemove.id) {
                        newNode.AddNext(oldNewNode.get(oldNext)!);
                    }
                }

                if (adapters.length > 0) {
                    newNode.AddNext(adapters[0]);
                    for (let j = 1; j < adapters.length; j++) {
                        adapters[j - 1].AddNext(adapters[j]);
                    }
                    adapters[adapters.length - 1].AddNext(oldNewNode.get(nextNodeOriginal)!);
                } else {
                    newNode.AddNext(oldNewNode.get(nextNodeOriginal)!);
                }
            } else {
                // Normal node
                for (let oldNext of oldNode.next) {
                    if (oldNext.id !== nodeToRemove.id) {
                        newNode.AddNext(oldNewNode.get(oldNext)!);
                    }
                }
            }
        }

        newNodes.push(...adapters);

        console.log(`[MutateRemoveNode] Removed node ${nodeToRemove.id} (${nodeToRemove.GetNodeType()}).`);
        console.log(`[MutateRemoveNode] Generated ${adapters.length} adapters: [${adapters.map(a => a.GetNodeType()).join(', ')}].`);
        console.log(`[MutateRemoveNode] New graph size: ${newNodes.length} (Old: ${allNodes.length})`);

        return {
            genome: new Genome(newInputNodes, newOutputNodes),
            nodes: newNodes,
            isValid: isValidFlag
        };
    }

    public MutateAddNode(): { genome: Genome, nodes: BaseNode[], isValid: boolean } | null {
        // 1. Gather all edges in the graph
        const allEdges: { from: BaseNode; to: BaseNode }[] = [];
        const visitedBFS = new Set<string>();
        const queue: BaseNode[] = [...this.inputNodes];

        this.inputNodes.forEach(node => visitedBFS.add(node.id));

        while (queue.length > 0) {
            const node = queue.shift()!;

            node.next.forEach(nextNode => {
                allEdges.push({ from: node, to: nextNode });

                if (!visitedBFS.has(nextNode.id)) {
                    visitedBFS.add(nextNode.id);
                    queue.push(nextNode);
                }
            });
        }

        if (allEdges.length === 0) return null;

        // 2. Pick a random edge
        const selectedEdge = allEdges[Math.floor(Math.random() * allEdges.length)];
        const fromNodeOriginal = selectedEdge.from;
        const toNodeOriginal = selectedEdge.to;

        console.log(`[MutateAddNode] Breaking edge ${fromNodeOriginal.id} -> ${toNodeOriginal.id}`);

        // 3. Generate a random layer node (Conv2D, Dense, Pooling) based on the input shape
        // For simplicity, we choose a simple Dense or Conv layer depending on the shape
        const inputShape = fromNodeOriginal.GetOutputShape();
        let newLayer: BaseNode;

        if (inputShape.length === 3) {
            // 3D data: Conv2D or Pooling
            const layerType = Math.random() > 0.5 ? 'conv2d' : 'pooling';
            if (layerType === 'conv2d') {
                const filters = [16, 32, 64][Math.floor(Math.random() * 3)];
                newLayer = new Conv2DNode(filters, { h: 3, w: 3 }, 1, 1, 1, true);
            } else {
                newLayer = new PoolingNode('max', { h: 2, w: 2 }, 2, 0);
            }
        } else {
            // 1D data: Dense
            const units = [32, 64, 128][Math.floor(Math.random() * 3)];
            newLayer = new DenseNode(units, 'relu', true);
        }

        console.log(`[MutateAddNode] Generated random layer: ${newLayer.GetNodeType()}`);

        // 4. Test compatibility and create adapters for the two new connections
        // Connection 1: fromNodeOriginal -> [adapter1?] -> newLayer
        let inputAdapters: BaseNode[] = [];
        if (!fromNodeOriginal.CheckCompability(newLayer)) {
            const adapters = this.createAdapter(fromNodeOriginal.GetOutputShape(), newLayer.GetInputShape());
            if (adapters) {
                inputAdapters = adapters;
            } else {
                return null;
            }
        }

        // We must calculate the new layer's internal shape logic manually for the adapter check
        if (inputAdapters.length > 0) {
            newLayer.CheckCompability(inputAdapters[inputAdapters.length - 1]);
        } else {
            newLayer.CheckCompability(fromNodeOriginal);
        }

        // Connection 2: newLayer -> [adapter2?] -> toNodeOriginal
        let outputAdapters: BaseNode[] = [];
        if (!newLayer.CheckCompability(toNodeOriginal)) {
            const adapters = this.createAdapter(newLayer.GetOutputShape(), toNodeOriginal.GetInputShape());
            if (adapters) {
                outputAdapters = adapters;
            } else {
                return null;
            }
        }

        // 5. Build the new Genome by cloning the graph and routing the selected edge through the new nodes
        const nodesToCheck = [...this.inputNodes];
        const nodesChecked = new Set<BaseNode>();
        const oldNodes: BaseNode[] = [];
        const newNodes: BaseNode[] = [];
        const oldNewNode = new Map<BaseNode, BaseNode>();

        const newInputNodes: BaseNode[] = [];
        const newOutputNodes: BaseNode[] = [];
        let isValidFlag = true;

        // First pass: Clone all nodes
        while (nodesToCheck.length > 0) {
            const currentNode = nodesToCheck.shift()!;
            if (nodesChecked.has(currentNode)) continue;
            nodesChecked.add(currentNode);

            const newNode = currentNode.Clone();
            newNodes.push(newNode);

            if (currentNode.next.length === 0) {
                if (currentNode.GetNodeType() !== "Output") isValidFlag = false;
                newOutputNodes.push(newNode);
            }

            if (currentNode.previous.length === 0) {
                if (currentNode.GetNodeType() !== "Input") isValidFlag = false;
                newInputNodes.push(newNode);
            }

            oldNodes.push(currentNode);
            oldNewNode.set(currentNode, newNode);
            nodesToCheck.push(...currentNode.next);
        }

        // Second pass: Restore connections but break the selected edge and insert new nodes
        for (let i = 0; i < newNodes.length; i++) {
            const oldNode = oldNodes[i];
            const newNode = newNodes[i];

            if (oldNode.id === fromNodeOriginal.id) {
                // Route all edges EXCEPT the broken one
                for (let oldNext of oldNode.next) {
                    if (oldNext.id === toNodeOriginal.id) {
                        // This is the broken edge. Replace `fromNode -> toNode` with
                        // `fromNode -> inputAdapters -> newLayer -> outputAdapters -> toNode`

                        let currentTail = newNode;

                        // Append Input Adapters
                        for (let adapter of inputAdapters) {
                            currentTail.AddNext(adapter);
                            currentTail = adapter;
                        }

                        // Append New Layer
                        currentTail.AddNext(newLayer);
                        currentTail = newLayer;

                        // Append Output Adapters
                        for (let adapter of outputAdapters) {
                            currentTail.AddNext(adapter);
                            currentTail = adapter;
                        }

                        // Final connection to the original target Node
                        currentTail.AddNext(oldNewNode.get(toNodeOriginal)!);
                    } else {
                        // Normal edge, just reconnect
                        newNode.AddNext(oldNewNode.get(oldNext)!);
                    }
                }
            } else {
                // Normal node
                for (let oldNext of oldNode.next) {
                    newNode.AddNext(oldNewNode.get(oldNext)!);
                }
            }
        }

        newNodes.push(...inputAdapters, newLayer, ...outputAdapters);

        console.log(`[MutateAddNode] Mutation successful. Inserted layer + ${inputAdapters.length + outputAdapters.length} adapters. New graph size: ${newNodes.length}`);

        return {
            genome: new Genome(newInputNodes, newOutputNodes),
            nodes: newNodes,
            isValid: isValidFlag
        };
    }
}