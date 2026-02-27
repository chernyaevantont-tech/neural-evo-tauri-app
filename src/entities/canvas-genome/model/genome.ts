import { BaseNode } from "./nodes/base_node"
import { Conv2DNode } from "./nodes/layers/conv_node"
import { DenseNode } from "./nodes/layers/dense_node"
import { FlattenNode } from "./nodes/layers/flatten_node"
import { PoolingNode } from "./nodes/layers/pooling_node"
import { AddNode } from "./nodes/merge/add_node"

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

    /**
     * Calculates the total resource cost (params count, ram, macs) of the entire Genome topological graph.
     * Useful for Parsimony Pressure and Resource-Aware Fitness evaluation.
     */
    public GetGenomeResources(dtype: number = 4): { totalFlash: number; totalRam: number; totalMacs: number; totalNodes: number } {
        const allNodes = this.getAllNodes();
        let totalFlash = 0;
        let totalRam = 0;
        let totalMacs = 0;

        for (const node of allNodes) {
            const res = node.GetResources(dtype);
            totalFlash += res.flash;
            totalRam += res.ram;
            totalMacs += res.macs;
        }

        return {
            totalFlash,
            totalRam,
            totalMacs,
            totalNodes: allNodes.length
        };
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
                const adapter = this.createAdapter(fromNode.GetOutputShape(), subgenomeInputNode.GetInputShape());
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
                const adapter = this.createAdapter(subgenomeOutputNode.GetOutputShape(), toNode.GetInputShape());
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
        // Для Dense слоёв количество нейронов предыдущего слоя обычно становится 
        // input_shape следующего слоя при сборке. Но если мы вставляем AddNode, 
        // нам нужно строгое совпадение размерности, поэтому мы генерируем линейный Dense-адаптер.
        if (fromShape.length === 1 && toShape.length === 1) {
            if (fromShape[0] !== toShape[0]) {
                adapters.push(new DenseNode(toShape[0], 'relu', false));
                return adapters;
            }
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
            n.GetNodeType() !== "Output" &&
            n.GetNodeType() !== "Flatten"
        );

        if (candidates.length === 0) return null;

        // 3. Shuffle candidates to try them randomly
        const shuffledCandidates = [...candidates].sort(() => Math.random() - 0.5);

        for (const nodeToRemove of shuffledCandidates) {
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
                    continue; // Cannot create valid adapter to close the gap, try next candidate
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

        return null;
    }

    public MutateAddNode(maxNodes?: number): { genome: Genome, nodes: BaseNode[], isValid: boolean } | null {
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

        // Shuffle edges to try them randomly
        const shuffledEdges = [...allEdges].sort(() => Math.random() - 0.5);

        for (const selectedEdge of shuffledEdges) {
            const fromNodeOriginal = selectedEdge.from;
            const toNodeOriginal = selectedEdge.to;

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

            // 4. Test compatibility and create adapters for the two new connections
            // Connection 1: fromNodeOriginal -> [adapter1?] -> newLayer
            let inputAdapters: BaseNode[] = [];
            if (!fromNodeOriginal.CheckCompability(newLayer)) {
                const adapters = this.createAdapter(fromNodeOriginal.GetOutputShape(), newLayer.GetInputShape());
                if (adapters) {
                    inputAdapters = adapters;
                } else {
                    continue; // Skip to next edge if adapter creation fails
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
                    continue; // Skip to next edge if adapter creation fails
                }
            }

            console.log(`[MutateAddNode] Breaking edge ${fromNodeOriginal.id} -> ${toNodeOriginal.id}`);
            console.log(`[MutateAddNode] Generated random layer: ${newLayer.GetNodeType()}`);

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

            if (maxNodes !== undefined && newNodes.length > maxNodes) {
                return null;
            }

            console.log(`[MutateAddNode] Mutation successful. Inserted layer + ${inputAdapters.length + outputAdapters.length} adapters. New graph size: ${newNodes.length}`);

            return {
                genome: new Genome(newInputNodes, newOutputNodes),
                nodes: newNodes,
                isValid: isValidFlag
            };
        }


        return null;
    }

    public MutateAddSkipConnection(maxNodes?: number): { genome: Genome, nodes: BaseNode[], isValid: boolean } | null {
        try {
            const allNodes = this.getAllNodes();

            // Map reachability (BFS from each node)
            const reachable = new Map<string, Set<string>>();
            for (const node of allNodes) {
                const reached = new Set<string>();
                const queue: BaseNode[] = [...node.next];
                while (queue.length > 0) {
                    const curr = queue.shift()!;
                    if (!reached.has(curr.id)) {
                        reached.add(curr.id);
                        queue.push(...curr.next);
                    }
                }
                reachable.set(node.id, reached);
            }

            // Find valid pairs
            const validPairs: { source: BaseNode, target: BaseNode }[] = [];
            for (const source of allNodes) {
                const reached = reachable.get(source.id)!;
                for (const target of allNodes) {
                    if (reached.has(target.id)) {
                        // target is not immediate child
                        if (!source.next.find(n => n.id === target.id)) {
                            // target has exactly 1 previous node, and is not Input/Output
                            if (target.previous.length === 1 && target.GetNodeType() !== "Input" && target.GetNodeType() !== "Output") {
                                validPairs.push({ source, target });
                            }
                        }
                    }
                }
            }

            if (validPairs.length === 0) return null;

            const shuffledPairs = validPairs.sort(() => Math.random() - 0.5);

            for (const pair of shuffledPairs) {
                const sourceOrig = pair.source;
                const targetOrig = pair.target;
                const prevOrig = targetOrig.previous[0];

                const targetShape = prevOrig.GetOutputShape();
                let adapters: BaseNode[] = [];

                const sourceShapeOrig = sourceOrig.GetOutputShape();
                let isCompatible = true;

                if (sourceShapeOrig.length !== targetShape.length || !sourceShapeOrig.every((val, i) => val === targetShape[i])) {
                    const ad = this.createAdapter(sourceShapeOrig, targetShape);
                    if (ad) {
                        adapters = ad;
                    } else {
                        isCompatible = false;
                    }
                }

                if (!isCompatible) continue;

                const addNode = new AddNode();

                // Standard clone
                const nodesToCheck = [...this.inputNodes];
                const nodesChecked = new Set<BaseNode>();
                const oldNodes: BaseNode[] = [];
                const newNodes: BaseNode[] = [];
                const oldNewNode = new Map<BaseNode, BaseNode>();
                const newInputNodes: BaseNode[] = [];
                const newOutputNodes: BaseNode[] = [];
                let isValidFlag = true;

                while (nodesToCheck.length > 0) {
                    const currentNode = nodesToCheck.shift()!;
                    if (nodesChecked.has(currentNode)) continue;
                    nodesChecked.add(currentNode);

                    const newNode = currentNode.Clone();
                    newNodes.push(newNode);
                    oldNodes.push(currentNode);
                    oldNewNode.set(currentNode, newNode);

                    if (currentNode.next.length === 0) {
                        if (currentNode.GetNodeType() !== "Output") isValidFlag = false;
                        newOutputNodes.push(newNode);
                    }
                    if (currentNode.previous.length === 0) {
                        if (currentNode.GetNodeType() !== "Input") isValidFlag = false;
                        newInputNodes.push(newNode);
                    }
                    nodesToCheck.push(...currentNode.next);
                }

                // Wiring
                for (let i = 0; i < newNodes.length; i++) {
                    const oldNode = oldNodes[i];
                    const newNode = newNodes[i];

                    if (oldNode.id === prevOrig.id) {
                        for (let oldNext of oldNode.next) {
                            if (oldNext.id === targetOrig.id) {
                                newNode.AddNext(addNode);
                            } else {
                                newNode.AddNext(oldNewNode.get(oldNext)!);
                            }
                        }
                    } else if (oldNode.id === sourceOrig.id) {
                        for (let oldNext of oldNode.next) {
                            newNode.AddNext(oldNewNode.get(oldNext)!);
                        }

                        let tail = newNode;
                        for (const ad of adapters) {
                            tail.AddNext(ad);
                            tail = ad;
                        }
                        tail.AddNext(addNode);
                    } else {
                        for (let oldNext of oldNode.next) {
                            newNode.AddNext(oldNewNode.get(oldNext)!);
                        }
                    }
                }

                addNode.AddNext(oldNewNode.get(targetOrig)!);

                newNodes.push(addNode, ...adapters);

                if (maxNodes !== undefined && newNodes.length > maxNodes) {
                    return null;
                }

                console.log(`[MutateAddSkipConnection] Added skip connection from ${sourceOrig.GetNodeType()} to AddNode before ${targetOrig.GetNodeType()}. Adapters: ${adapters.length}`);

                return {
                    genome: new Genome(newInputNodes, newOutputNodes),
                    nodes: newNodes,
                    isValid: isValidFlag
                };
            }

            return null;
        } catch (e) {
            console.error("Add Skip Connection mutation error", e);
            return null;
        }
    }

    public MutateChangeLayerType(maxNodes?: number): { genome: Genome, nodes: BaseNode[], isValid: boolean } | null {
        try {
            const allNodes = this.getAllNodes();
            const candidates = allNodes.filter(n => {
                const type = n.GetNodeType();
                return type !== "Input" && type !== "Output" && type !== "Flatten" && !n.GetIsMerging();
            });

            if (candidates.length === 0) return null;

            const shuffledCandidates = candidates.sort(() => Math.random() - 0.5);

            for (const targetNodeOrig of shuffledCandidates) {
                // Generate a random new layer that is strategically DIFFERENT from the old one
                const layerTypes = [
                    { type: 'Dense', instance: () => new DenseNode(Math.floor(Math.random() * 64) + 16, Math.random() > 0.5 ? 'relu' : 'softmax', true) },
                    { type: 'Conv2D', instance: () => new Conv2DNode(Math.floor(Math.random() * 32) + 8, { h: 3, w: 3 }, 1, 0, 1, true) },
                    { type: 'Pooling', instance: () => new PoolingNode("max", { h: 2, w: 2 }, 2, 0) }
                ];

                const validLayerTypes = layerTypes.filter(l => l.type !== targetNodeOrig.GetNodeType());
                if (validLayerTypes.length === 0) continue;

                const newLayer = validLayerTypes[Math.floor(Math.random() * validLayerTypes.length)].instance();

                const inputShape = targetNodeOrig.GetInputShape();
                if (inputShape.length !== newLayer.GetExpectedInputDimensions() && newLayer.GetExpectedInputDimensions() !== "any") {
                    continue; // Skip if new layer fundamentally hates the incoming dimensions
                }

                // Standard clone pass: duplicate all nodes EXCEPT targetNodeOrig
                const nodesToCheck = [...this.inputNodes];
                const nodesChecked = new Set<BaseNode>();
                const oldNodes: BaseNode[] = [];
                const newNodes: BaseNode[] = [];
                const oldNewNode = new Map<BaseNode, BaseNode>();
                const newInputNodes: BaseNode[] = [];
                const newOutputNodes: BaseNode[] = [];
                let isValidFlag = true;

                while (nodesToCheck.length > 0) {
                    const currentNode = nodesToCheck.shift()!;
                    if (nodesChecked.has(currentNode)) continue;
                    nodesChecked.add(currentNode);

                    if (currentNode.id !== targetNodeOrig.id) {
                        const newNode = currentNode.Clone();
                        newNodes.push(newNode);
                        oldNodes.push(currentNode);
                        oldNewNode.set(currentNode, newNode);

                        if (currentNode.next.length === 0) {
                            if (currentNode.GetNodeType() !== "Output") isValidFlag = false;
                            newOutputNodes.push(newNode);
                        }
                        if (currentNode.previous.length === 0) {
                            if (currentNode.GetNodeType() !== "Input") isValidFlag = false;
                            newInputNodes.push(newNode);
                        }
                    }

                    nodesToCheck.push(...currentNode.next);
                }

                let inputsConnected = 0;
                for (let i = 0; i < newNodes.length; i++) {
                    const oldNode = oldNodes[i];
                    const newNode = newNodes[i];

                    if (oldNode.next.find(n => n.id === targetNodeOrig.id)) {
                        newNode.AddNext(newLayer);
                        inputsConnected++;
                    }
                }

                // If no incoming connections could be made, invalid mutation
                if (inputsConnected === 0) continue;

                const newOutputShape = newLayer.GetOutputShape();
                let allAdaptersValid = true;
                const adaptersForChildren: Map<BaseNode, BaseNode[]> = new Map();

                // Assess outputs and build adapters
                for (const oldChild of targetNodeOrig.next) {
                    const childInputShape = oldChild.GetInputShape();

                    let adapters: BaseNode[] = [];
                    if (newOutputShape.length !== childInputShape.length || !newOutputShape.every((val, i) => val === childInputShape[i])) {
                        const ad = this.createAdapter(newOutputShape, childInputShape);
                        if (ad) {
                            adapters = ad;
                        } else {
                            allAdaptersValid = false;
                            break;
                        }
                    }
                    adaptersForChildren.set(oldChild, adapters);
                }

                if (!allAdaptersValid) continue;

                // Wire outputs from the newly created layer to its adapted destinations
                for (const oldChild of targetNodeOrig.next) {
                    const adapters = adaptersForChildren.get(oldChild) || [];
                    const mappedChild = oldNewNode.get(oldChild);
                    if (!mappedChild) continue; // Safe guard against deleted references

                    let tail: BaseNode = newLayer;
                    for (const ad of adapters) {
                        tail.AddNext(ad);
                        tail = ad;
                        newNodes.push(ad);
                    }
                    tail.AddNext(mappedChild);
                }

                // Restore all other non-involved edges
                for (let i = 0; i < newNodes.length; i++) {
                    const oldNode = oldNodes[i];
                    const newNode = newNodes[i];

                    for (const oldChild of oldNode.next) {
                        if (oldChild.id !== targetNodeOrig.id) {
                            const mappedChild = oldNewNode.get(oldChild);
                            if (mappedChild) {
                                newNode.AddNext(mappedChild);
                            }
                        }
                    }
                }

                const finalNodes = [...newNodes, newLayer];

                if (maxNodes !== undefined && finalNodes.length > maxNodes) {
                    return null;
                }

                console.log(`[MutateChangeLayerType] Replaced ${targetNodeOrig.GetNodeType()} with ${newLayer.GetNodeType()}`);

                return {
                    genome: new Genome(newInputNodes, newOutputNodes),
                    nodes: finalNodes,
                    isValid: isValidFlag
                };
            }

            return null;
        } catch (e) {
            console.error("Change Layer Type mutation error", e);
            return null;
        }
    }

    public BreedByReplacement(donor: Genome, maxNodes?: number): { genome: Genome, nodes: BaseNode[], isValid: boolean } | null {
        // Try up to 10 permutations to find compatible replacements
        for (let attempt = 0; attempt < 10; attempt++) {
            try {
                // 1. Extract random target subgraph from self (recipient)
                const recipientSubgenomeOriginal = this._getRandomSubgenome();
                if (!recipientSubgenomeOriginal || recipientSubgenomeOriginal.length === 0) continue;

                const cutFromNodeId = recipientSubgenomeOriginal[0].id;
                const cutToNodeId = recipientSubgenomeOriginal[recipientSubgenomeOriginal.length - 1].id;

                // What node comes before the cut? What comes after?
                // Notice that `_getRandomSubgenome()` guarantees nodes have exactly 1 input and 1 output.
                const recipientPrevOriginal = recipientSubgenomeOriginal[0].previous[0];
                const recipientNextOriginal = recipientSubgenomeOriginal[recipientSubgenomeOriginal.length - 1].next[0];

                if (!recipientPrevOriginal || !recipientNextOriginal) continue;

                // 2. Extract random subgraph from donor
                const donorSubgenome = donor.GetRandomSubgenome();
                if (!donorSubgenome || donorSubgenome.length === 0) continue;

                const donorInShape = donorSubgenome[0].GetInputShape();
                const donorOutShape = donorSubgenome[donorSubgenome.length - 1].GetOutputShape();

                // 3. Test compatibility and create mapping adapters
                // Adapter 1: recipientPrev -> donorSubgenome[0]
                let inputAdapters: BaseNode[] = [];
                let inputCompatible = recipientPrevOriginal.CheckCompability(donorSubgenome[0]);
                if (!inputCompatible) {
                    const adapter = this.createAdapter(recipientPrevOriginal.GetOutputShape(), donorInShape);
                    if (adapter) {
                        inputAdapters = adapter;
                    } else {
                        continue; // Incompatible bounds, skip attempt
                    }
                }

                // Adapter 2: donorSubgenome[end] -> recipientNext
                let outputAdapters: BaseNode[] = [];
                let outputCompatible = donorSubgenome[donorSubgenome.length - 1].CheckCompability(recipientNextOriginal);
                if (!outputCompatible) {
                    const adapter = this.createAdapter(donorOutShape, recipientNextOriginal.GetInputShape());
                    if (adapter) {
                        outputAdapters = adapter;
                    } else {
                        continue;
                    }
                }

                // 4. Trace the nodes that belong to the recipient's subgenome so we can delete them
                const nodesToRemove = new Set<string>();
                for (const node of recipientSubgenomeOriginal) {
                    nodesToRemove.add(node.id);
                }

                // 5. Build the new Genome by cloning the graph, omitting the old subgraph, 
                // and inserting the new subgenome + adapters
                const nodesToCheck = [...this.inputNodes];
                const nodesChecked = new Set<BaseNode>();
                const oldNodes: BaseNode[] = [];
                const newNodes: BaseNode[] = [];
                const oldNewNode = new Map<BaseNode, BaseNode>();

                const newInputNodes: BaseNode[] = [];
                const newOutputNodes: BaseNode[] = [];
                let isValidFlag = true;

                // First pass: Clone all nodes EXCEPT the removed subgenome
                while (nodesToCheck.length > 0) {
                    const currentNode = nodesToCheck.shift()!;
                    if (nodesChecked.has(currentNode)) continue;
                    nodesChecked.add(currentNode);

                    if (!nodesToRemove.has(currentNode.id)) {
                        const newNode = currentNode.Clone();
                        newNodes.push(newNode);

                        // If the actual un-mutated node used to end the graph, but it's now an input to the donor...?
                        // Wait, previous rules caught this. If it was Input/Output, _getRandomSubgenome excludes them if length > 1, 
                        // but actually its previous/next 1 check isolates it from multi inputs. 
                        // Nonetheless, check standard bound conditions:
                        if (currentNode.next.length === 0 || (currentNode.next.length === 1 && nodesToRemove.has(currentNode.next[0].id))) {
                            if (currentNode.GetNodeType() !== "Output" && currentNode.id !== recipientPrevOriginal.id) {
                                isValidFlag = false;
                            }
                            if (currentNode.id !== recipientPrevOriginal.id) {
                                newOutputNodes.push(newNode);
                            }
                        }

                        if (currentNode.previous.length === 0 || (currentNode.previous.length === 1 && nodesToRemove.has(currentNode.previous[0].id))) {
                            if (currentNode.GetNodeType() !== "Input" && currentNode.id !== recipientNextOriginal.id) {
                                isValidFlag = false;
                            }
                            if (currentNode.id !== recipientNextOriginal.id) {
                                newInputNodes.push(newNode);
                            }
                        }

                        oldNodes.push(currentNode);
                        oldNewNode.set(currentNode, newNode);
                    }

                    nodesToCheck.push(...currentNode.next);
                }

                // Second pass: Restore connections, routing `recipientPrev` to the donor subgenome
                for (let i = 0; i < newNodes.length; i++) {
                    const oldNode = oldNodes[i];
                    const newNode = newNodes[i];

                    if (oldNode.id === recipientPrevOriginal.id) {
                        // Route to adapters -> donor 
                        let currentTail = newNode;

                        for (let adapter of inputAdapters) {
                            currentTail.AddNext(adapter);
                            currentTail = adapter;
                        }

                        currentTail.AddNext(donorSubgenome[0]);
                        currentTail = donorSubgenome[donorSubgenome.length - 1];

                        for (let adapter of outputAdapters) {
                            currentTail.AddNext(adapter);
                            currentTail = adapter;
                        }

                        // Attach end of donor subsegment to `recipientNextOriginal`
                        currentTail.AddNext(oldNewNode.get(recipientNextOriginal)!);

                        // Keep going for any OTHER valid branches this node had
                        for (let oldNext of oldNode.next) {
                            if (!nodesToRemove.has(oldNext.id)) {
                                newNode.AddNext(oldNewNode.get(oldNext)!);
                            }
                        }
                    } else if (oldNode.id === recipientNextOriginal.id) {
                        for (let oldNext of oldNode.next) {
                            if (!nodesToRemove.has(oldNext.id)) {
                                newNode.AddNext(oldNewNode.get(oldNext)!);
                            }
                        }
                    } else {
                        for (let oldNext of oldNode.next) {
                            if (!nodesToRemove.has(oldNext.id)) {
                                newNode.AddNext(oldNewNode.get(oldNext)!);
                            }
                        }
                    }
                }

                newNodes.push(...donorSubgenome, ...inputAdapters, ...outputAdapters);

                if (maxNodes !== undefined && newNodes.length > maxNodes) {
                    continue; // Rollover and try a smaller replacement
                }

                console.log(`[BreedByReplacement] Replaced a ${recipientSubgenomeOriginal.length}-node subgraph with a ${donorSubgenome.length}-node subgraph from donor!`);

                return {
                    genome: new Genome(newInputNodes, newOutputNodes),
                    nodes: newNodes,
                    isValid: isValidFlag
                };

            } catch (e) {
                console.error("Replacement error during attempt", e);
                continue;
            }
        }

        return null;
    }

    public getAllNodes(): BaseNode[] {
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
        return allNodes;
    }

    public BreedNeatStyle(donor: Genome, maxNodes?: number): { genome: Genome, nodes: BaseNode[], isValid: boolean } | null {
        try {
            const recipientNodes = this.getAllNodes();
            const donorNodes = donor.getAllNodes();

            const recipientInnovations = new Map<number, BaseNode>();
            recipientNodes.forEach(n => recipientInnovations.set(n.innovationId, n));

            // Find Disjoint nodes in donor (nodes not in recipient) and filter to linear ones for safe insertion
            const donorDisjointLinear = donorNodes.filter(n =>
                !recipientInnovations.has(n.innovationId) &&
                n.previous.length === 1 &&
                n.next.length === 1 &&
                n.GetNodeType() !== "Input" &&
                n.GetNodeType() !== "Output"
            );

            // Find ONE valid disjoint node to transplant
            // A node is valid if BOTH its previous and next anchor points exist in the recipient graph 
            let transplantNode: BaseNode | null = null;
            let anchorPrev: BaseNode | null = null;
            let anchorNext: BaseNode | null = null;

            // Shuffle disjoints to pick roughly random ones
            const shuffledDisjoints = [...donorDisjointLinear].sort(() => Math.random() - 0.5);

            for (const disjoint of shuffledDisjoints) {
                const prevInnovation = disjoint.previous[0].innovationId;
                const nextInnovation = disjoint.next[0].innovationId;

                if (recipientInnovations.has(prevInnovation) && recipientInnovations.has(nextInnovation)) {
                    anchorPrev = recipientInnovations.get(prevInnovation)!;
                    anchorNext = recipientInnovations.get(nextInnovation)!;
                    transplantNode = disjoint;
                    break;
                }
            }

            // Standard clone logic
            const nodesToCheck = [...this.inputNodes];
            const nodesChecked = new Set<BaseNode>();
            const oldNodes: BaseNode[] = [];
            const newNodes: BaseNode[] = [];
            const oldNewNode = new Map<BaseNode, BaseNode>();
            const newInputNodes: BaseNode[] = [];
            const newOutputNodes: BaseNode[] = [];
            let isValidFlag = true;

            // First pass
            while (nodesToCheck.length > 0) {
                const currentNode = nodesToCheck.shift()!;
                if (nodesChecked.has(currentNode)) continue;
                nodesChecked.add(currentNode);

                const newNode = currentNode.Clone();
                newNodes.push(newNode);
                oldNodes.push(currentNode);
                oldNewNode.set(currentNode, newNode);

                if (currentNode.next.length === 0) {
                    if (currentNode.GetNodeType() !== "Output") isValidFlag = false;
                    newOutputNodes.push(newNode);
                }
                if (currentNode.previous.length === 0) {
                    if (currentNode.GetNodeType() !== "Input") isValidFlag = false;
                    newInputNodes.push(newNode);
                }
                nodesToCheck.push(...currentNode.next);
            }

            // No transplant found? Just return the cloned recipient.
            if (!transplantNode || !anchorPrev || !anchorNext) {
                // Return exact clone without mutations
                for (let i = 0; i < newNodes.length; i++) {
                    const oldNode = oldNodes[i];
                    for (let oldNext of oldNode.next) {
                        newNodes[i].AddNext(oldNewNode.get(oldNext)!);
                    }
                }
                return { genome: new Genome(newInputNodes, newOutputNodes), nodes: newNodes, isValid: isValidFlag };
            }

            // Transplant logic
            const clonedTransplant = transplantNode.Clone();
            let inputAdapters: BaseNode[] = [];
            let outputAdapters: BaseNode[] = [];

            // Check dimensions to create adapters if necessary
            if (!anchorPrev.CheckCompability(clonedTransplant)) {
                const adapters = this.createAdapter(anchorPrev.GetOutputShape(), clonedTransplant.GetInputShape());
                if (adapters) inputAdapters = adapters;
                else return null; // Reject crossover if incompatible bounds
            }
            if (inputAdapters.length > 0) clonedTransplant.CheckCompability(inputAdapters[inputAdapters.length - 1]);

            if (!clonedTransplant.CheckCompability(anchorNext)) {
                const adapters = this.createAdapter(clonedTransplant.GetOutputShape(), anchorNext.GetInputShape());
                if (adapters) outputAdapters = adapters;
                else return null;
            }

            // Second pass: Restore connections, breaking connection between anchors (if exists) and inserting disjoint
            for (let i = 0; i < newNodes.length; i++) {
                const oldNode = oldNodes[i];
                const newNode = newNodes[i];

                if (oldNode.id === anchorPrev.id) {
                    // Check if there was an existing edge `anchorPrev -> anchorNext`. If so, break it.
                    let edgeExisted = false;
                    for (let oldNext of oldNode.next) {
                        if (oldNext.id === anchorNext.id) {
                            edgeExisted = true;
                            // Insert sequence
                            let currentTail = newNode;
                            for (let adapter of inputAdapters) {
                                currentTail.AddNext(adapter);
                                currentTail = adapter;
                            }
                            currentTail.AddNext(clonedTransplant);
                            currentTail = clonedTransplant;
                            for (let adapter of outputAdapters) {
                                currentTail.AddNext(adapter);
                                currentTail = adapter;
                            }
                            currentTail.AddNext(oldNewNode.get(anchorNext)!);
                        } else {
                            newNode.AddNext(oldNewNode.get(oldNext)!);
                        }
                    }

                    // If they weren't directly connected, just branch out!
                    if (!edgeExisted) {
                        for (let oldNext of oldNode.next) {
                            newNode.AddNext(oldNewNode.get(oldNext)!);
                        }

                        let currentTail = newNode;
                        for (let adapter of inputAdapters) {
                            currentTail.AddNext(adapter);
                            currentTail = adapter;
                        }
                        currentTail.AddNext(clonedTransplant);
                        currentTail = clonedTransplant;
                        for (let adapter of outputAdapters) {
                            currentTail.AddNext(adapter);
                            currentTail = adapter;
                        }
                        currentTail.AddNext(oldNewNode.get(anchorNext)!);
                    }
                } else {
                    for (let oldNext of oldNode.next) {
                        // Skip if this is anchorNext and we just hooked it up from anchorPrev? 
                        // No, anchorNext can receive connections from multiple places.
                        // We only skipped connecting `anchorPrev -> anchorNext` directly.
                        newNode.AddNext(oldNewNode.get(oldNext)!);
                    }
                }
            }

            newNodes.push(clonedTransplant, ...inputAdapters, ...outputAdapters);

            if (maxNodes !== undefined && newNodes.length > maxNodes) {
                return null;
            }

            console.log(`[BreedNeatStyle] Inserted disjoint node ${clonedTransplant.GetNodeType()} between anchors!`);

            return {
                genome: new Genome(newInputNodes, newOutputNodes),
                nodes: newNodes,
                isValid: isValidFlag
            };

        } catch (e) {
            console.error("NEAT crossover error", e);
            return null;
        }
    }

    public BreedMultiPoint(donor: Genome, maxNodes?: number): { genome: Genome, nodes: BaseNode[], isValid: boolean } | null {
        try {
            // Number of points to transplant (between 2 and 4, depending on donor size)
            const numPoints = Math.floor(Math.random() * 3) + 2;

            // Gather independent linear chains from Donor
            const donorNodes = donor.getAllNodes();
            const donorLinearChains: BaseNode[][] = [];
            const visitedDonor = new Set<string>();

            // Small copy of _getRandomSubgenome logic to find multiple chains
            donorNodes.forEach(node => {
                if (!visitedDonor.has(node.id) && node.previous.length === 1 && node.next.length === 1) {
                    const chain: BaseNode[] = [];
                    let current: BaseNode | null = node;

                    while (current && current.previous.length === 1 && current.next.length === 1) {
                        const prev: BaseNode = current.previous[0];
                        if (visitedDonor.has(prev.id) || prev.previous.length !== 1 || prev.next.length !== 1) break;
                        current = prev;
                    }

                    while (current && current.previous.length === 1 && current.next.length === 1 && !visitedDonor.has(current.id)) {
                        visitedDonor.add(current.id);
                        chain.push(current);
                        current = current.next[0];
                    }

                    if (chain.length >= 1) {
                        donorLinearChains.push(chain);
                    }
                }
            });

            if (donorLinearChains.length === 0) return null;

            // Pick up to `numPoints` random disjoint sequences
            const shuffledChains = [...donorLinearChains].sort(() => Math.random() - 0.5);
            const selectedDonorSubgraphs = shuffledChains.slice(0, numPoints).map(chain => {
                // Take a random chunk from the chain
                const maxStart = chain.length - 1;
                const start = maxStart > 0 ? Math.floor(Math.random() * maxStart) : 0;
                const len = Math.floor(Math.random() * (chain.length - start)) + 1;
                return chain.slice(start, start + len);
            });

            // Standard clone logic for Recipient (Base Graph)
            const nodesToCheck = [...this.inputNodes];
            const nodesChecked = new Set<BaseNode>();
            const oldNodes: BaseNode[] = [];
            const newNodes: BaseNode[] = [];
            const oldNewNode = new Map<BaseNode, BaseNode>();
            const newInputNodes: BaseNode[] = [];
            const newOutputNodes: BaseNode[] = [];
            let isValidFlag = true;

            while (nodesToCheck.length > 0) {
                const currentNode = nodesToCheck.shift()!;
                if (nodesChecked.has(currentNode)) continue;
                nodesChecked.add(currentNode);

                const newNode = currentNode.Clone();
                newNodes.push(newNode);
                oldNodes.push(currentNode);
                oldNewNode.set(currentNode, newNode);

                if (currentNode.next.length === 0) {
                    if (currentNode.GetNodeType() !== "Output") isValidFlag = false;
                    newOutputNodes.push(newNode);
                }
                if (currentNode.previous.length === 0) {
                    if (currentNode.GetNodeType() !== "Input") isValidFlag = false;
                    newInputNodes.push(newNode);
                }
                nodesToCheck.push(...currentNode.next);
            }

            // Second pass: Restore connections
            for (let i = 0; i < newNodes.length; i++) {
                const oldNode = oldNodes[i];
                const newNode = newNodes[i];
                for (const oldNext of oldNode.next) {
                    newNode.AddNext(oldNewNode.get(oldNext)!);
                }
            }

            // For each valid subgraph, find a random insertion edge in the fully cloned graph and splice it
            for (const donorGraphBase of selectedDonorSubgraphs) {
                // Clone the donor chunk so it gets fresh IPs and UUIDs
                const clonedDonorChunk: BaseNode[] = [donorGraphBase[0].Clone()];
                for (let j = 1; j < donorGraphBase.length; j++) {
                    clonedDonorChunk.push(donorGraphBase[j].Clone());
                    clonedDonorChunk[j - 1].AddNext(clonedDonorChunk[j]);
                }

                const subIn = clonedDonorChunk[0];
                const subOut = clonedDonorChunk[clonedDonorChunk.length - 1];

                // Gather live edges in our working newNodes
                const liveEdges: { from: BaseNode, to: BaseNode }[] = [];
                for (const liveNode of newNodes) {
                    for (const nextNode of liveNode.next) {
                        liveEdges.push({ from: liveNode, to: nextNode });
                    }
                }

                if (liveEdges.length === 0) break;

                // Pick a random edge
                const edge = liveEdges[Math.floor(Math.random() * liveEdges.length)];
                const fromNode = edge.from;
                const toNode = edge.to;

                let inAdapters: BaseNode[] = [];
                let outAdapters: BaseNode[] = [];

                if (!fromNode.CheckCompability(subIn)) {
                    const ad = this.createAdapter(fromNode.GetOutputShape(), subIn.GetInputShape());
                    if (ad) inAdapters = ad;
                    else continue; // Cannot connect
                }
                if (inAdapters.length > 0) subIn.CheckCompability(inAdapters[inAdapters.length - 1]);

                if (!subOut.CheckCompability(toNode)) {
                    const ad = this.createAdapter(subOut.GetOutputShape(), toNode.GetInputShape());
                    if (ad) outAdapters = ad;
                    else continue;
                }

                // Break the edge fromNode -> toNode
                fromNode.RemoveNext(toNode);

                // Re-route: fromNode -> (inAdapters) -> subIn ... subOut -> (outAdapters) -> toNode
                let tail = fromNode;
                for (const ad of inAdapters) {
                    tail.AddNext(ad);
                    tail = ad;
                }
                tail.AddNext(subIn);

                tail = subOut;
                for (const ad of outAdapters) {
                    tail.AddNext(ad);
                    tail = ad;
                }
                tail.AddNext(toNode);

                newNodes.push(...clonedDonorChunk, ...inAdapters, ...outAdapters);
            }

            if (maxNodes !== undefined && newNodes.length > maxNodes) {
                return null;
            }

            return {
                genome: new Genome(newInputNodes, newOutputNodes),
                nodes: newNodes,
                isValid: isValidFlag
            };

        } catch (e) {
            console.error("Multi-point crossover error", e);
            return null;
        }
    }
}