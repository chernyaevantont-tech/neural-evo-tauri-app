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
            const inputCompatible = subgenomeInputNode.CheckCompabilityDisconnected(fromNode);

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
            const outputCompatible = toNode.CheckCompabilityDisconnected(subgenomeOutputNode);

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

        // Случай 2: 3D -> 1D (нужен Flatten + возможно Dense)
        if (fromShape.length === 3 && toShape.length === 1) {
            adapters.push(new FlattenNode());

            // Вычисляем размер после flatten
            const flattenedSize = fromShape[0] * fromShape[1] * fromShape[2];

            // Если размеры не совпадают, добавляем Dense слой
            if (flattenedSize !== toShape[0]) {
                adapters.push(new DenseNode(toShape[0], 'relu', true));
            }

            return adapters;
        }

        // Случай 3: 1D -> 1D (различное количество нейронов)
        if (fromShape.length === 1 && toShape.length === 1) {
            const [fromUnits] = fromShape;
            const [toUnits] = toShape;

            if (fromUnits !== toUnits) {
                adapters.push(new DenseNode(toUnits, 'relu', true));
                return adapters;
            }

            return null; // Формы совпадают
        }

        // Случай 4: 1D -> 3D (очень сложно, не поддерживается)
        if (fromShape.length === 1 && toShape.length === 3) {
            return null; // Reshape из 1D в 3D пока не поддерживается
        }

        // Неизвестный случай
        return null;
    }

    public Breed(genome: Genome): { genome: Genome, nodes: BaseNode[], isValid: boolean} | null {
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
                    console.log(oldNextNode, oldNodes[i].next, oldNewNode, oldNewNode.get(oldNextNode))
                    newNodes[i].AddNext(oldNewNode.get(oldNextNode)!);
                }
            }
        }

        newNodes.push(
            ...fromSubgenome,
            ...insertion.inputAdapterNodes,
            ...insertion.outputAdapterNodes
        );

        return {
            genome: new Genome(newInputNodes, newOutputNodes),
            nodes: newNodes,
            isValid: isValidFlag
        }
    }
}