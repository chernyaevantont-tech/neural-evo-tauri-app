import { BaseNode } from "./nodes/base_node"

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

    public GetRandomSubgenome(): string[] {
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

        return subchain.map(node => node.id);
    }

    
}