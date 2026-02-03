import { BaseNode } from "./nodes/base_node"
import { Subgenome } from "./types";

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
        // Получаем все узлы после входных, которые имеют продолжение
        const startNodes = this.inputNodes.map(x => x.next).flat().filter(x => x.next.length > 0)
        
        if (startNodes.length === 0) {
            throw new Error("No valid start nodes found")
        }

        // Выбираем случайный стартовый узел
        let currentNode = startNodes[Math.floor(Math.random() * startNodes.length)]
        const pathNodes: BaseNode[] = [currentNode]

        // Идём по случайному линейному пути, выбирая одного следующего на каждом шаге
        while (currentNode.next.length > 0) {
            // Выбираем случайного следующего узла
            const nextNode = currentNode.next[Math.floor(Math.random() * currentNode.next.length)]
            pathNodes.push(nextNode)
            currentNode = nextNode

            // С вероятностью 0.3 останавливаемся (чтобы не всегда брать весь путь до конца)
            if (Math.random() < 0.3 && pathNodes.length >= 2) {
                break
            }
        }

        // Если путь слишком короткий, берём как минимум 2 узла
        if (pathNodes.length < 2) {
            // Если нет продолжения, возвращаем минимальный подграф из одного узла
            return [pathNodes[0].id]
            
        }

        return pathNodes.map(node => node.id);
        
    }
}