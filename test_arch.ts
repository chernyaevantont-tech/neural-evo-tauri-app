import { generateRandomArchitecture } from "./src/entities/canvas-genome/lib/randomArchitectureGenerator";
import { Genome } from "./src/entities/canvas-genome/model/genome";

try {
    const genome = generateRandomArchitecture([50, 4], [2], {
        maxDepth: 8,
        useAttention: false,
        dataTypeHint: "TemporalSequence"
    });
    
    const nodes = genome.getAllNodes();
    console.log("Feasible?", Genome.isGenomeFeasible(nodes));
} catch (e) {
    console.error(e);
}
