import { invoke } from "@tauri-apps/api/core";

export const testTrainGenomeAPI = async (genomeStr: string) => {
    return invoke('test_neural_net_training', { genomeStr });
}