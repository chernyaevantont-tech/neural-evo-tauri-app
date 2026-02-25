import { invoke } from "@tauri-apps/api/core";

export const testTrainGenomeAPI = async (genomeStr: string) => {
    return invoke('test_neural_net_training', { genomeStr });
}

export const testTrainOnImageFolderAPI = async (genomeStr: string) => {
    return invoke('test_train_on_image_folder', { genomeStr });
}