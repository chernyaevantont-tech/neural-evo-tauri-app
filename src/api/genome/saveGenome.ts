import { invoke } from "@tauri-apps/api/core"

export const saveGenomeApi = (genomeStr: String, saveGenomeCallback: () => void) => {
    invoke("save_genome", { genomeStr })
        .then(saveGenomeCallback)
        .catch(e => console.log(e));
}