import { invoke } from "@tauri-apps/api/core"

export const saveGenomeApi = (genomeStr: String, pathStr: String, saveGenomeCallback: () => void) => {
    invoke("save_genome", { genomeStr, pathStr })
        .then(saveGenomeCallback)
        .catch(e => console.log(e));
}