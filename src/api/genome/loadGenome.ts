import { invoke } from "@tauri-apps/api/core"

export const loadGenomeApi = (loadGenomeCallback: (genomeStr: String) => void) => {
    invoke<String>("load_genome", {})
        .then(res => loadGenomeCallback(res))
        .catch(e => console.log(e));
}