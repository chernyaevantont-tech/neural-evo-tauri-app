import { invoke } from "@tauri-apps/api/core"

export const loadGenomeApi = (pathStr: String, loadGenomeCallback: (genomeStr: String) => void) => {
    invoke<String>("load_genome", {pathStr})
        .then(res => loadGenomeCallback(res))
        .catch(e => console.log(e));
}