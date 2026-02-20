import { invoke } from "@tauri-apps/api/core"

export const saveGenomeApi = async (genomeStr: string) => {
    await invoke("save_genome", { genomeStr })
}