import { invoke } from "@tauri-apps/api/core"

export const loadGenomeApi = async (): Promise<string> => {
    return await invoke<string>("load_genome", {})
}