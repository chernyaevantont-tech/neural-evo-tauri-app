import { loadGenomeApi } from '../../api/genome/loadGenome';
import { saveGenomeApi } from '../../api/genome/saveGenome';
import { loadGenome } from '../../saver/loadGenome';
import { saveGenome } from '../../saver/saveGenome';
import { Genome } from '../../evo/genome';

export const loadGenomeFromFile = (
  onSuccess: (data: ReturnType<typeof loadGenome>) => void
) => {
  loadGenomeApi((genomeStr) => {
    const result = loadGenome(genomeStr as string);
    onSuccess(result);
  });
};

export const saveGenomeToFile = (
  genome: Genome,
  onSuccess: () => void
) => {
  const genomeStr = saveGenome(genome);
  saveGenomeApi(genomeStr, onSuccess);
};
