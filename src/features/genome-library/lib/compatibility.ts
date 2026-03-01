import { GenomeLibraryEntry, CompatibilityStatus } from '../model/store';
import { DataStream, DataType } from '../../dataset-manager/model/store';

/**
 * Get the tensor dimensionality for a given DataType.
 * Image = 3 (H,W,C), Vector/Categorical/Text = 1
 */
function getDataTypeDims(dataType: DataType): number {
    switch (dataType) {
        case 'Image':
            return 3; // H, W, C
        case 'Vector':
        case 'Categorical':
        case 'Text':
            return 1;
        default:
            return 1;
    }
}

/**
 * Checks compatibility of a genome library entry against a set of data streams.
 * 
 * - 'compatible': dims match exactly for all inputs and outputs
 * - 'incompatible': wrong number of inputs/outputs or wrong dimensionality
 */
export function checkCompatibility(
    entry: GenomeLibraryEntry,
    streams: DataStream[]
): CompatibilityStatus {
    const inputStreams = streams.filter(s => s.role === 'Input');
    const targetStreams = streams.filter(s => s.role === 'Target');

    // Check input/output count
    if (entry.inputDims.length !== inputStreams.length) return 'incompatible';
    if (entry.outputDims.length !== targetStreams.length) return 'incompatible';

    // Check dimensionality of each input
    for (let i = 0; i < inputStreams.length; i++) {
        const expectedDims = getDataTypeDims(inputStreams[i].dataType);
        if (entry.inputDims[i] !== expectedDims) return 'incompatible';
    }

    // Check dimensionality of each output
    for (let i = 0; i < targetStreams.length; i++) {
        const expectedDims = getDataTypeDims(targetStreams[i].dataType);
        if (entry.outputDims[i] !== expectedDims) return 'incompatible';
    }

    return 'compatible';
}
