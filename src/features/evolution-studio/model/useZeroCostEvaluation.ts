/**
 * useZeroCostEvaluation Hook
 * 
 * Implements two-stage evaluation strategy using zero-cost proxies
 * for fast architecture scoring without training.
 */

import { invoke } from '@tauri-apps/api/core';

export interface ZeroCostMetrics {
  synflow: number;
  normalized_score: number;
  strategy_decision: 'skip' | 'partial_train' | 'full_train';
}

export interface ZeroCostConfig {
  enabled: boolean;
  strategy: 'two-stage' | 'early-stopping';
  fastPassThreshold: number;
  partialTrainingEpochs: number;
  useVoting: boolean;
}

/**
 * Compute zero-cost proxy score for an architecture
 * 
 * This performs a single forward-backward pass on a sample batch
 * to estimate architecture quality without training.
 */
export const computeZeroCostScore = async (
  genomeJSON: string,
  config: ZeroCostConfig
): Promise<ZeroCostMetrics> => {
  try {
    // Prepare config for serialization
    const configDto = {
      enabled: config.enabled,
      strategy: config.strategy,
      fast_pass_threshold: config.fastPassThreshold,
      partial_training_epochs: config.partialTrainingEpochs,
      use_voting: config.useVoting,
    };

    const result = await invoke<any>('compute_zero_cost_score', {
      genomeJson: genomeJSON,
      configJson: JSON.stringify(configDto),
    });

    // Normalize the result
    return {
      synflow: result.synflow ?? 5.0,
      normalized_score: result.normalized_score ?? 0.5,
      strategy_decision: (result.strategy_decision?.replace(/_/g, '_') ?? 'full_train') as 'skip' | 'partial_train' | 'full_train',
    };
  } catch (error) {
    console.error('Error computing zero-cost score:', error);
    // Fallback: return neutral score if computation fails
    return {
      synflow: 5.0,
      normalized_score: 0.5,
      strategy_decision: 'full_train',
    };
  }
};

/**
 * Get training epochs recommendation based on zero-cost score
 */
export const getRecommendedEpochs = (
  metrics: ZeroCostMetrics,
  fullEpochs: number,
  partialEpochs: number
): number | null => {
  switch (metrics.strategy_decision) {
    case 'skip':
      return null;
    case 'partial_train':
      return partialEpochs;
    case 'full_train':
      return fullEpochs;
  }
};

/**
 * Calculate fitness score combining zero-cost proxy and actual accuracy
 * 
 * Two-stage strategy:
 * - If score > threshold: fitness = accuracy (after full training)
 * - Otherwise: fitness = proxy_score (no training)
 */
export const calculateHybridFitness = (
  zeroCostScore: number,
  accuracy: number | null,
  config: ZeroCostConfig
): number => {
  if (accuracy !== null) {
    // Architecture was trained, use actual accuracy
    // Weight the result: partial weight to proxy for diversity
    return 0.7 * accuracy + 0.3 * zeroCostScore;
  } else {
    // Architecture was not trained, use proxy score
    return zeroCostScore;
  }
};

/**
 * Estimate total training time savings with zero-cost proxies
 */
export const estimateTimeSavings = (
  populationSize: number,
  fullTrainingTime: number, // minutes per architecture
  avgProxyScore: number, // average normalized score (0-1)
  config: ZeroCostConfig
): { savedTime: number; speedup: number } => {
  // Estimate: (1 - avgProxyScore) * populationSize will be skipped/quick-trained
  const quickTrainRatio = Math.max(0, 1 - avgProxyScore * 0.7); // 70% of high-scorers get full training
  const avgQuickTrainingTime = (fullTrainingTime * config.partialTrainingEpochs) / 100; // Rough estimate

  const timeWithProxy =
    populationSize *
    (avgProxyScore * fullTrainingTime + (1 - avgProxyScore) * avgQuickTrainingTime);
  const timeWithout = populationSize * fullTrainingTime;

  return {
    savedTime: timeWithout - timeWithProxy,
    speedup: timeWithout / timeWithProxy,
  };
};
