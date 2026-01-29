import React, { useState, useEffect } from 'react';
import { BaseNode } from '../evo/nodes/base_node';
import { InputNode } from '../evo/nodes/layers/input_node';
import { DenseNode } from '../evo/nodes/layers/dense_node';
import { Conv2DNode } from '../evo/nodes/layers/conv_node';
import { PoolingNode } from '../evo/nodes/layers/pooling_node';
import { AddNode } from '../evo/nodes/merge/add_node';
import { Concat2DNode } from '../evo/nodes/merge/concatinate_2d_node';
import { ActivationFunction, KernelSize, PoolType } from '../evo/nodes/types';
import { FlattenNode } from '../evo/nodes/layers/flatten_node';

interface NodeConfigPanelProps {
  nodeType: string;
  existingNode?: BaseNode;
  onSave: (node: BaseNode) => void;
  onCancel: () => void;
}

export const NodeConfigPanel: React.FC<NodeConfigPanelProps> = ({
  nodeType,
  existingNode,
  onSave,
  onCancel,
}) => {
  // InputNode parameters
  const [d1, setD1] = useState<number>(28);
  const [d2, setD2] = useState<number | undefined>(28);
  const [d3, setD3] = useState<number | undefined>(undefined);

  // DenseNode parameters
  const [units, setUnits] = useState<number>(128);
  const [activation, setActivation] = useState<ActivationFunction>('relu');
  const [useBias, setUseBias] = useState<boolean>(true);

  // Conv2DNode parameters
  const [filters, setFilters] = useState<number>(32);
  const [kernelH, setKernelH] = useState<number>(3);
  const [kernelW, setKernelW] = useState<number>(3);
  const [stride, setStride] = useState<number>(1);
  const [padding, setPadding] = useState<number>(0);
  const [dilation, setDilation] = useState<number>(1);
  const [convUseBias, setConvUseBias] = useState<boolean>(true);

  // PoolingNode parameters
  const [poolType, setPoolType] = useState<PoolType>('max');
  const [poolKernelH, setPoolKernelH] = useState<number>(2);
  const [poolKernelW, setPoolKernelW] = useState<number>(2);
  const [poolStride, setPoolStride] = useState<number>(2);
  const [poolPadding, setPoolPadding] = useState<number>(0);

  useEffect(() => {
    if (existingNode) {
      loadNodeParameters(existingNode);
    }
  }, [existingNode]);

  const loadNodeParameters = (node: BaseNode) => {
    const info = JSON.parse(node.GetInfo() as string);
    const params = info.params;

    if (node instanceof InputNode) {
      const shape = params.output_shape;
      setD1(shape[0]);
      setD2(shape[1]);
      setD3(shape[2]);
    } else if (node instanceof DenseNode) {
      setUnits(params.units);
      setActivation(params.activation);
      setUseBias(params.use_bias);
    } else if (node instanceof Conv2DNode) {
      setFilters(params.filters);
      setKernelH(params.kernel_size.h);
      setKernelW(params.kernel_size.w);
      setStride(params.stride || 1);
      setPadding(params.padding);
      setDilation(params.delation);
      setConvUseBias(params.use_bias);
    } else if (node instanceof PoolingNode) {
      setPoolType(params.pool_type);
      setPoolKernelH(params.kernel_size.h);
      setPoolKernelW(params.kernel_size.w);
      setPoolStride(params.stride);
      setPoolPadding(params.padding);
    }
  };

  const handleSave = () => {
    try {
      let newNode: BaseNode;

      switch (nodeType) {
        case 'Input':
          newNode = new InputNode(d1, d2, d3);
          break;
        case 'Dense':
          newNode = new DenseNode(units, activation, useBias);
          break;
        case 'Conv2D':
          const kernelSize: KernelSize = { h: kernelH, w: kernelW };
          newNode = new Conv2DNode(filters, kernelSize, stride, padding, dilation, convUseBias);
          break;
        case 'Pooling':
          const poolKernelSize: KernelSize = { h: poolKernelH, w: poolKernelW };
          newNode = new PoolingNode(poolType, poolKernelSize, poolStride, poolPadding);
          break;
        case 'Flatten':
          newNode = new FlattenNode();
          break;
        case 'Add':
          newNode = new AddNode();
          break;
        case 'Concat2D':
          newNode = new Concat2DNode();
          break;
        default:
          throw new Error(`Unknown node type: ${nodeType}`);
      }

      onSave(newNode);
    } catch (error) {
      alert(`Error creating node: ${error}`);
    }
  };

  const renderInputNodeConfig = () => (
    <div className="config-section">
      <h4>Input Shape</h4>
      <div className="config-row">
        <label>Dimension 1 (required):</label>
        <input
          type="number"
          value={d1}
          onChange={(e) => setD1(parseInt(e.target.value) || 0)}
          min="1"
        />
      </div>
      <div className="config-row">
        <label>Dimension 2 (optional):</label>
        <input
          type="number"
          value={d2 || ''}
          onChange={(e) => setD2(e.target.value ? parseInt(e.target.value) : undefined)}
          min="1"
          placeholder="Optional"
        />
      </div>
      <div className="config-row">
        <label>Dimension 3 (optional):</label>
        <input
          type="number"
          value={d3 || ''}
          onChange={(e) => setD3(e.target.value ? parseInt(e.target.value) : undefined)}
          min="1"
          placeholder="Optional"
        />
      </div>
    </div>
  );

  const renderDenseNodeConfig = () => (
    <div className="config-section">
      <h4>Dense Layer</h4>
      <div className="config-row">
        <label>Units:</label>
        <input
          type="number"
          value={units}
          onChange={(e) => setUnits(parseInt(e.target.value) || 1)}
          min="1"
        />
      </div>
      <div className="config-row">
        <label>Activation:</label>
        <select value={activation} onChange={(e) => setActivation(e.target.value as ActivationFunction)}>
          <option value="relu">ReLU</option>
          <option value="leaky_relu">Leaky ReLU</option>
          <option value="softmax">Softmax</option>
        </select>
      </div>
      <div className="config-row">
        <label>Use Bias:</label>
        <input
          type="checkbox"
          checked={useBias}
          onChange={(e) => setUseBias(e.target.checked)}
        />
      </div>
    </div>
  );

  const renderConv2DNodeConfig = () => (
    <div className="config-section">
      <h4>Conv2D Layer</h4>
      <div className="config-row">
        <label>Filters:</label>
        <input
          type="number"
          value={filters}
          onChange={(e) => setFilters(parseInt(e.target.value) || 1)}
          min="1"
        />
      </div>
      <div className="config-row">
        <label>Kernel Height:</label>
        <input
          type="number"
          value={kernelH}
          onChange={(e) => setKernelH(parseInt(e.target.value) || 1)}
          min="1"
        />
      </div>
      <div className="config-row">
        <label>Kernel Width:</label>
        <input
          type="number"
          value={kernelW}
          onChange={(e) => setKernelW(parseInt(e.target.value) || 1)}
          min="1"
        />
      </div>
      <div className="config-row">
        <label>Stride:</label>
        <input
          type="number"
          value={stride}
          onChange={(e) => setStride(parseInt(e.target.value) || 1)}
          min="1"
        />
      </div>
      <div className="config-row">
        <label>Padding:</label>
        <input
          type="number"
          value={padding}
          onChange={(e) => setPadding(parseInt(e.target.value) || 0)}
          min="0"
        />
      </div>
      <div className="config-row">
        <label>Dilation:</label>
        <input
          type="number"
          value={dilation}
          onChange={(e) => setDilation(parseInt(e.target.value) || 1)}
          min="1"
        />
      </div>
      <div className="config-row">
        <label>Use Bias:</label>
        <input
          type="checkbox"
          checked={convUseBias}
          onChange={(e) => setConvUseBias(e.target.checked)}
        />
      </div>
    </div>
  );

  const renderPoolingNodeConfig = () => (
    <div className="config-section">
      <h4>Pooling Layer</h4>
      <div className="config-row">
        <label>Pool Type:</label>
        <select value={poolType} onChange={(e) => setPoolType(e.target.value as PoolType)}>
          <option value="max">Max</option>
          <option value="avg">Average</option>
        </select>
      </div>
      <div className="config-row">
        <label>Kernel Height:</label>
        <input
          type="number"
          value={poolKernelH}
          onChange={(e) => setPoolKernelH(parseInt(e.target.value) || 1)}
          min="1"
        />
      </div>
      <div className="config-row">
        <label>Kernel Width:</label>
        <input
          type="number"
          value={poolKernelW}
          onChange={(e) => setPoolKernelW(parseInt(e.target.value) || 1)}
          min="1"
        />
      </div>
      <div className="config-row">
        <label>Stride:</label>
        <input
          type="number"
          value={poolStride}
          onChange={(e) => setPoolStride(parseInt(e.target.value) || 1)}
          min="1"
        />
      </div>
      <div className="config-row">
        <label>Padding:</label>
        <input
          type="number"
          value={poolPadding}
          onChange={(e) => setPoolPadding(parseInt(e.target.value) || 0)}
          min="0"
        />
      </div>
    </div>
  );

  const renderMergeNodeConfig = () => (
    <div className="config-section">
      <h4>{nodeType} Node</h4>
      <p>This node has no configurable parameters.</p>
    </div>
  );

  const renderFlattenNodeConfig = () => (
    <div className="config-section">
      <h4>{nodeType} Flatten</h4>
      <p>This node has no configurable parameters.</p>
    </div>
  );

  return (
    <div className="config-panel">
      <h3>{existingNode ? 'Edit Node' : 'Add Node'} - {nodeType}</h3>
      
      {nodeType === 'Input' && renderInputNodeConfig()}
      {nodeType === 'Dense' && renderDenseNodeConfig()}
      {nodeType === 'Conv2D' && renderConv2DNodeConfig()}
      {nodeType === 'Pooling' && renderPoolingNodeConfig()}
      {(nodeType === 'Add' || nodeType === 'Concat2D') && renderMergeNodeConfig()}
      {nodeType === 'Flatten' && renderFlattenNodeConfig()}

      <div className="config-buttons">
        <button onClick={handleSave} className="btn-save">
          {existingNode ? 'Update' : 'Create'}
        </button>
        <button onClick={onCancel} className="btn-cancel">
          Cancel
        </button>
      </div>
    </div>
  );
};
