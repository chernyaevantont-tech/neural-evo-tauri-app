import React, { useState, useEffect } from 'react';
import { BaseNode } from '../../evo/nodes/base_node';
import { InputNode } from '../../evo/nodes/layers/input_node';
import { DenseNode } from '../../evo/nodes/layers/dense_node';
import { Conv2DNode } from '../../evo/nodes/layers/conv_node';
import { PoolingNode } from '../../evo/nodes/layers/pooling_node';
import { AddNode } from '../../evo/nodes/merge/add_node';
import { Concat2DNode } from '../../evo/nodes/merge/concatinate_2d_node';
import { ActivationFunction, KernelSize, PoolType } from '../../evo/nodes/types';
import { FlattenNode } from '../../evo/nodes/layers/flatten_node';
import { OutputNode } from '../../evo/nodes/layers/output_node';
import { Modal, Button } from '../../shared/ui';
import styles from './NodeConfigForm.module.css';

interface NodeConfigFormProps {
  nodeType: string;
  existingNode?: BaseNode;
  onSave: (node: BaseNode) => void;
  onCancel: () => void;
}

export const NodeConfigForm: React.FC<NodeConfigFormProps> = ({
  nodeType,
  existingNode,
  onSave,
  onCancel,
}) => {
  const [inputShapeLength, setInputShapeLength] = useState<number>(1);
  const [inputShape, setInputShape] = useState<number[]>([1]);
  const [units, setUnits] = useState<number>(128);
  const [activation, setActivation] = useState<ActivationFunction>('relu');
  const [useBias, setUseBias] = useState<boolean>(true);
  const [filters, setFilters] = useState<number>(32);
  const [kernelH, setKernelH] = useState<number>(3);
  const [kernelW, setKernelW] = useState<number>(3);
  const [stride, setStride] = useState<number>(1);
  const [padding, setPadding] = useState<number>(0);
  const [dilation, setDilation] = useState<number>(1);
  const [convUseBias, setConvUseBias] = useState<boolean>(true);
  const [poolType, setPoolType] = useState<PoolType>('max');
  const [poolKernelH, setPoolKernelH] = useState<number>(2);
  const [poolKernelW, setPoolKernelW] = useState<number>(2);
  const [poolStride, setPoolStride] = useState<number>(2);
  const [poolPadding, setPoolPadding] = useState<number>(0);
  const [outputShapeLength, setOutputShapeLength] = useState<number>(1);
  const [outputShape, setOutputShape] = useState<number[]>([1]);

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
      setInputShapeLength(shape.length);
      setInputShape(shape);
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
    } else if (node instanceof OutputNode) {
      const shape = params.output_shape;
      setOutputShapeLength(shape.length);
      setOutputShape(shape);
    }
  };

  const handleSave = () => {
    try {
      let newNode: BaseNode;

      switch (nodeType) {
        case 'Input':
          newNode = new InputNode(inputShape);
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
        case 'Output':
          newNode = new OutputNode(outputShape);
          break;
        default:
          throw new Error(`Unknown node type: ${nodeType}`);
      }

      if (existingNode && !existingNode.next.every((val: BaseNode) => val.CheckCompabilityDisconnected(newNode))) {
        alert('Edited node is incompatible');
        return;
      }

      onSave(newNode);
    } catch (error) {
      alert(`Error creating node: ${error}`);
    }
  };

  const FormField: React.FC<{ label: string; children: React.ReactNode }> = ({ label, children }) => (
    <div className={styles.field}>
      <label className={styles.label}>{label}</label>
      {children}
    </div>
  );

  const renderInputConfig = () => (
    <>
      <FormField label="Shape Length">
        <input
          className={styles.input}
          type="number"
          value={inputShapeLength}
          onChange={(e) => {
            const newLen = parseInt(e.target.value) || 0;
            setInputShapeLength(newLen);
            const minLen = Math.min(newLen, inputShape.length);
            const newShape = new Array(newLen).fill(1, minLen);
            for (let i = 0; i < minLen; i++) newShape[i] = inputShape[i];
            setInputShape(newShape);
          }}
          min="1"
        />
      </FormField>
      {inputShape.map((d, i) => (
        <FormField key={i} label={`Dimension ${i + 1}`}>
          <input
            className={styles.input}
            type="number"
            value={d}
            onChange={(e) => {
              const newShape = [...inputShape];
              newShape[i] = parseInt(e.target.value) || 1;
              setInputShape(newShape);
            }}
            min="1"
          />
        </FormField>
      ))}
    </>
  );

  const renderOutputConfig = () => (
    <>
      <FormField label="Shape Length">
        <input
          className={styles.input}
          type="number"
          value={outputShapeLength}
          onChange={(e) => {
            const newLen = parseInt(e.target.value) || 0;
            setOutputShapeLength(newLen);
            const minLen = Math.min(newLen, outputShape.length);
            const newShape = new Array(newLen).fill(1, minLen);
            for (let i = 0; i < minLen; i++) newShape[i] = outputShape[i];
            setOutputShape(newShape);
          }}
          min="1"
        />
      </FormField>
      {outputShape.map((d, i) => (
        <FormField key={i} label={`Dimension ${i + 1}`}>
          <input
            className={styles.input}
            type="number"
            value={d}
            onChange={(e) => {
              const newShape = [...outputShape];
              newShape[i] = parseInt(e.target.value) || 1;
              setOutputShape(newShape);
            }}
            min="1"
          />
        </FormField>
      ))}
    </>
  );

  const renderDenseConfig = () => (
    <>
      <FormField label="Units">
        <input
          className={styles.input}
          type="number"
          value={units}
          onChange={(e) => setUnits(parseInt(e.target.value) || 1)}
          min="1"
        />
      </FormField>
      <FormField label="Activation">
        <select className={styles.select} value={activation} onChange={(e) => setActivation(e.target.value as ActivationFunction)}>
          <option value="relu">ReLU</option>
          <option value="leaky_relu">Leaky ReLU</option>
          <option value="softmax">Softmax</option>
        </select>
      </FormField>
      <FormField label="Use Bias">
        <input type="checkbox" checked={useBias} onChange={(e) => setUseBias(e.target.checked)} />
      </FormField>
    </>
  );

  const renderConv2DConfig = () => (
    <>
      <FormField label="Filters">
        <input className={styles.input} type="number" value={filters} onChange={(e) => setFilters(parseInt(e.target.value) || 1)} min="1" />
      </FormField>
      <FormField label="Kernel Height">
        <input className={styles.input} type="number" value={kernelH} onChange={(e) => setKernelH(parseInt(e.target.value) || 1)} min="1" />
      </FormField>
      <FormField label="Kernel Width">
        <input className={styles.input} type="number" value={kernelW} onChange={(e) => setKernelW(parseInt(e.target.value) || 1)} min="1" />
      </FormField>
      <FormField label="Stride">
        <input className={styles.input} type="number" value={stride} onChange={(e) => setStride(parseInt(e.target.value) || 1)} min="1" />
      </FormField>
      <FormField label="Padding">
        <input className={styles.input} type="number" value={padding} onChange={(e) => setPadding(parseInt(e.target.value) || 0)} min="0" />
      </FormField>
      <FormField label="Dilation">
        <input className={styles.input} type="number" value={dilation} onChange={(e) => setDilation(parseInt(e.target.value) || 1)} min="1" />
      </FormField>
      <FormField label="Use Bias">
        <input type="checkbox" checked={convUseBias} onChange={(e) => setConvUseBias(e.target.checked)} />
      </FormField>
    </>
  );

  const renderPoolingConfig = () => (
    <>
      <FormField label="Pool Type">
        <select className={styles.select} value={poolType} onChange={(e) => setPoolType(e.target.value as PoolType)}>
          <option value="max">Max</option>
          <option value="avg">Average</option>
        </select>
      </FormField>
      <FormField label="Kernel Height">
        <input className={styles.input} type="number" value={poolKernelH} onChange={(e) => setPoolKernelH(parseInt(e.target.value) || 1)} min="1" />
      </FormField>
      <FormField label="Kernel Width">
        <input className={styles.input} type="number" value={poolKernelW} onChange={(e) => setPoolKernelW(parseInt(e.target.value) || 1)} min="1" />
      </FormField>
      <FormField label="Stride">
        <input className={styles.input} type="number" value={poolStride} onChange={(e) => setPoolStride(parseInt(e.target.value) || 1)} min="1" />
      </FormField>
      <FormField label="Padding">
        <input className={styles.input} type="number" value={poolPadding} onChange={(e) => setPoolPadding(parseInt(e.target.value) || 0)} min="0" />
      </FormField>
    </>
  );

  return (
    <Modal isOpen={true} onClose={onCancel} title={`${existingNode ? 'Edit' : 'Add'} ${nodeType} Node`} maxWidth="500px">
      <div className={styles.form}>
        {nodeType === 'Input' && renderInputConfig()}
        {nodeType === 'Output' && renderOutputConfig()}
        {nodeType === 'Dense' && renderDenseConfig()}
        {nodeType === 'Conv2D' && renderConv2DConfig()}
        {nodeType === 'Pooling' && renderPoolingConfig()}
        {(nodeType === 'Add' || nodeType === 'Concat2D' || nodeType === 'Flatten') && (
          <p className={styles.nodeConfigText}>
            This node has no configurable parameters.
          </p>
        )}

        <div className={styles.buttons}>
          <Button onClick={handleSave} variant="primary">
            {existingNode ? 'Update' : 'Create'}
          </Button>
          <Button onClick={onCancel} variant="secondary">
            Cancel
          </Button>
        </div>
      </div>
    </Modal>
  );
};

