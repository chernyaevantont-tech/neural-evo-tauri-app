import { BaseNode } from "../model/nodes/base_node";
import { Position } from "../model/types";

export const calculateLayoutForNewGraph = (
    nodesToLayout: BaseNode[],
    canvasWidth: number,
    canvasHeight: number,
    translateX: number,
    translateY: number,
    canvasScale: number,
    iterations: number = 300,
) => {
    if (nodesToLayout.length === 0) return new Map();

    const REPULSION_STRENGTH = 5000;
    const ATTRACTION_STRENGTH = 0.01;
    const DAMPING = 0.85;
    const MIN_DISTANCE = 50;
    const IDEAL_DISTANCE = 150;

    const positions = new Map<string, Position>();
    const velocities = new Map<string, Position>();

    nodesToLayout.forEach((node, index) => {
        const angle = (index / nodesToLayout.length) * Math.PI * 2;
        const radius = 100;
        positions.set(node.id, {
            x: Math.cos(angle) * radius,
            y: Math.sin(angle) * radius
        });
        velocities.set(node.id, { x: 0, y: 0 });
    });

    for (let iter = 0; iter < iterations; iter++) {
        const forces = new Map<string, Position>();

        nodesToLayout.forEach(node => {
            forces.set(node.id, { x: 0, y: 0 });
        });

        for (let i = 0; i < nodesToLayout.length; i++) {
            for (let j = i + 1; j < nodesToLayout.length; j++) {
                const node1 = nodesToLayout[i];
                const node2 = nodesToLayout[j];
                const pos1 = positions.get(node1.id)!;
                const pos2 = positions.get(node2.id)!;

                const dx = pos2.x - pos1.x;
                const dy = pos2.y - pos1.y;
                const distance = Math.sqrt(dx * dx + dy * dy) || 1;

                if (distance < MIN_DISTANCE) continue;

                const repulsionForce = REPULSION_STRENGTH / (distance * distance);
                const fx = (dx / distance) * repulsionForce;
                const fy = (dy / distance) * repulsionForce;

                const force1 = forces.get(node1.id)!;
                const force2 = forces.get(node2.id)!;
                force1.x -= fx;
                force1.y -= fy;
                force2.x += fx;
                force2.y += fy;
            }
        }

        nodesToLayout.forEach(node => {
            node.next.forEach(nextNode => {
                const pos1 = positions.get(node.id);
                const pos2 = positions.get(nextNode.id);

                if (!pos1 || !pos2) return;

                const dx = pos2.x - pos1.x;
                const dy = pos2.y - pos1.y;
                const distance = Math.sqrt(dx * dx + dy * dy) || 1;

                const attractionForce = ATTRACTION_STRENGTH * (distance - IDEAL_DISTANCE);
                const fx = (dx / distance) * attractionForce;
                const fy = (dy / distance) * attractionForce;

                const force1 = forces.get(node.id)!;
                const force2 = forces.get(nextNode.id)!;
                force1.x += fx;
                force1.y += fy;
                force2.x -= fx;
                force2.y -= fy;
            });
        });

        nodesToLayout.forEach(node => {
            const velocity = velocities.get(node.id)!;
            const force = forces.get(node.id)!;
            const pos = positions.get(node.id)!;

            velocity.x = (velocity.x + force.x) * DAMPING;
            velocity.y = (velocity.y + force.y) * DAMPING;

            pos.x += velocity.x;
            pos.y += velocity.y;
        });
    }

    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;

    positions.forEach(pos => {
        minX = Math.min(minX, pos.x);
        maxX = Math.max(maxX, pos.x);
        minY = Math.min(minY, pos.y);
        maxY = Math.max(maxY, pos.y);
    });

    const graphCenterX = (minX + maxX) / 2;
    const graphCenterY = (minY + maxY) / 2;


    const screenCenterX = canvasWidth / 2;
    const screenCenterY = canvasHeight / 2;
    const worldCenterX = (screenCenterX - translateX) / canvasScale;
    const worldCenterY = (screenCenterY - translateY) / canvasScale;

    const offsetX = worldCenterX - graphCenterX;
    const offsetY = worldCenterY - graphCenterY;

    const finalPositions = new Map<string, Position>();
    positions.forEach((pos, id) => {
        finalPositions.set(id, {
            x: Math.round(pos.x + offsetX),
            y: Math.round(pos.y + offsetY)
        });
    });

    return finalPositions;
}