use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::dtos::MutationType;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct GenomeLineageRecord {
    pub genome_id: String,
    pub generation: u32,
    pub parent_ids: Vec<String>,
    pub mutation_type: MutationType,
    pub created_at_unix_ms: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct GenealogyGraph {
    pub nodes: HashMap<String, GenomeLineageRecord>,
    pub edges: Vec<(String, String)>, // parent -> child
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct GenealogyPath {
    pub target_genome_id: String,
    pub records: Vec<GenomeLineageRecord>,
    pub edges: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenealogyError {
    GenomeAlreadyExists(String),
    GenomeNotFound(String),
    ParentNotFound(String),
    SelfParenting(String),
    CycleDetected,
}

impl fmt::Display for GenealogyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GenealogyError::GenomeAlreadyExists(id) => {
                write!(f, "Genome '{}' already exists in genealogy graph", id)
            }
            GenealogyError::GenomeNotFound(id) => write!(f, "Genome '{}' not found", id),
            GenealogyError::ParentNotFound(id) => write!(f, "Parent genome '{}' not found", id),
            GenealogyError::SelfParenting(id) => {
                write!(f, "Genome '{}' cannot reference itself as parent", id)
            }
            GenealogyError::CycleDetected => write!(f, "Cycle detected in genealogy graph"),
        }
    }
}

impl std::error::Error for GenealogyError {}

#[derive(Debug, Default)]
pub struct GenealogyStore {
    graph: GenealogyGraph,
}

impl GenealogyStore {
    pub fn new() -> Self {
        Self {
            graph: GenealogyGraph::default(),
        }
    }

    pub fn graph(&self) -> &GenealogyGraph {
        &self.graph
    }

    pub fn register_founder_if_missing(&mut self, genome_id: &str, generation: u32) {
        if self.graph.nodes.contains_key(genome_id) {
            return;
        }

        let _ = self.register_founder(genome_id.to_string(), generation);
    }

    pub fn register_founder(
        &mut self,
        genome_id: String,
        generation: u32,
    ) -> Result<(), GenealogyError> {
        if self.graph.nodes.contains_key(&genome_id) {
            return Err(GenealogyError::GenomeAlreadyExists(genome_id));
        }

        self.graph.nodes.insert(
            genome_id.clone(),
            GenomeLineageRecord {
                genome_id,
                generation,
                parent_ids: vec![],
                mutation_type: MutationType::Random,
                created_at_unix_ms: now_unix_ms(),
            },
        );

        Ok(())
    }

    pub fn register_mutation(
        &mut self,
        parent_id: String,
        child_id: String,
        mutation_type: MutationType,
        generation: u32,
    ) -> Result<(), GenealogyError> {
        self.register_child(vec![parent_id], child_id, mutation_type, generation)
    }

    pub fn register_crossover(
        &mut self,
        parent_a: String,
        parent_b: String,
        child_id: String,
        generation: u32,
    ) -> Result<(), GenealogyError> {
        self.register_child(
            vec![parent_a.clone(), parent_b.clone()],
            child_id,
            MutationType::Crossover {
                parent1: parent_a,
                parent2: parent_b,
            },
            generation,
        )
    }

    pub fn get_genealogy(&self, genome_id: &str) -> Result<GenealogyPath, GenealogyError> {
        if !self.graph.nodes.contains_key(genome_id) {
            return Err(GenealogyError::GenomeNotFound(genome_id.to_string()));
        }

        let mut ids = HashSet::new();
        ids.insert(genome_id.to_string());

        let mut queue = VecDeque::new();
        queue.push_back(genome_id.to_string());

        while let Some(current) = queue.pop_front() {
            if let Some(record) = self.graph.nodes.get(&current) {
                for parent in &record.parent_ids {
                    if ids.insert(parent.clone()) {
                        queue.push_back(parent.clone());
                    }
                }
            }
        }

        let mut records: Vec<_> = ids
            .iter()
            .filter_map(|id| self.graph.nodes.get(id).cloned())
            .collect();
        records.sort_by(|a, b| {
            a.generation
                .cmp(&b.generation)
                .then_with(|| a.genome_id.cmp(&b.genome_id))
        });

        let edges = self
            .graph
            .edges
            .iter()
            .filter(|(parent, child)| ids.contains(parent) && ids.contains(child))
            .cloned()
            .collect();

        Ok(GenealogyPath {
            target_genome_id: genome_id.to_string(),
            records,
            edges,
        })
    }

    pub fn get_ancestors(
        &self,
        genome_id: &str,
        depth: Option<u32>,
    ) -> Result<Vec<GenomeLineageRecord>, GenealogyError> {
        if !self.graph.nodes.contains_key(genome_id) {
            return Err(GenealogyError::GenomeNotFound(genome_id.to_string()));
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back((genome_id.to_string(), 0_u32));
        visited.insert(genome_id.to_string());

        let mut out = Vec::new();
        while let Some((current, d)) = queue.pop_front() {
            if let Some(max_depth) = depth {
                if d >= max_depth {
                    continue;
                }
            }

            if let Some(record) = self.graph.nodes.get(&current) {
                for parent in &record.parent_ids {
                    if visited.insert(parent.clone()) {
                        if let Some(parent_record) = self.graph.nodes.get(parent) {
                            out.push(parent_record.clone());
                        }
                        queue.push_back((parent.clone(), d + 1));
                    }
                }
            }
        }

        out.sort_by(|a, b| {
            a.generation
                .cmp(&b.generation)
                .then_with(|| a.genome_id.cmp(&b.genome_id))
        });
        Ok(out)
    }

    pub fn get_descendants(
        &self,
        genome_id: &str,
        depth: Option<u32>,
    ) -> Result<Vec<GenomeLineageRecord>, GenealogyError> {
        if !self.graph.nodes.contains_key(genome_id) {
            return Err(GenealogyError::GenomeNotFound(genome_id.to_string()));
        }

        let mut children_by_parent: HashMap<&str, Vec<&str>> = HashMap::new();
        for (parent, child) in &self.graph.edges {
            children_by_parent
                .entry(parent.as_str())
                .or_default()
                .push(child.as_str());
        }

        let mut visited = HashSet::new();
        visited.insert(genome_id.to_string());
        let mut queue = VecDeque::new();
        queue.push_back((genome_id.to_string(), 0_u32));

        let mut out = Vec::new();
        while let Some((current, d)) = queue.pop_front() {
            if let Some(max_depth) = depth {
                if d >= max_depth {
                    continue;
                }
            }

            if let Some(children) = children_by_parent.get(current.as_str()) {
                for child in children {
                    let child_id = (*child).to_string();
                    if visited.insert(child_id.clone()) {
                        if let Some(child_record) = self.graph.nodes.get(&child_id) {
                            out.push(child_record.clone());
                        }
                        queue.push_back((child_id, d + 1));
                    }
                }
            }
        }

        out.sort_by(|a, b| {
            a.generation
                .cmp(&b.generation)
                .then_with(|| a.genome_id.cmp(&b.genome_id))
        });
        Ok(out)
    }

    fn register_child(
        &mut self,
        parent_ids: Vec<String>,
        child_id: String,
        mutation_type: MutationType,
        generation: u32,
    ) -> Result<(), GenealogyError> {
        if self.graph.nodes.contains_key(&child_id) {
            return Err(GenealogyError::GenomeAlreadyExists(child_id));
        }

        for parent_id in &parent_ids {
            if parent_id == &child_id {
                return Err(GenealogyError::SelfParenting(child_id));
            }
            if !self.graph.nodes.contains_key(parent_id) {
                return Err(GenealogyError::ParentNotFound(parent_id.clone()));
            }
        }

        self.graph.nodes.insert(
            child_id.clone(),
            GenomeLineageRecord {
                genome_id: child_id.clone(),
                generation,
                parent_ids: parent_ids.clone(),
                mutation_type,
                created_at_unix_ms: now_unix_ms(),
            },
        );

        let prev_edge_count = self.graph.edges.len();
        for parent_id in &parent_ids {
            self.graph
                .edges
                .push((parent_id.clone(), child_id.clone()));
        }

        if let Err(e) = validate_acyclic(&self.graph) {
            self.graph.nodes.remove(&child_id);
            self.graph.edges.truncate(prev_edge_count);
            eprintln!(
                "[genealogy] rejected mutation for child '{}': {}",
                child_id, e
            );
            return Err(e);
        }

        Ok(())
    }
}

pub fn validate_acyclic(graph: &GenealogyGraph) -> Result<(), GenealogyError> {
    let mut in_degree: HashMap<&str, usize> = graph
        .nodes
        .keys()
        .map(|k| (k.as_str(), 0_usize))
        .collect();
    let mut adjacency: HashMap<&str, Vec<&str>> = HashMap::new();

    for (parent, child) in &graph.edges {
        if !graph.nodes.contains_key(parent) {
            return Err(GenealogyError::ParentNotFound(parent.clone()));
        }
        if !graph.nodes.contains_key(child) {
            return Err(GenealogyError::GenomeNotFound(child.clone()));
        }

        adjacency
            .entry(parent.as_str())
            .or_default()
            .push(child.as_str());
        if let Some(degree) = in_degree.get_mut(child.as_str()) {
            *degree += 1;
        }
    }

    let mut queue: VecDeque<&str> = in_degree
        .iter()
        .filter_map(|(node, deg)| if *deg == 0 { Some(*node) } else { None })
        .collect();

    let mut visited = 0_usize;
    while let Some(node) = queue.pop_front() {
        visited += 1;

        if let Some(children) = adjacency.get(node) {
            for child in children {
                if let Some(degree) = in_degree.get_mut(child) {
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(child);
                    }
                }
            }
        }
    }

    if visited == graph.nodes.len() {
        Ok(())
    } else {
        Err(GenealogyError::CycleDetected)
    }
}

fn now_unix_ms() -> u64 {
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    dur.as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn founder_without_parents() {
        let mut store = GenealogyStore::new();
        store
            .register_founder("g0".to_string(), 0)
            .expect("founder must register");

        let rec = store.graph().nodes.get("g0").expect("founder record missing");
        assert!(rec.parent_ids.is_empty());
        assert_eq!(rec.generation, 0);
    }

    #[test]
    fn mutation_with_single_parent() {
        let mut store = GenealogyStore::new();
        store.register_founder("g0".to_string(), 0).unwrap();

        store
            .register_mutation(
                "g0".to_string(),
                "g1".to_string(),
                MutationType::AddNode {
                    node_type: "Dense".to_string(),
                    source: "n1".to_string(),
                    target: "n2".to_string(),
                },
                1,
            )
            .unwrap();

        let g1 = store.graph().nodes.get("g1").unwrap();
        assert_eq!(g1.parent_ids, vec!["g0".to_string()]);
        assert_eq!(store.graph().edges, vec![("g0".to_string(), "g1".to_string())]);
    }

    #[test]
    fn crossover_with_two_parents() {
        let mut store = GenealogyStore::new();
        store.register_founder("ga".to_string(), 0).unwrap();
        store.register_founder("gb".to_string(), 0).unwrap();

        store
            .register_crossover(
                "ga".to_string(),
                "gb".to_string(),
                "gc".to_string(),
                1,
            )
            .unwrap();

        let gc = store.graph().nodes.get("gc").unwrap();
        assert_eq!(gc.parent_ids.len(), 2);
        assert!(store
            .graph()
            .edges
            .contains(&("ga".to_string(), "gc".to_string())));
        assert!(store
            .graph()
            .edges
            .contains(&("gb".to_string(), "gc".to_string())));
    }

    #[test]
    fn ancestry_traversal_is_correct() {
        let mut store = GenealogyStore::new();
        store.register_founder("g0".to_string(), 0).unwrap();
        store
            .register_mutation(
                "g0".to_string(),
                "g1".to_string(),
                MutationType::Random,
                1,
            )
            .unwrap();
        store
            .register_mutation(
                "g1".to_string(),
                "g2".to_string(),
                MutationType::Random,
                2,
            )
            .unwrap();

        let ancestors = store.get_ancestors("g2", None).unwrap();
        assert_eq!(ancestors.len(), 2);
        assert_eq!(ancestors[0].genome_id, "g0");
        assert_eq!(ancestors[1].genome_id, "g1");

        let descendants = store.get_descendants("g0", None).unwrap();
        assert_eq!(descendants.len(), 2);
        assert_eq!(descendants[0].genome_id, "g1");
        assert_eq!(descendants[1].genome_id, "g2");
    }

    #[test]
    fn rejects_cyclic_relationships() {
        let mut graph = GenealogyGraph::default();
        graph.nodes.insert(
            "a".to_string(),
            GenomeLineageRecord {
                genome_id: "a".to_string(),
                generation: 0,
                parent_ids: vec![],
                mutation_type: MutationType::Random,
                created_at_unix_ms: 1,
            },
        );
        graph.nodes.insert(
            "b".to_string(),
            GenomeLineageRecord {
                genome_id: "b".to_string(),
                generation: 1,
                parent_ids: vec!["a".to_string()],
                mutation_type: MutationType::Random,
                created_at_unix_ms: 2,
            },
        );
        graph.edges.push(("a".to_string(), "b".to_string()));
        graph.edges.push(("b".to_string(), "a".to_string()));

        let res = validate_acyclic(&graph);
        assert!(matches!(res, Err(GenealogyError::CycleDetected)));
    }

    #[test]
    fn generation_chain_integration() {
        let mut store = GenealogyStore::new();
        store.register_founder("g0".to_string(), 0).unwrap();
        store.register_founder("g1".to_string(), 0).unwrap();
        store
            .register_mutation(
                "g0".to_string(),
                "g2".to_string(),
                MutationType::ParameterMutation {
                    layer_id: "l1".to_string(),
                    param_name: "weight".to_string(),
                },
                1,
            )
            .unwrap();
        store
            .register_crossover(
                "g1".to_string(),
                "g2".to_string(),
                "g3".to_string(),
                2,
            )
            .unwrap();
        store
            .register_mutation(
                "g3".to_string(),
                "g4".to_string(),
                MutationType::RemoveNode {
                    node_id: "n7".to_string(),
                },
                3,
            )
            .unwrap();

        let path = store.get_genealogy("g4").unwrap();
        assert_eq!(path.target_genome_id, "g4");
        assert_eq!(path.records.len(), 5);

        let ancestors_d2 = store.get_ancestors("g4", Some(2)).unwrap();
        assert_eq!(ancestors_d2.len(), 3); // g3, g1, g2
    }
}