from collections import OrderedDict, defaultdict
from itertools import chain, permutations
from functools import cache
import re

import torch
from evenet.utilities.group_theory import complete_indices, symmetry_group

from evenet.dataset.types import *
from evenet.utilities.group_theory import (
    power_set,
    complete_symbolic_symmetry_group,
    complete_symmetry_group,
    expand_permutations
)


def cached_property(func):
    return property(cache(func))


def with_default(value, default):
    return default if value is None else value


def key_with_default(database, key, default):
    if key not in database:
        return default

    value = database[key]
    return default if value is None else value


def normalize_child_key(child: str) -> str:
    # Remove trailing digits, e.g., 'q1' -> 'q', 'l2' -> 'l'
    return re.sub(r'\d+$', '', child)


def build_topology_key(resonance: str, children: dict[str, int]) -> str:
    base = re.sub(r'\d+', '', resonance)
    norm_keys = ''.join(sorted(normalize_child_key(child) for child in children))
    return f'{base}/{norm_keys}'


def compute_head_weights_per_process(product_mappings, pairing_topology) -> dict[str, dict[str, float]]:
    process_to_head_weights = {}
    for process, particles in product_mappings.items():
        head_count = defaultdict(int)
        for resonance_name, child_map in particles.items():
            topology_key = build_topology_key(resonance_name, child_map)
            if topology_key in pairing_topology:
                category = pairing_topology[topology_key]["pairing_topology_category"]
                head_count[category] += 1
            # else:
            #     print(f"[warn] '{topology_key}' not found in pairing_topology")
        total = sum(head_count.values())
        weights = {cat: count / total for cat, count in head_count.items()} if total > 0 else {}
        process_to_head_weights[process] = weights
    return process_to_head_weights


def compute_segment_tags(product_mappings, pairing_topology, resonance_info) -> dict[str, dict[str, float]]:
    process_to_segment_tags = {}
    for process, particles in product_mappings.items():
        process_to_segment_tags[process] = {}
        for resonance_name, child_map in particles.items():
            topology_key = build_topology_key(resonance_name, child_map)
            if topology_key in pairing_topology:
                process_to_segment_tags[process][resonance_name] = \
                resonance_info[pairing_topology[topology_key]["pairing_topology_category"]][topology_key].get(
                    'segment_tag', 0)
            # else:
            #     print(f"[warn] '{topology_key}' not found in pairing_topology")
    return process_to_segment_tags


class EventInfo:
    def __init__(
            self,

            # Information about observable inputs for this event.
            input_types: InputDict[str, InputType],
            input_features: InputDict[str, Tuple[FeatureInfo, ...]],

            # Information about the target structure for this event.
            event_particles: Dict[str, Particles],
            product_particles: Dict[str, EventDict[str, Particles]],

            # Information about auxiliary values attached to this event.
            segmentations: Dict[str, Dict[str, str]],
            regressions: FeynmanDict[str, List[RegressionInfo]],
            classifications: FeynmanDict[str, List[ClassificationInfo]],
            class_label: Dict[str, Dict],
            resonance_info: Dict[str, Dict],
            resonance_particle_properties: List,
            generations: Dict[str, Dict],
            invisible_input_features: Tuple[FeatureInfo, ...],
            resonance_label=None,
    ):

        if resonance_label is None:
            resonance_label = []
        self.input_types = input_types
        self.input_names = list(input_types.keys())
        self.input_features = input_features

        self.event_particles = event_particles
        self.event_mapping = OrderedDict()
        self.event_symmetries = OrderedDict()

        self.product_particles = product_particles
        self.product_mappings: ODict[str, ODict[str, ODict[str, int]]] = OrderedDict()
        self.product_symmetries: ODict[str, ODict[str, Symmetries]] = OrderedDict()

        self.process_names = list(self.event_particles.keys())
        self.resonance_info = resonance_info
        self.resonance_particle_properties = resonance_particle_properties
        self.segmentation_indices = []  # 0: null class
        for decay_mode in self.resonance_info:
            for decay_channel in self.resonance_info[decay_mode]:
                if 'segment_tag' in self.resonance_info[decay_mode][decay_channel]:
                    segment_tag = self.resonance_info[decay_mode][decay_channel]['segment_tag']
                    if segment_tag not in self.segmentation_indices:
                        self.segmentation_indices.append(segment_tag)

        self.generations = generations
        self.invisible_input_features = invisible_input_features

        for process in self.event_particles:

            self.event_mapping[process] = self.construct_mapping(self.event_particles[process])
            self.event_symmetries[process] = Symmetries(
                len(self.event_particles[process]),
                self.apply_mapping(self.event_particles[process].permutations, self.event_mapping[process])
            )

            product_mappings_process = OrderedDict()
            product_symmetries_process = OrderedDict()

            for event_particle, product_particles in self.product_particles[process].items():
                product_mapping = self.construct_mapping(product_particles)

                product_mappings_process[event_particle] = product_mapping
                product_symmetries_process[event_particle] = Symmetries(
                    len(product_particles),
                    self.apply_mapping(product_particles.permutations, product_mapping)
                )
            self.product_mappings[process] = product_mappings_process
            self.product_symmetries[process] = product_symmetries_process

        self.event_permutations = OrderedDict()
        self.max_event_particles = 1
        for process_name in self.process_names:
            event_permutation = complete_indices(
                self.event_symmetries[process_name].degree,
                self.event_symmetries[process_name].permutations
            )
            self.event_permutations[process_name] = event_permutation
            for permutation_group in event_permutation:
                for permutation_element in permutation_group:
                    max_indices = len(list(permutation_element))
                    if max_indices > self.max_event_particles:
                        self.max_event_particles = max_indices

        self.segmentations = segmentations

        self.regressions = regressions
        self.regression_types = {
            "/".join([SpecialKey.Event] + [target.name]): target.type
            for target in regressions[SpecialKey.Event]
        }
        for process in self.product_particles:
            if process in regressions:
                for particle in regressions[process]:
                    if isinstance(regressions[process][particle], dict):
                        for product in regressions[process][particle]:
                            for target in regressions[process][particle][product]:
                                key = "/".join([process, particle, product] + [target.name])
                                self.regression_types[key] = target.type
                    else:
                        for target in regressions[process][particle]:
                            key = "/".join([process, particle] + [target.name])
                            self.regression_types[key] = target.type

        self.regression_names = self.regression_types.keys()
        self.num_regressions = len(self.regression_names)

        self.classifications = classifications
        self.classification_names = [
            '/'.join([SpecialKey.Event, target]) for target in
            self.classifications[SpecialKey.Event]
        ]
        self.class_label = class_label if len(class_label) > 0 else {}
        self.num_classes = dict()
        self.num_classes_total = 0
        if 'EVENT' in class_label:
            for name in class_label['EVENT']:
                self.num_classes[name] = (np.array(class_label['EVENT'][name])).shape[-1]
                self.num_classes_total += self.num_classes[name]
        else:
            self.num_classes_total = 1

        self.pairing_topology = OrderedDict()
        self.pairing_topology_category = OrderedDict()

        resonance_particle_properties_summary = []
        for process in self.process_names:
            for event_particle_name, product_symmetry in self.product_symmetries[process].items():
                topology_name = ''.join(self.product_particles[process][event_particle_name].names)
                topology_name = f"{event_particle_name}/{topology_name}"
                topology_name = re.sub(r'\d+', '', topology_name)

                pairing_topology_category = topology_name
                resonance_particle_properties_tmp = []
                for name, sub_dict in self.resonance_info.items():
                    for sub_name, third_dict in sub_dict.items():
                        if topology_name == sub_name:
                            pairing_topology_category = name
                            for property_name in self.resonance_particle_properties:
                                if property_name in third_dict:
                                    resonance_particle_properties_tmp.append(third_dict[property_name])
                                    # print(f"----{property_name}: {third_dict[property_name]}")
                                else:
                                    resonance_particle_properties_tmp.append(0.0)
                                    # print(f"----{property_name}: 0.0 (no entry so pad zero)")

                resonance_particle_properties_summary.append(np.array(resonance_particle_properties_tmp))
                if topology_name not in self.pairing_topology:
                    self.pairing_topology[topology_name] = {
                        "product_particles": self.product_particles[process][event_particle_name],
                        "product_symmetry": product_symmetry,
                        "pairing_topology_category": pairing_topology_category,
                        "resonance_particle_properties": torch.Tensor(np.array(resonance_particle_properties_tmp))}
                    if pairing_topology_category not in self.pairing_topology_category:
                        self.pairing_topology_category[pairing_topology_category] = {
                            "product_particles": self.product_particles[process][event_particle_name],
                            "product_symmetry": product_symmetry,
                            "nCond": len(resonance_particle_properties_tmp)}

        self.assignment_names = OrderedDict()
        for process in self.product_particles:
            self.assignment_names[process] = []
            for event_particle, daughter_particles in self.product_particles[process].items():
                self.assignment_names[process].append(event_particle)

        if len(resonance_particle_properties_summary) == 0:
            self.resonance_particle_properties_mean = np.array([0.0])
            self.resonance_particle_properties_std = np.array([1.0])
        else:
            resonance_particle_properties_summary = np.stack(resonance_particle_properties_summary)
            self.resonance_particle_properties_mean = np.mean(resonance_particle_properties_summary, axis=0)
            self.resonance_particle_properties_std = np.std(resonance_particle_properties_summary, axis=0)
            self.resonance_particle_properties_std[self.resonance_particle_properties_std < 1e-6] = 1.0

        self.resonance_particle_properties_mean = torch.Tensor(self.resonance_particle_properties_mean)
        self.resonance_particle_properties_std = torch.Tensor(self.resonance_particle_properties_std)

        # Generation Head setting
        # For point cloud generation
        self.sequential_feature_names = []
        self.sequential_inv_cdf_index = []
        iglobal_index = 0
        seq_index = 0
        self.generation_condition_indices = []
        self.generation_target_indices = []
        self.generation_target_names = []
        self.generation_pc_condition_indices = []
        self.generation_pc_condition_names = []

        self.generation_pc_indices = []
        self.generation_pc_names = []

        for input_name, input_feature in self.input_features.items():
            if self.input_types[input_name] == InputType.Global:
                for input_feature_element in input_feature:
                    if input_feature_element.name in self.generations["Conditions"]:
                        self.generation_condition_indices.append(iglobal_index)
                        self.generation_pc_condition_indices.append(iglobal_index)
                        self.generation_pc_condition_names.append(input_feature_element.name)
                    elif input_feature_element.name in self.generations.get("GlobalTargets", []):
                        self.generation_target_indices.append(iglobal_index)
                        self.generation_target_names.append(input_feature_element.name)
                        self.generation_pc_condition_indices.append(iglobal_index)
                        self.generation_pc_condition_names.append(input_feature_element.name)
                    iglobal_index += 1
            elif self.input_types[input_name] == InputType.Sequential:
                for input_feature_element in input_feature:
                    log_prefix = "log_" if input_feature_element.log_scale else ""
                    name = f"{log_prefix}{input_feature_element.name}"
                    self.sequential_feature_names.append(name)
                    if input_feature_element.uniform:
                        self.sequential_inv_cdf_index.append(seq_index)

                    if input_feature_element.name in self.generations.get("Events", []):
                        self.generation_pc_indices.append(seq_index)
                        self.generation_pc_names.append(name)
                    seq_index += 1

        # For invisible generation
        self.invisible_feature_names = []
        self.invisible_inv_cdf_index = []
        for idx, input_feature_element in enumerate(self.invisible_input_features):
            log_prefix = "log_" if input_feature_element.log_scale else ""
            name = f"{log_prefix}{input_feature_element.name}"
            self.invisible_feature_names.append(name)
            if input_feature_element.uniform:
                self.invisible_inv_cdf_index.append(idx)

        search_name = ["pt", "eta", "phi", "energy"]
        self.ptetaphienergy_index = []
        for target_name in search_name:
            sequential_index = 0
            for input_name, input_feature in self.input_features.items():
                if self.input_types[input_name] == InputType.Sequential:
                    for input_feature_element in input_feature:
                        if input_feature_element.name.lower() == target_name.lower():
                            self.ptetaphienergy_index.append(sequential_index)
                            break
                        sequential_index += 1

        ### Process to Topology Dictionary ###
        self.process_to_topology: dict[str, dict[str, float]] = compute_head_weights_per_process(
            self.product_mappings, self.pairing_topology
        )
        self.process_to_segment_tags: dict[str, dict[str, float]] = compute_segment_tags(
            self.product_mappings, self.pairing_topology, self.resonance_info
        )
        if self.process_to_segment_tags:
            self.total_segment_tags = max(
                v for _, tags in self.process_to_segment_tags.items() for v in tags.values()) + 1
        else:
            self.total_segment_tags = 0
        self.segment_label = {
            label: clsnum for clsnum, label in enumerate(resonance_label[0])
        } if len(resonance_label) > 0 else {}

    # def normalized_features(self, input_name: str) -> NDArray[bool]:
    #     return np.array([feature.normalize for feature in self.input_features[input_name]])
    #
    # def log_features(self, input_name: str) -> NDArray[bool]:
    #     return np.array([feature.log_scale for feature in self.input_features[input_name]])

    @cached_property
    def event_symbolic_group(self) -> ODict[str, SymbolicPermutationGroup]:
        event_symbolic_group_dict = OrderedDict()
        for process in self.process_names:
            event_symbolic_group_dict[process] = complete_symbolic_symmetry_group(*(self.event_symmetries[process]))
        return event_symbolic_group_dict

    @cached_property
    def event_permutation_group(self) -> ODict[str, PermutationGroup]:
        event_permutation_group_dict = OrderedDict()
        for process in self.process_names:
            event_permutation_group_dict[process] = complete_symmetry_group(*(self.event_symmetries[process]))
        return event_permutation_group_dict

    @cached_property
    def ordered_event_transpositions(self) -> ODict[str, Set[List[int]]]:
        ordered_event_transpositions_dict = OrderedDict()
        for process in self.event_symbolic_group:
            ordered_event_transpositions_dict[process] = set(chain.from_iterable(
                e.transpositions()
                for e in self.event_symbolic_group[process].elements
            ))
        return ordered_event_transpositions_dict

    @cached_property
    def event_transpositions(self) -> ODict[str, Set[Tuple[int, int]]]:
        event_transpositions_dict = OrderedDict()
        for process in self.ordered_event_transpositions:
            event_transpositions_dict[process] = set(
                map(tuple, map(sorted, self.ordered_event_transpositions[process])))
        return event_transpositions_dict

    @cached_property
    def event_equivalence_classes(self) -> ODict[str, Set[FrozenSet[FrozenSet[int]]]]:

        event_equivalence_classes_dict = OrderedDict()
        for process in self.event_symmetries:
            num_particles = self.event_symmetries[process].degree
            group = self.event_symbolic_group[process]
            sets = map(frozenset, power_set(range(num_particles)))
            event_equivalence_classes_dict[process] = set(
                frozenset(frozenset(g(x) for x in s) for g in group.elements) for s in sets)
        return event_equivalence_classes_dict

    @cached_property
    def product_permutation_groups(self) -> ODict[str, ODict[str, PermutationGroup]]:

        product_permutation_groups_dict = OrderedDict()
        for process in self.product_symmetries:
            output = []
            for name, (degree, symmetries) in self.product_symmetries[process].items():
                symmetries = [] if symmetries is None else symmetries
                permutation_group = complete_symmetry_group(degree, symmetries)
                output.append((name, permutation_group))
            product_permutation_groups_dict[process] = OrderedDict(output)

        return product_permutation_groups_dict

    @cached_property
    def product_symbolic_groups(self) -> ODict[str, ODict[str, SymbolicPermutationGroup]]:

        product_symbolic_groups_dict = OrderedDict()
        for process in self.product_symmetries:
            output = []
            for name, (degree, symmetries) in self.product_symmetries[process].items():
                symmetries = [] if symmetries is None else symmetries
                permutation_group = complete_symbolic_symmetry_group(degree, symmetries)
                output.append((name, permutation_group))
            product_symbolic_groups_dict[process] = OrderedDict(output)
        return product_symbolic_groups_dict

    def num_features(self, input_name: str) -> int:
        return len(self.input_features[input_name])

    def input_type(self, input_name: str) -> InputType:
        return self.input_types[input_name].upper()

    @staticmethod
    def parse_list(list_string: str):
        return tuple(map(str.strip, list_string.strip("][").strip(")(").split(",")))

    @staticmethod
    def construct_mapping(variables: Iterable[str]) -> ODict[str, int]:
        return OrderedDict(map(reversed, enumerate(variables)))

    @staticmethod
    def apply_mapping(permutations: Permutations, mapping: Dict[str, int]) -> MappedPermutations:
        return [
            [
                tuple(
                    mapping[element]
                    for element in cycle
                )
                for cycle in permutation
            ]
            for permutation in permutations
        ]

    @classmethod
    def construct(cls, config: dict, resonance_info: dict):
        # Extract input feature information.
        # ----------------------------------
        input_types = OrderedDict()
        input_features = OrderedDict()

        for input_type in config[SpecialKey.Inputs]:
            current_inputs = with_default(config[SpecialKey.Inputs][input_type], default={})

            for input_name, input_information in current_inputs.items():
                input_types[input_name] = input_type.upper()
                input_features[input_name] = tuple(
                    FeatureInfo(
                        name=name,
                        normalize=("normalize" in normalize.lower()) or ("true" in normalize.lower()),
                        log_scale="log" in normalize.lower(),
                        uniform="uniform" in normalize.lower()
                    )

                    for name, normalize in input_information.items()
                )

        def synthesize_permutations_from_symmetry():
            cfg = deepcopy(config)
            if SpecialKey.Permutations in cfg:
                return cfg
            if SpecialKey.Event not in cfg:
                return cfg

            perms = defaultdict(dict)

            for proc, node in cfg[SpecialKey.Event].items():
                diag = node.get("diagram", node)

                # event-level SYMMETRY
                evt_groups = diag.get("SYMMETRY")
                if isinstance(evt_groups, list):
                    perms[proc][SpecialKey.Event] = [evt_groups]

                # product-level SYMMETRY for each event particle
                for ep, prods in diag.items():
                    if ep == "SYMMETRY":
                        continue
                    if isinstance(prods, dict):
                        groups = prods.get("SYMMETRY")
                        if isinstance(groups, list):
                            perms[proc][ep] = [groups]

                if proc not in perms:
                    perms[proc] = {}

            cfg[SpecialKey.Permutations] = dict(perms)
            return cfg

        # Extract event and permutation information.
        # ------------------------------------------
        config = synthesize_permutations_from_symmetry()
        permutation_config = key_with_default(config, SpecialKey.Permutations, default={})

        event_particles_summary = OrderedDict()
        product_particles_summary = OrderedDict()

        event_particles = OrderedDict()  # Default value
        product_particles = OrderedDict()  # Default value

        for process in permutation_config:
            event_cfg = config[SpecialKey.Event].get(process, {})
            diagram = event_cfg.get("diagram", event_cfg)
            event_names = tuple([k for k in diagram.keys() if k != "SYMMETRY"])
            event_permutations = key_with_default(permutation_config[process], SpecialKey.Event, default=[])
            event_permutations = expand_permutations(event_permutations)
            event_particles = Particles(event_names, event_permutations)
            product_particles = OrderedDict()

            for event_particle in event_particles:
                if isinstance(diagram[event_particle], list):
                    products = diagram[event_particle]
                else:
                    products = {k: v for k, v in diagram[event_particle].items() if k != "SYMMETRY"}

                product_names = [
                    next(iter(product.keys())) if isinstance(product, dict) else product
                    for product in products
                ]

                product_sources = [
                    next(iter(product.values())) if isinstance(product, dict) else None
                    for product in products
                ]

                input_names = list(input_types.keys())
                product_sources = [
                    input_names.index(source) if source is not None else -1
                    for source in product_sources
                ]

                product_permutations = key_with_default(permutation_config[process], event_particle, default=[])
                product_permutations = expand_permutations(product_permutations)

                product_particles[event_particle] = Particles(product_names, product_permutations, product_sources)

            event_particles_summary[process] = event_particles
            product_particles_summary[process] = product_particles

        # Extract Segmentation Information.
        # -------------------------------
        segmentations = key_with_default(config, SpecialKey.Segmentations, default={})

        # Extract Regression Information.
        # -------------------------------
        regressions = key_with_default(config, SpecialKey.Regressions, default={})
        regressions = feynman_fill(regressions, event_particles, product_particles, constructor=list)

        # Fill in any default parameters for regressions such as gaussian type.
        regressions = feynman_map(
            lambda raw_regressions: [
                RegressionInfo(*(regression if isinstance(regression, list) else [regression]))
                for regression in raw_regressions
            ],
            regressions
        )

        # Extract Classification Information.
        # -----------------------------------
        classifications = key_with_default(config, SpecialKey.Classifications, default={})
        classifications = feynman_fill(classifications, event_particles, product_particles, constructor=list)

        class_label = key_with_default(config, SpecialKey.ClassLabel, default={})
        resonance_label = key_with_default(config, "RESONANCE_LABEL", default=[])

        generations = key_with_default(config, SpecialKey.Generations, default={})

        resonance_particle_property = key_with_default(config, SpecialKey.ParticleProperties, default=[])

        # Extract Neutrino Information.
        # Extract input feature information.
        # ----------------------------------
        invisible = key_with_default(config[SpecialKey.Generations], SpecialKey.Invisible, default={})
        invisible_input_features = tuple(
            FeatureInfo(
                name=name,
                normalize=("normalize" in normalize.lower()) or ("true" in normalize.lower()),
                log_scale="log" in normalize.lower(),
                uniform="uniform" in normalize.lower()
            )

            for name, normalize in invisible.items()
        )

        return cls(
            input_types,
            input_features,
            event_particles_summary,
            product_particles_summary,
            segmentations,
            regressions,
            classifications,
            class_label,
            resonance_info,
            resonance_particle_property,
            generations,
            invisible_input_features,
            resonance_label
        )
