import torch
import pickle
from evenet.control.global_config import DotDict

from evenet.network.layers.utils import RandomDrop

from evenet.network.body.normalizer import Normalizer
from evenet.network.body.embedding import GlobalVectorEmbedding, PETBody
from evenet.network.body.object_encoder import ObjectEncoder
from evenet.network.heads.classification.classification_head import ClassificationHead, RegressionHead
from evenet.network.heads.assignment.assignment_head import SharedAssignmentHead
from evenet.network.heads.generation.generation_head import GlobalCondGenerationHead, EventGenerationHead
from evenet.network.heads.segmentation.segmentation_head import SegmentationHead
from evenet.utilities.diffusion_sampler import get_logsnr_alpha_sigma
from evenet.network.layers.debug_layer import PointCloudTransformer
from evenet.utilities.group_theory import complete_indices
import torch.nn.functional as F

from evenet.utilities.diffusion_sampler import add_noise
from evenet.utilities.tool import gather_index
from torch import Tensor, nn
from typing import Dict, Optional, Any, Union
import re


class EveNetModel(nn.Module):
    def __init__(
            self,
            config: DotDict,
            device,
            classification: bool = False,
            regression: bool = False,
            global_generation: bool = False,
            point_cloud_generation: bool = False,
            neutrino_generation: bool = False,
            assignment: bool = False,
            segmentation: bool = False,
            normalization_dict: dict = None,
    ):
        super().__init__()
        # # Initialize the model with the given configuration
        self.options = config.options
        self.network_cfg = config.network
        self.event_info = config.event_info
        # # self.save_hyperparameters(self.options)
        self.include_classification = classification
        self.include_regression = regression
        self.include_global_generation = global_generation
        self.include_point_cloud_generation = point_cloud_generation
        self.include_neutrino_generation = neutrino_generation
        self.include_assignment = assignment
        self.include_segmentation = segmentation
        self.device = device

        # self.normalization_dict = normalization_dict

        # Initialize the normalization layer
        input_normalizers_setting = dict()
        for input_name, input_type in self.event_info.input_types.items():
            input_normalizers_setting_local = {
                "norm_mask": torch.tensor(
                    [feature_info.normalize for feature_info in self.event_info.input_features[input_name]],
                    device=self.device
                ),
                "mean": normalization_dict["input_mean"][input_name].to(self.device),
                "std": normalization_dict["input_std"][input_name].to(self.device)
            }

            if input_type in input_normalizers_setting:
                for element in input_normalizers_setting[input_type]:
                    input_normalizers_setting[input_type][element] = torch.cat(
                        input_normalizers_setting[input_type][element],
                        input_normalizers_setting_local[element],
                    )
            else:
                input_normalizers_setting[input_type] = input_normalizers_setting_local

        global_normalizer_info = input_normalizers_setting.get(
            "GLOBAL",
            {
                "norm_mask": torch.ones(1, dtype=torch.bool).to(self.device),
                "mean": torch.zeros(1).to(self.device),
                "std": torch.ones(1).to(self.device)
            }
        )

        self.global_input_dim: int = global_normalizer_info["norm_mask"].size()[-1]
        self.sequential_input_dim: int = input_normalizers_setting["SEQUENTIAL"]["norm_mask"].size()[-1]
        self.local_feature_indices = self.network_cfg.Body.PET.local_point_index

        self.sequential_normalizer = Normalizer(
            norm_mask=input_normalizers_setting["SEQUENTIAL"]["norm_mask"].to(self.device),
            mean=input_normalizers_setting["SEQUENTIAL"]["mean"].to(self.device),
            std=input_normalizers_setting["SEQUENTIAL"]["std"].to(self.device),
            inv_cdf_index=self.event_info.sequential_inv_cdf_index
        )

        self.global_normalizer = Normalizer(
            norm_mask=global_normalizer_info["norm_mask"].to(self.device),
            mean=global_normalizer_info["mean"].to(self.device),
            std=global_normalizer_info["std"].to(self.device),
        )

        if self.include_point_cloud_generation:
            self.num_point_cloud_normalizer = Normalizer(
                mean=normalization_dict["input_num_mean"]["Source"].unsqueeze(-1).to(self.device),
                std=normalization_dict["input_num_std"]["Source"].unsqueeze(-1).to(self.device),
                norm_mask=torch.tensor([1], device=self.device, dtype=torch.bool)
            )

        if self.include_neutrino_generation:
            self.invisible_input_dim: int = len(normalization_dict["invisible_mean"]["Source"])
            self.invisible_padding: int = self.sequential_input_dim - self.invisible_input_dim
            assert self.invisible_padding >= 0, f"Invisible Padding size {self.invisible_padding} is negative. "

            self.invisible_normalizer = Normalizer(
                mean=normalization_dict["invisible_mean"]["Source"].to(self.device),
                std=normalization_dict["invisible_std"]["Source"].to(self.device),
                norm_mask=torch.tensor([1], device=self.device, dtype=torch.bool),
                inv_cdf_index=self.event_info.invisible_inv_cdf_index,
                padding_size=self.invisible_padding,
            )

        # [1] Body
        global_embedding_cfg = self.network_cfg.Body.GlobalEmbedding
        self.GlobalEmbedding = GlobalVectorEmbedding(
            linear_block_type=global_embedding_cfg.linear_block_type,
            input_dim=self.global_input_dim,
            hidden_dim_scale=global_embedding_cfg.transformer_dim_scale,
            initial_embedding_dim=global_embedding_cfg.initial_embedding_dim,
            final_embedding_dim=global_embedding_cfg.hidden_dim,
            normalization_type=global_embedding_cfg.normalization,
            activation_type=global_embedding_cfg.linear_activation,
            skip_connection=global_embedding_cfg.skip_connection,
            num_embedding_layers=global_embedding_cfg.num_embedding_layers,
            dropout=global_embedding_cfg.dropout
        )

        # [1] Body
        pet_config = self.network_cfg.Body.PET
        self.PET = PETBody(
            num_feat=self.sequential_input_dim,
            num_keep=pet_config.num_feature_keep,
            feature_drop=pet_config.feature_drop,
            projection_dim=pet_config.hidden_dim,
            local=pet_config.enable_local_embedding,
            K=pet_config.local_Krank,
            num_local=pet_config.num_local_layer,
            num_layers=pet_config.num_layers,
            num_heads=pet_config.num_heads,
            drop_probability=pet_config.drop_probability,
            talking_head=pet_config.talking_head,
            layer_scale=pet_config.layer_scale,
            layer_scale_init=pet_config.layer_scale_init,
            dropout=pet_config.dropout,
            mode=pet_config.mode,
        )

        # [2] Classification + Regression + Assignment Body
        obj_encoder_cfg = self.network_cfg.Body.ObjectEncoder
        self.ObjectEncoder = ObjectEncoder(
            input_dim=pet_config.hidden_dim,
            hidden_dim=obj_encoder_cfg.hidden_dim,
            output_dim=obj_encoder_cfg.hidden_dim,
            position_embedding_dim=obj_encoder_cfg.position_embedding_dim,
            num_heads=obj_encoder_cfg.num_attention_heads,
            transformer_dim_scale=obj_encoder_cfg.transformer_dim_scale,
            num_linear_layers=obj_encoder_cfg.num_embedding_layers,
            num_encoder_layers=obj_encoder_cfg.num_encoder_layers,
            dropout=obj_encoder_cfg.dropout,
            conditioned=False,
            skip_connection=obj_encoder_cfg.skip_connection,
            encoder_skip_connection=obj_encoder_cfg.encoder_skip_connection,
        )

        # [3] Classification Head
        if self.include_classification:
            cls_cfg = self.network_cfg.Classification
            self.Classification = ClassificationHead(
                input_dim=obj_encoder_cfg.hidden_dim,
                class_label=self.event_info.class_label.get("EVENT", None),
                event_num_classes=self.event_info.num_classes,
                num_layers=cls_cfg.num_classification_layers,
                hidden_dim=cls_cfg.hidden_dim,
                skip_connection=cls_cfg.skip_connection,
                dropout=cls_cfg.dropout,
                num_attention_heads=cls_cfg.num_attention_heads,
            )
        # [4] Regression Head
        if self.include_regression:
            reg_cfg = self.network_cfg.Regression
            self.Regression = RegressionHead(
                input_dim=obj_encoder_cfg.hidden_dim,
                regressions_target=self.event_info.regressions,
                regression_names=self.event_info.regression_names,
                means=normalization_dict["regression_mean"],
                stds=normalization_dict["regression_std"],
                num_layers=reg_cfg.num_regression_layers,
                hidden_dim=reg_cfg.hidden_dim,
                dropout=reg_cfg.dropout,
                skip_connection=reg_cfg.skip_connection,
                device=self.device,
            )

        if self.include_assignment:
            # [5] Assignment Head
            self.Assignment = SharedAssignmentHead(
                resonance_particle_properties_mean=self.event_info.resonance_particle_properties_mean,
                resonance_particle_properties_std=self.event_info.resonance_particle_properties_std,
                pairing_topology=self.event_info.pairing_topology,
                process_names=self.event_info.process_names,
                pairing_topology_category=self.event_info.pairing_topology_category,
                event_particles=self.event_info.event_particles,
                event_permutation=self.event_info.event_permutations,
                product_particles=self.event_info.product_particles,
                product_symmetries=self.event_info.product_symmetries,
                feature_drop=self.network_cfg.Assignment.feature_drop,
                num_feature_keep=self.network_cfg.Assignment.num_feature_keep,
                input_dim=obj_encoder_cfg.hidden_dim,
                split_attention=self.network_cfg.Assignment.split_symmetric_attention,
                hidden_dim=self.network_cfg.Assignment.hidden_dim,
                position_embedding_dim=self.network_cfg.Assignment.position_embedding_dim,
                num_attention_heads=self.network_cfg.Assignment.num_attention_heads,
                transformer_dim_scale=self.network_cfg.Assignment.transformer_dim_scale,
                num_linear_layers=self.network_cfg.Assignment.num_linear_layers,
                num_encoder_layers=self.network_cfg.Assignment.num_encoder_layers,
                num_jet_embedding_layers=self.network_cfg.Assignment.num_jet_embedding_layers,
                num_jet_encoder_layers=self.network_cfg.Assignment.num_jet_encoder_layers,
                num_max_event_particles=self.event_info.max_event_particles,
                num_detection_layers=self.network_cfg.Assignment.num_detection_layers,
                dropout=self.network_cfg.Assignment.dropout,
                combinatorial_scale=self.network_cfg.Assignment.combinatorial_scale,
                encode_event_token=self.network_cfg.Assignment.encode_event_token,
                activation=self.network_cfg.Assignment.activation,
                skip_connection=self.network_cfg.Assignment.skip_connection,
                encoder_skip_connection=self.network_cfg.Assignment.encoder_skip_connection,
                device=self.device
            )

        # [6-1] Global Generation Head (for point cloud generation only)
        if self.include_global_generation:
            self.global_generation_target_indices = self.event_info.generation_target_indices
            self.GlobalGeneration = GlobalCondGenerationHead(
                num_layer=self.network_cfg.GlobalGeneration.num_layers,
                num_resnet_layer=self.network_cfg.GlobalGeneration.num_resnet_layers,
                input_dim=1 + len(self.event_info.generation_target_indices),
                hidden_dim=self.network_cfg.GlobalGeneration.hidden_dim,
                output_dim=1 + len(self.event_info.generation_target_indices),
                input_cond_indices=self.event_info.generation_condition_indices,
                num_classes=self.event_info.num_classes_total,
                resnet_dim=self.network_cfg.GlobalGeneration.resnet_dim,
                layer_scale_init=self.network_cfg.GlobalGeneration.layer_scale_init,
                feature_drop_for_stochastic_depth=self.network_cfg.GlobalGeneration.feature_drop_for_stochastic_depth,
                activation=self.network_cfg.GlobalGeneration.activation,
                dropout=self.network_cfg.GlobalGeneration.dropout
            )

        # [6-2] Event Generation Head (Recon-Level)
        if self.include_point_cloud_generation:
            self.generation_pc_condition_indices = self.event_info.generation_pc_condition_indices
            self.generation_pc_indices = self.event_info.generation_pc_indices
            self.ReconGeneration = EventGenerationHead(
                input_dim=pet_config.hidden_dim,
                projection_dim=self.network_cfg.ReconGeneration.hidden_dim,
                num_global_cond=global_embedding_cfg.hidden_dim,
                num_classes=self.event_info.num_classes_total,
                output_dim=self.sequential_input_dim,
                num_layers=self.network_cfg.ReconGeneration.num_layers,
                num_heads=self.network_cfg.ReconGeneration.num_heads,
                dropout=self.network_cfg.ReconGeneration.dropout,
                layer_scale=self.network_cfg.ReconGeneration.layer_scale,
                layer_scale_init=self.network_cfg.ReconGeneration.layer_scale_init,
                drop_probability=self.network_cfg.ReconGeneration.drop_probability,
                feature_drop=self.network_cfg.ReconGeneration.feature_drop,
            )

        # [6-3] Event Generation Head (Truth-Level)
        if self.include_neutrino_generation:
            self.TruthGeneration = EventGenerationHead(
                input_dim=pet_config.hidden_dim,
                projection_dim=self.network_cfg.TruthGeneration.hidden_dim,
                num_global_cond=global_embedding_cfg.hidden_dim,
                num_classes=self.event_info.num_classes_total,
                output_dim=self.invisible_input_dim + self.invisible_padding,
                num_layers=self.network_cfg.TruthGeneration.num_layers,
                num_heads=self.network_cfg.TruthGeneration.num_heads,
                dropout=self.network_cfg.TruthGeneration.dropout,
                layer_scale=self.network_cfg.TruthGeneration.layer_scale,
                layer_scale_init=self.network_cfg.TruthGeneration.layer_scale_init,
                drop_probability=self.network_cfg.TruthGeneration.drop_probability,
                feature_drop=self.network_cfg.TruthGeneration.feature_drop,
                position_encode=self.network_cfg.TruthGeneration.neutrino_position_encode,
                max_position_length=self.network_cfg.TruthGeneration.max_position_length
            )
            self.neutrino_position_encode = self.network_cfg.TruthGeneration.neutrino_position_encode

        # [7] Segmentation Head
        if self.include_segmentation:
            self.Segmentation = SegmentationHead(
                projection_dim=self.network_cfg.Segmentation.projection_dim,
                mask_dim=self.network_cfg.Segmentation.projection_dim,
                num_heads=self.network_cfg.Segmentation.num_heads,
                dropout=self.network_cfg.Segmentation.dropout,
                num_layers=self.network_cfg.Segmentation.num_layers,
                num_mask_mlp_layers=self.network_cfg.Segmentation.mask_mlp_layers,
                num_class=len(self.event_info.segmentation_indices),  # Binary classification for mask prediction
                num_queries=self.network_cfg.Segmentation.num_queries,
                return_intermediate=self.network_cfg.Segmentation.return_intermediate,
                norm_before=self.network_cfg.Segmentation.norm_before,
                encode_event_token=self.network_cfg.Segmentation.encode_event_token,
            )

        self.schedule_flags = [
            ("generation", self.include_point_cloud_generation),
            ("neutrino_generation", self.include_neutrino_generation),
            ("deterministic", self.include_classification or self.include_assignment or self.include_regression or self.include_segmentation),
        ]

    def forward(
            self, x: Dict[str, Tensor], time: Tensor,
            progressive_params: dict = None,
            schedules: list[tuple[str, bool]] = None
    ) -> dict[str, dict[Any, Any] | Any]:
        """

        :param schedules:
        :param time:
        :param progressive_params:
        :param x:
            - x['x']: point cloud, shape (batch_size, num_objects, num_features)
            - x['x_mask']: Mask for point cloud, shape (batch_size, num_objects)
                - 1: valid point
                - 0: invalid point
            - x['conditions']: conditions, shape (batch_size, num_conditions)
            - x['conditions_mask']: Mask for conditions, shape (batch_size, 1)
                - 1: valid condition
                - 0: invalid condition
            - x['classification']: classification targets, shape (batch_size,)
            - x['regression']: regression targets, shape (batch_size, num_regression_targets)
            - x['regression_mask']: Mask for regression targets, shape (batch_size, num_regression_targets)
                - 1: valid regression target
                - 0: invalid regression target
            - x['num_vectors']: number of vectors in the batch, shape (batch_size,)
            - x['num_sequential_vectors']: number of sequential vectors in the batch, shape (batch_size,)
            - x['assignment_indices']: assignment indices, shape (batch_size, num_resonaces, num_targets)
            - x['assignment_indices_mask']: Mask for assignment indices, shape (batch_size, num_resonances)
                - True: valid assignment index
                - False: invalid assignment index
            - x['assignment_mask']: assignment mask, shape (batch_size, num_resonances)
                - 1: valid assignment
                - 0: invalid assignment

            - x['x_invisible']: invisible point cloud, shape (batch_size, num_objects, num_features)
            - x['x_invisible_mask']: Mask for invisible point cloud, shape (batch_size, num_objects)
        """

        #############
        ##  Input  ##
        #############
        if progressive_params is None:
            progressive_params = dict()

        _, alpha, _ = get_logsnr_alpha_sigma(time)

        input_point_cloud = x['x']
        input_point_cloud_mask = x['x_mask'].unsqueeze(-1)
        global_conditions = x['conditions'].unsqueeze(1)  # (batch_size, 1, num_conditions)
        global_conditions_mask = x['conditions_mask'].unsqueeze(-1)  # (batch_size, 1, 1)

        class_label = x['classification'].unsqueeze(-1) if 'classification' in x else torch.zeros_like(
            x['conditions_mask']).long()  # (batch_size, 1)

        num_point_cloud = None
        if self.include_global_generation or self.include_point_cloud_generation:
            num_point_cloud = x['num_sequential_vectors'].unsqueeze(-1)  # (batch_size, 1)

        B, _, num_features = input_point_cloud.shape
        if 'x_invisible' in x:
            invisible_point_cloud = x['x_invisible']
            pad_size = self.invisible_padding

            # Pad invisible features to match input
            invisible_point_cloud = F.pad(invisible_point_cloud, (0, pad_size), value=0.0)
        else:
            invisible_point_cloud = torch.zeros(B, 1, num_features, device=input_point_cloud.device)

        invisible_point_cloud_mask = x['x_invisible_mask'].unsqueeze(
            -1) if 'x_invisible_mask' in x else torch.zeros_like(input_point_cloud_mask[:, [0], :]).bool()

        #######################################################
        ##  Produce visible + invisible point cloud masking  ##
        #######################################################

        # Requirement: input_point_cloud and invisible_point_cloud should have the same features

        # Create attention mask
        n_vis = input_point_cloud_mask.shape[1]
        n_invis = invisible_point_cloud_mask.shape[1]
        is_invisible_query = torch.cat([
            torch.zeros(n_vis, dtype=torch.bool, device=self.device),
            torch.ones(n_invis, dtype=torch.bool, device=self.device)
        ], dim=0)  # (L,)
        # Rule: visible query (False) cannot attend to invisible key (True)
        invisible_attn_mask = (~is_invisible_query[:, None]) & is_invisible_query[None, :]  # (L, L) , Q->K

        #########################
        ## Input normalization ##
        #########################

        input_point_cloud = self.sequential_normalizer(
            x=input_point_cloud,
            mask=input_point_cloud_mask,
        )

        if self.include_neutrino_generation:
            invisible_point_cloud = self.invisible_normalizer(
                x=invisible_point_cloud,
                mask=invisible_point_cloud_mask
            )

        global_conditions = self.global_normalizer(
            x=global_conditions,
            mask=global_conditions_mask
        )

        if self.include_global_generation or self.include_point_cloud_generation:
            num_point_cloud = self.num_point_cloud_normalizer(
                x=num_point_cloud,
                mask=None
            )

        ###########################
        ## Global Generator Head ##
        ###########################
        generations = dict()
        if self.include_global_generation:
            # [6-1] Global Generation Head
            if len(self.global_generation_target_indices) > 0:
                target_global = global_conditions[..., self.global_generation_target_indices].squeeze(1)
                target_global = torch.cat([num_point_cloud, target_global], dim=1)
            else:
                target_global = num_point_cloud

            target_global_noised, truth_target_global_vector = add_noise(target_global, time)
            predict_target_global_vector = self.GlobalGeneration(
                x=target_global_noised,
                time=time,
                global_cond=global_conditions,
                label=class_label
            )

            generations["global"] = {
                "vector": predict_target_global_vector,
                "truth": truth_target_global_vector.detach(),
            }

        outputs = dict()
        if schedules is None:
            schedules = self.schedule_flags

        full_input_point_cloud = None
        full_global_conditions = None

        for schedule_name, flag  in schedules:
            if not flag:
                continue

            ####################
            ##  Inject noise  ##
            ####################

            if schedule_name == "deterministic":
                full_input_point_cloud = input_point_cloud.contiguous()
                full_input_point_cloud_mask = input_point_cloud_mask.contiguous()
                full_attn_mask = None
                full_time = torch.zeros_like(time)
                time_masking = torch.zeros_like(full_input_point_cloud_mask).float()
                global_feature_mask = torch.ones_like(global_conditions).float()
            elif schedule_name == "generation":

                noise_prob = progressive_params.get("noise_prob", 1.0)
                attn_mask_turn_on = (progressive_params.get("reco_attn_mask", 0.0) > 0.5)

                noise_mask = (torch.rand(
                    input_point_cloud.size(0), input_point_cloud.size(1),
                    device=input_point_cloud.device
                ) < noise_prob).float().unsqueeze(-1)  # (B, L, 1)
                noise_mask = noise_mask * input_point_cloud_mask

                input_point_cloud_noised, truth_input_point_cloud_vector = add_noise(input_point_cloud, time)
                input_point_cloud_noised_tmp_mask = torch.zeros_like(input_point_cloud_noised)
                input_point_cloud_noised_tmp_mask[..., self.generation_pc_indices] = 1.0
                input_point_cloud_noised = input_point_cloud_noised * input_point_cloud_noised_tmp_mask

                full_input_point_cloud = input_point_cloud * (1.0 - noise_mask) + input_point_cloud_noised * noise_mask
                full_input_point_cloud_mask = input_point_cloud_mask.contiguous()

                is_noise_query = (noise_mask > 0.1).squeeze(-1)  # (B,L)
                is_noise_padding_query = is_noise_query | ~(full_input_point_cloud_mask.squeeze(-1).bool())  # (B,L)
                full_attn_mask = (~(is_noise_padding_query[:, :, None]) & is_noise_padding_query[:, None,
                                                                  :]) if attn_mask_turn_on else None  # (B, L, L)
                full_time = time
                time_masking = noise_mask
                global_feature_mask = torch.zeros_like(global_conditions).float()
                global_feature_mask[..., self.generation_pc_condition_indices] = 1.0


            else:
                invisible_point_cloud_noised, truth_invisible_point_cloud_vector = add_noise(
                    invisible_point_cloud, time
                )
                full_input_point_cloud = torch.cat([input_point_cloud, invisible_point_cloud_noised], dim=1)
                full_input_point_cloud_mask = torch.cat([input_point_cloud_mask, invisible_point_cloud_mask], dim=1)
                full_attn_mask = invisible_attn_mask
                full_time = time
                time_masking = torch.cat(
                    [torch.zeros_like(input_point_cloud_mask), invisible_point_cloud_mask], dim=1
                ).float()
                global_feature_mask = torch.ones_like(global_conditions).float()

            #############################
            ## Central embedding (PET) ##
            #############################

            full_global_conditions = self.GlobalEmbedding(
                x=global_conditions * global_feature_mask,
                mask=global_conditions_mask
            )

            local_points = full_input_point_cloud[..., self.local_feature_indices]
            full_input_point_cloud = self.PET(
                input_features=full_input_point_cloud,
                input_points=local_points,
                mask=full_input_point_cloud_mask,
                attn_mask=full_attn_mask,
                time=full_time,
                time_masking=time_masking
            )

            if schedule_name == "deterministic" or schedule_name == "generation":
                ######################################
                ## Embedding for deterministic task ##
                ######################################
                embeddings, embedded_global_conditions, event_token = self.ObjectEncoder(
                    encoded_vectors=full_input_point_cloud,
                    mask=full_input_point_cloud_mask,
                    condition_vectors=full_global_conditions,
                    condition_mask=global_conditions_mask
                )

                ########################################
                ## Output Head for deterministic task ##
                ########################################

                # Assignment head
                # Create output lists for each particle in event.
                assignments = dict()
                detections = dict()
                final_event_token = event_token.clone()
                if self.include_assignment:
                    assignments, detections, event_token_assignments = self.Assignment(
                        x=embeddings,
                        x_mask=full_input_point_cloud_mask,
                        global_condition=embedded_global_conditions,
                        global_condition_mask=global_conditions_mask,
                        event_token=event_token,
                        return_type="process_base"
                    )
                    final_event_token += event_token_assignments

                segmentation_out = {}
                if self.include_segmentation:
                    segmentation_out = self.Segmentation(
                        memory = embeddings,
                        memory_mask = full_input_point_cloud_mask,
                        event_token = event_token
                    )

                    if segmentation_out.get("event-token", None) is not None:
                        final_event_token += segmentation_out["event-token"]

                # Classification head
                classifications = None
                if self.include_classification:
                    classifications = self.Classification(
                        x = embeddings,
                        x_mask = full_input_point_cloud_mask,
                        event_token=event_token
                    )

                # Regression head
                regressions = None
                if self.include_regression:
                    regressions = self.Regression(event_token)

                outputs[schedule_name] = {
                    "classification": classifications,
                    "regression": regressions,
                    "assignments": assignments,
                    "detections": detections,
                    "segmentation-out": segmentation_out,
                }

            #######################################
            ##  Output Head For Diffusion Model  ##
            #######################################
            if self.include_point_cloud_generation and schedule_name == "generation":
                pred_point_cloud_vector = self.ReconGeneration(
                    x=full_input_point_cloud,
                    x_mask=full_input_point_cloud_mask,
                    global_cond=full_global_conditions,
                    global_cond_mask=global_conditions_mask,
                    num_x=num_point_cloud,
                    time=full_time,
                    label=class_label,
                    attn_mask=full_attn_mask,
                    time_masking=time_masking,
                )
                generations["point_cloud"] = {
                    "vector": pred_point_cloud_vector[..., self.generation_pc_indices] * noise_mask,
                    "truth": (truth_input_point_cloud_vector[..., self.generation_pc_indices] * noise_mask).detach(),
                    "mask": noise_mask * full_input_point_cloud_mask
                }

            if self.include_neutrino_generation and schedule_name == "neutrino_generation":
                pred_point_cloud_vector = self.TruthGeneration(
                    x=full_input_point_cloud,
                    x_mask=full_input_point_cloud_mask,
                    global_cond=full_global_conditions,
                    global_cond_mask=global_conditions_mask,
                    num_x=None,
                    time=full_time,
                    label=class_label,
                    attn_mask=full_attn_mask,
                    time_masking=time_masking,
                    position_encode=(self.neutrino_position_encode and schedule_name == "neutrino_generation")
                )
                generations["neutrino"] = {
                    "vector": pred_point_cloud_vector[:, is_invisible_query, :],
                    "truth": truth_invisible_point_cloud_vector.detach(),
                    "mask": invisible_point_cloud_mask.contiguous()
                }

        return {
            "classification": outputs.get("deterministic", {}).get("classification", None),
            "regression": outputs.get("deterministic", {}).get("regression", None),
            "assignments": outputs.get("deterministic", {}).get("assignments", None),
            "detections": outputs.get("deterministic", {}).get("detections", None),
            "classification-noised": outputs.get("generation", {}).get("classification", None),
            "regression-noised": outputs.get("generation", {}).get("regression", None),
            "generations": generations,
            "segmentation-cls": outputs.get("deterministic", {}).get("segmentation-out", {}).get("pred_logits", None),
            # "full_input_point_cloud": full_input_point_cloud,
            # "full_global_conditions": full_global_conditions,
            "alpha": alpha,
            "segmentation-mask": outputs.get("deterministic", {}).get("segmentation-out", {}).get("pred_masks", None),
            "segmentation-aux": outputs.get("deterministic", {}).get("segmentation-out", {}).get("aux_outputs", None)
        }

    def predict_diffusion_vector(
            self, noise_x: Tensor, cond_x: Dict[str, Tensor], time: Tensor, mode: str,
            noise_mask: Optional[Tensor] = None
    ) -> Any | None:

        """
        Predict the number of point clouds in the batch.
        """

        if mode == "global":
            """
            Predict the number of point clouds diffusion vector in the batches.
            noise_x: (batch_size, 1)
            """
            global_conditions = cond_x['conditions'].unsqueeze(1)  # (batch_size, 1, num_conditions)
            global_conditions_mask = cond_x['conditions_mask'].unsqueeze(-1)  # (batch_size, 1)
            class_label = cond_x['classification'].unsqueeze(-1) if 'classification' in cond_x else torch.zeros_like(
                cond_x['conditions_mask']).long()  # (batch_size, 1)
            global_conditions = self.global_normalizer(
                x=global_conditions,
                mask=global_conditions_mask
            )
            predict_num_point_cloud_vector = self.GlobalGeneration(
                x=noise_x,
                time=time,
                global_cond=global_conditions,
                label=class_label
            )
            return predict_num_point_cloud_vector

        elif mode == "event":
            global_conditions = cond_x['conditions'].unsqueeze(1)  # (batch_size, 1, num_conditions)
            global_conditions_mask = cond_x['conditions_mask'].unsqueeze(-1)  # (batch_size, 1)
            class_label = cond_x['classification'].unsqueeze(-1) if 'classification' in cond_x else torch.zeros_like(
                cond_x['conditions_mask']).long()  # (batch_size, 1)
            num_point_cloud = cond_x['num_sequential_vectors'].unsqueeze(-1)  # (batch_size, 1)

            global_feature_mask = torch.zeros_like(global_conditions).float()
            global_feature_mask[..., self.generation_pc_condition_indices] = 1.0

            global_conditions = self.global_normalizer(
                x=global_conditions * global_feature_mask,
                mask=global_conditions_mask
            )
            num_point_cloud = self.num_point_cloud_normalizer(
                x=num_point_cloud,
                mask=None
            )
            global_conditions = self.GlobalEmbedding(
                x=global_conditions,
                mask=global_conditions_mask
            )

            noise_x_mask = torch.zeros_like(noise_x)
            noise_x_mask[..., self.generation_pc_indices] = 1.0
            noise_x = noise_x * noise_x_mask

            local_points = noise_x[..., self.local_feature_indices]
            input_point_cloud = self.PET(
                input_features=noise_x,
                input_points=local_points,
                mask=noise_mask,
                time=time
            )
            pred_point_cloud_vector = self.ReconGeneration(
                x=input_point_cloud,
                x_mask=noise_mask,
                global_cond=global_conditions,
                global_cond_mask=global_conditions_mask,
                num_x=num_point_cloud,
                time=time,
                label=class_label
            )
            return pred_point_cloud_vector * noise_x_mask

        elif mode == "neutrino":
            global_conditions = cond_x['conditions'].unsqueeze(1)  # (batch_size, 1, num_conditions)
            global_conditions_mask = cond_x['conditions_mask'].unsqueeze(-1)  # (batch_size, 1, 1)
            class_label = cond_x['classification'].unsqueeze(-1) if 'classification' in cond_x else torch.zeros_like(
                cond_x['conditions_mask']).long()  # (batch_size, 1)
            # num_point_cloud = cond_x['num_sequential_vectors'].unsqueeze(-1)  # (batch_size, 1)
            input_point_cloud = cond_x['x']
            input_point_cloud_mask = cond_x['x_mask'].unsqueeze(-1)

            input_point_cloud = self.sequential_normalizer(
                x=input_point_cloud,
                mask=input_point_cloud_mask,
            )

            global_conditions = self.global_normalizer(
                x=global_conditions,
                mask=global_conditions_mask
            )

            # num_point_cloud = self.num_point_cloud_normalizer(
            #     x=num_point_cloud,
            #     mask=None
            # )

            invisible_point_cloud_noised = noise_x
            invisible_point_cloud_mask = noise_mask

            # padding invisible features to match input
            if self.invisible_padding > 0:
                invisible_point_cloud_noised = F.pad(
                    invisible_point_cloud_noised, (0, self.invisible_padding), value=0.0
                )

            # Create attention mask
            n_vis = input_point_cloud_mask.shape[1]
            n_invis = invisible_point_cloud_mask.shape[1]
            is_invisible_query = torch.cat([
                torch.zeros(n_vis, dtype=torch.bool, device=self.device),
                torch.ones(n_invis, dtype=torch.bool, device=self.device)
            ], dim=0)  # (L,)
            # Rule: visible query (False) cannot attend to invisible key (True)
            invisible_attn_mask = (~is_invisible_query[:, None]) & is_invisible_query[None, :]  # (L, L) , Q->K

            full_input_point_cloud = torch.cat([input_point_cloud, invisible_point_cloud_noised], dim=1)
            full_input_point_cloud_mask = torch.cat([input_point_cloud_mask, invisible_point_cloud_mask], dim=1)
            full_attn_mask = invisible_attn_mask
            full_time = time
            time_masking = torch.cat(
                [torch.zeros_like(input_point_cloud_mask), invisible_point_cloud_mask], dim=1
            ).float()
            full_global_conditions = self.GlobalEmbedding(
                x=global_conditions,
                mask=global_conditions_mask
            )

            local_points = full_input_point_cloud[..., self.local_feature_indices]
            full_input_point_cloud = self.PET(
                input_features=full_input_point_cloud,
                input_points=local_points,
                mask=full_input_point_cloud_mask,
                attn_mask=full_attn_mask,
                time=full_time,
                time_masking=time_masking
            )

            pred_point_cloud_vector = self.TruthGeneration(
                x=full_input_point_cloud,
                x_mask=full_input_point_cloud_mask,
                global_cond=full_global_conditions,
                global_cond_mask=global_conditions_mask,
                num_x=None,
                time=full_time,
                label=class_label,
                attn_mask=full_attn_mask,
                time_masking=time_masking,
                position_encode=self.neutrino_position_encode
            )

            # remove the padding
            if self.invisible_padding > 0:
                pred_point_cloud_vector = pred_point_cloud_vector[..., :-self.invisible_padding]

            return pred_point_cloud_vector[:, is_invisible_query, :]
        return None

    def shared_step(
            self, batch: Dict[str, Tensor], batch_size,
            train_parameters: Union[dict, None],
            schedules: Union[list[tuple[str, bool]], None] = None
    ) -> dict:
        time = torch.rand((batch_size,), device=batch['x'].device, dtype=batch['x'].dtype)
        output = self.forward(batch, time, progressive_params=train_parameters, schedules=schedules)
        return output

    def freeze_module(self, logical_name: str, cfg: dict):
        """
        Freeze parameters of a head using main_modules_name lookup and freeze config.

        Parameters
        ----------
        logical_name : str
            Logical name used in main_modules_name dict (e.g., "classification_head").
        cfg : dict
            Configuration dict under that head (from config.Classification etc).
        """
        head_module = getattr(self, logical_name, None)
        if head_module is None:
            print(f"[Warning] Attribute '{logical_name}' not found")
            return

        freeze_type = cfg.get("type", "none")
        components = cfg.get("partial_freeze_components", [])

        if freeze_type == "none":
            return

        elif freeze_type == "full":
            for param in head_module.parameters():
                param.requires_grad = False

        elif freeze_type == "partial":
            for name, module in head_module.named_modules():
                if name in components:
                    for param in module.parameters():
                        param.requires_grad = False

        elif freeze_type == "random":
            import random
            freeze_fraction = cfg.get("freeze_fraction", 0.5)
            all_params = list(head_module.parameters())
            num_to_freeze = int(len(all_params) * freeze_fraction)
            to_freeze = random.sample(all_params, num_to_freeze)
            for param in to_freeze:
                param.requires_grad = False

        else:
            raise ValueError(f"Unsupported freeze type: {freeze_type}")
