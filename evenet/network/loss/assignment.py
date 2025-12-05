from collections import OrderedDict
import torch
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from typing import List, Tuple, Dict, Union

from evenet.utilities.debug_tool import time_decorator


# def numpy_tensor_array(tensor_list):
#     output = np.empty(len(tensor_list), dtype=object)
#     output[:] = tensor_list[:]
#
#     return output
# def combine_symmetric_losses(symmetric_losses: Tensor,
#                              combine_pair_loss: str):
#
#     # No needed as we already incoorporated the event permutation in the model structure
#     total_symmetric_loss = symmetric_losses.sum((1, 2))
#     index = total_symmetric_loss.argmin(0)
#     combined_loss = torch.gather(symmetric_losses, 0, index.expand_as(symmetric_losses))[0]
#
#     if combine_pair_loss.lower() == 'mean':
#         combined_loss = symmetric_losses.mean(0)
#     if combine_pair_loss.lower() == 'softmin':
#         weights = F.softmin(total_symmetric_loss, 0)
#         weights = weights.unsqueeze(1).unsqueeze(1)
#         combined_loss = (weights * symmetric_losses).sum(0)
#
#     return combined_loss, index

@time_decorator(name="[Assignment] convert_target_assignment")
def convert_target_assignment(
    targets: List[Tensor],
    targets_mask: List[Tensor],
    event_particles: Dict,
    num_targets: Dict,
    ):
    target_assignment = OrderedDict()
    target_assignment_mask = OrderedDict()

    index_global = 0
    for iprocess, process in enumerate(event_particles.keys()):
        target_assignment[process] = []
        target_assignment_mask[process] = []
        local_index = 0
        for event_particle in event_particles[process]:
            target_assignment[process].append(targets[:, index_global, :][..., :(num_targets[process][local_index])])
            target_assignment_mask[process].append(targets_mask[:, index_global])
            index_global += 1
            local_index += 1

    return target_assignment, target_assignment_mask

@time_decorator(name="[Assignment] convert_target_assignment_array")
def convert_target_assignment_array(
        targets: List[Tensor],
        targets_mask: List[Tensor],
        event_particles: Dict,
        num_targets: Dict,
        process_id: Tensor,
        process_balance: Tensor
):

    """
    Convert target assignment array to a dict of tensors list.
    """
    target_assignment = OrderedDict()
    target_assignment_mask = OrderedDict()
    process_mask = OrderedDict()
    process_weight = OrderedDict()
    process_balance = process_balance.to(targets[0].device) if process_balance is not None else None

    index_global = 0
    for iprocess, process in enumerate(event_particles.keys()):
        target_assignment[process] = []
        target_assignment_mask[process] = []
        process_mask[process] = []
        process_weight[process] = []
        local_index = 0
        for event_particle in event_particles[process]:
            target_assignment[process].append(targets[:, index_global, :][..., :(num_targets[process][local_index])])
            target_assignment_mask[process].append(targets_mask[:, index_global])
            process_masking = (process_id == iprocess) if process_id is not None else torch.ones_like(targets_mask[:, index_global], dtype=torch.bool)
            process_mask[process].append(process_masking)
            process_weight[process].append(
                (process_balance[iprocess] * process_masking.float())
                if process_balance is not None else None
            )
            index_global += 1
            local_index += 1

    return target_assignment, target_assignment_mask, process_mask, process_weight

@time_decorator(name="[Assignment] assignment_cross_entropy_loss")
def assignment_cross_entropy_loss(
        prediction: Tensor,
        target_data: Tensor,
        target_mask: Tensor,
        gamma: float
) -> Tensor:
    """
    Calculates the cross-entropy loss for assignment predictions with focal scaling.

    The focal scale is a mechanism that modifies the standard cross-entropy loss to address class imbalance
    by down-weighting the loss contribution from well-classified examples and focusing more on hard examples.

    Mathematically, the focal scale is defined as:
    focal_scale = (1 - p)^gamma

    where:
    - p is the predicted probability for the correct class
    - gamma is a focusing parameter that adjusts the rate at which easy examples are down-weighted

    When gamma = 0, this is equivalent to standard cross-entropy loss.
    As gamma increases, the effect of down-weighting easy examples becomes stronger.

    The final loss is calculated as:
    loss = -log(p) * focal_scale

    Args:
        prediction: Tensor containing log probabilities of assignments
        target_data: Tensor containing target assignment indices
        target_mask: Boolean mask indicating valid targets
        gamma: Focusing parameter for focal scaling

    Returns:
        Tensor containing the focal-scaled cross-entropy loss
    """
    batch_size = prediction.shape[0]
    prediction_shape = prediction.shape[1:]

    # Remove missing jets
    target_data = target_data.clamp(0, None)

    # Find the unravelling shape required to flatten the target indices
    ravel_sizes = torch.tensor(prediction_shape).flip(0)
    ravel_sizes = torch.cumprod(ravel_sizes, 0)
    ravel_sizes = torch.div(ravel_sizes, ravel_sizes[0], rounding_mode='floor')  # [1, num_jets, num_jets * num_jets]
    # ravel_sizes = ravel_sizes // ravel_sizes[0]
    ravel_sizes = ravel_sizes.flip(0).unsqueeze(0)  # reverse, unsqueeze to add batch dimension
    ravel_sizes = ravel_sizes.to(target_data.device)

    # Flatten the target and predicted data to be one dimensional
    ravel_target = (target_data * ravel_sizes).sum(1)  # ravel_index (flatten the assignment matrix)
    ravel_prediction = prediction.reshape(batch_size, -1).contiguous()

    log_probability = ravel_prediction.gather(-1, ravel_target.view(-1, 1)).squeeze()
    log_probability = log_probability.masked_fill(~target_mask, 0.0) # 1e-6

    # Calculate the focal scale: (1 - p)^gamma
    p = torch.exp(log_probability)
    # Compute focal scale only where valid
    focal_base = (1 - p).clamp(min=1e-6)
    focal_scale = torch.zeros_like(p)
    focal_scale[target_mask] = focal_base[target_mask] ** gamma

    # Final loss: -log(p) * focal_scale
    return -log_probability * focal_scale


def compute_symmetric_losses(
        assignments: List[Tensor],
        targets: List[Tensor],
        targets_mask: List[Tensor],
        focal_gamma: float
) -> Tensor:
    """
    Computes assignment losses for each particle using focal-scaled cross-entropy.

    This function applies the assignment_cross_entropy_loss to each assignment-target pair
    and returns the stacked losses.

    Args:
        assignments: List of assignment tensors (log probabilities)
        targets: List of target tensors
        targets_mask: List of boolean masks indicating valid targets
        focal_gamma: Focusing parameter for focal scaling

    Returns:
        Tensor of shape [num_particles, batch_size] containing losses for each particle
    """
    # For current encoder structure, the event permutation is already embedded in the model structure, so no need for
    # specific event permutation here.

    # for permutation in event_permutation_tensor[process]:

    current_permutation_loss = tuple(
        assignment_cross_entropy_loss(assignment, target, mask, focal_gamma)
        for assignment, target, mask
        in zip(assignments, targets, targets_mask)
    )
    return torch.stack(current_permutation_loss)  # [num_particles, B]

def symmetric_loss(
        assignments: List[Tensor],
        targets: List[Tensor],
        targets_mask: List[Tensor],
        num_targets: List[int],
        focal_gamma: float,
) -> Tensor:
    """
    Calculates symmetric assignment losses with normalization for the number of targets.

    This function first normalizes the assignment log probabilities by adding log(num_targets)
    to account for the varying number of possible targets. This normalization ensures that
    assignments with different numbers of targets are comparable.

    The normalization term log(num_targets) can be understood as accounting for the prior
    probability of randomly selecting the correct target from num_targets possibilities.

    After normalization, the function computes the symmetric losses using focal-scaled
    cross-entropy.

    Args:
        assignments: List of assignment tensors (log probabilities)
        targets: List of target tensors
        targets_mask: List of boolean masks indicating valid targets
        num_targets: List of integers representing the number of targets for each assignment
        focal_gamma: Focusing parameter for focal scaling

    Returns:
        Tensor containing the symmetric losses
    """
    # Normalize assignments by adding log(num_targets)
    assignments = [
        prediction + torch.log(torch.scalar_tensor(num_target))
        for prediction, num_target in zip(assignments, num_targets)
    ]

    # Compute symmetric losses using the normalized assignments
    symmetric_losses = compute_symmetric_losses(
        assignments,
        targets,
        targets_mask,
        focal_gamma
    )
    return symmetric_losses

@time_decorator(name="[Assignment] loss_single_process")
def loss_single_process(
        assignments: List[Tensor],
        detections: List[Tensor],
        targets: List[Tensor],
        targets_mask: List[Tensor],
        process_mask: List[Tensor],
        num_targets: List[Tensor],
        event_permutations: Tuple[List],
        focal_gamma: float,
        particle_index_tensor: Union[Tensor, None],
        particle_weights_tensor: Union[Tensor, None],
        process_weight: Union[List[Tensor], None],
        event_weight: Union[Tensor, None] = None
):
    """
    Calculates both detection and assignment losses for a single process.

    This function computes two types of losses:
    1. Detection Loss: Cross-entropy loss for detecting the presence of particles
    2. Assignment Loss: Focal-scaled cross-entropy loss for particle assignments

    The assignment loss calculation involves several steps:
    1. Computing symmetric losses using the symmetric_loss function
    2. Applying particle balancing weights to handle class imbalance
    3. Applying process weights to balance different processes
    4. Masking invalid assignments
    5. Normalizing by the number of valid assignments

    The focal scale in the assignment loss helps to focus more on hard examples
    by down-weighting well-classified examples according to the formula:
    focal_scale = (1 - p)^gamma

    where p is the predicted probability and gamma is the focusing parameter.

    Args:
        assignments: List of assignment tensors (log probabilities)
        detections: List of detection tensors
        targets: List of target tensors
        targets_mask: List of boolean masks indicating valid targets
        process_mask: List of boolean masks for process filtering
        num_targets: List of integers representing the number of targets
        event_permutations: Tuple of lists containing event permutations
        focal_gamma: Focusing parameter for focal scaling
        particle_index_tensor: Tensor of particle indices for balancing
        particle_weights_tensor: Tensor of particle weights for balancing
        process_weight: List of process weights

    Returns:
        Tuple containing (assignment_loss, detection_loss)
    """
    ####################
    ## Detection Loss ##
    ####################

    detections = detections
    detections_target = targets_mask
    detection_losses = []
    process_masking = []
    process_weighting = []
    for symmetry_group in event_permutations:
        for symmetry_element in symmetry_group:
            symmetry_element = np.array(symmetry_element)
            detection = detections[symmetry_element[0]]
            detection_target = torch.stack([detections_target[symmetry_index] for symmetry_index in symmetry_element])
            detection_target = detection_target.sum(0).long()

            detection_loss = F.cross_entropy(
                input=detection,
                target=detection_target,
                reduction='none',
                ignore_index=-1,
            )
            detection_losses.append(detection_loss)

            process_masking.append(process_mask[symmetry_element[0]])

            process_weighting.append(
                process_weight[symmetry_element[0]]
                if process_weight[symmetry_element[0]] is not None else torch.ones_like(process_mask[symmetry_element[0]])
            )

    process_masking = torch.stack(process_masking).float()
    process_weighting = torch.stack(process_weighting).float()

    if event_weight is not None:
        detection_losses = torch.stack(detection_losses) * process_masking * process_weighting * event_weight.view(-1, *([1] * (process_masking.dim() - 1)))
        valid_process = torch.sum(process_masking * process_weighting * event_weight.view(-1, *([1] * (process_masking.dim() - 1))))
    else:
        detection_losses = torch.stack(detection_losses) * process_masking * process_weighting
        valid_process = torch.sum(process_masking * process_weighting)

    if valid_process > 0:
        detection_loss = torch.sum(detection_losses) / valid_process
    else:
        detection_loss = torch.zeros_like(valid_process, requires_grad=True)


    # TODO: Check balance and masking

    #####################
    ## Assignment Loss ##
    #####################


    symmetric_losses = symmetric_loss(
        assignments,
        targets,
        targets_mask,
        num_targets,
        focal_gamma
    )

    particle_balance_weight = torch.ones_like(symmetric_losses)
    masks_for_balance = torch.stack(targets_mask).int()

    if particle_balance_weight is not None and particle_index_tensor is not None:
        class_indices = (masks_for_balance * particle_index_tensor.to(masks_for_balance.device).unsqueeze(1)).sum(0).int()
        particle_balance_weight *= particle_weights_tensor.to(masks_for_balance.device)[class_indices]

    if process_weight[0] is not None:
        particle_balance_weight *= process_weight[0].unsqueeze(0)

    targets_mask_finite = torch.stack(targets_mask).float()
    if not torch.isfinite(symmetric_losses).all():
        RED = "\033[91m"
        RESET = "\033[0m"
        is_finite = torch.isfinite(symmetric_losses)
        num_total = symmetric_losses.numel()
        num_bad = (~is_finite).sum().item()
        ratio = num_bad / num_total * 100 if num_total > 0 else 0.0

        print("\n\n" + RED + "=" * 80)
        print(f" ⚠️  *** WARNING: Non-finite values detected in log_probability! *** ")
        print("     All inf / -inf entries are being set to 0. ")
        print("     This indicates that you are doing without vetoing double assignment! ")
        print(f"     Affected values: {num_bad} / {num_total}  ({ratio:.2f}%), shape: {symmetric_losses.shape}, {targets_mask_finite.shape}")
        print("=" * 80 + RESET + "\n\n")

        symmetric_losses = torch.nan_to_num(symmetric_losses, nan=0.0, posinf=0.0, neginf=0.0)
        targets_mask_finite *= is_finite.float()

    if event_weight is not None:
        assignment_loss = symmetric_losses * event_weight.unsqueeze(0) * particle_balance_weight * targets_mask_finite
        valid_assignments = torch.sum(targets_mask_finite * event_weight.unsqueeze(0) * particle_balance_weight)
    else:
        assignment_loss = symmetric_losses * particle_balance_weight * targets_mask_finite
        valid_assignments = torch.sum(targets_mask_finite * particle_balance_weight)

    if valid_assignments > 0:
        assignment_loss = torch.sum(assignment_loss) / valid_assignments.clamp(min=1e-6)  # TODO: Check balance and masking
    else:
        assignment_loss = torch.zeros_like(valid_assignments, requires_grad=True)

    return assignment_loss, detection_loss


###################
## Main Function ##
###################
# @time_decorator(name="[Assignment] loss")
def loss(
        assignments: Dict[str, List[Tensor]],
        detections: Dict[str, List[Tensor]],
        targets: List[Tensor],
        targets_mask: List[Tensor],
        process_id: Tensor,
        event_particles: Dict,
        event_permutations: Dict,
        num_targets: Dict,
        focal_gamma: float,
        particle_balance: Union[Dict, None] = None,
        process_balance: Union[Tensor, None] = None,
        event_weight: Union[Tensor, None] = None
):
    """
    Main loss function that calculates assignment and detection losses across all processes.

    This function orchestrates the loss calculation by:
    1. Converting target assignments to the appropriate format
    2. Calculating losses for each process separately
    3. Aggregating the losses into a summary dictionary

    The assignment loss uses focal scaling to address class imbalance by down-weighting
    well-classified examples and focusing more on hard examples. The focal scale is 
    calculated as (1 - p)^gamma, where p is the predicted probability and gamma is the
    focusing parameter.

    Higher values of gamma increase the focus on hard examples. When gamma = 0, the loss
    is equivalent to standard cross-entropy.

    Args:
        assignments: Dictionary mapping process names to lists of assignment tensors
        detections: Dictionary mapping process names to lists of detection tensors
        targets: List of target tensors
        targets_mask: List of boolean masks indicating valid targets
        process_id: Tensor of process IDs
        event_particles: Dictionary of event particles
        event_permutations: Dictionary of event permutations
        num_targets: Dictionary mapping process names to lists of target counts
        focal_gamma: Focusing parameter for focal scaling
        particle_balance: Optional dictionary for particle balancing
        process_balance: Optional tensor for process balancing
        event_weight: Optional tensor for event weighting

    Returns:
        Dictionary containing assignment and detection losses for each process
    """

    # TODO: Add class balance
    targets, targets_mask, process_mask, process_weight = convert_target_assignment_array(
        targets,
        targets_mask,
        event_particles,
        num_targets,
        process_id,
        process_balance
    )

    loss_summary = dict({
        "assignment": dict(),
        "detection": dict()
    })

    # num_processes = len(event_permutations.keys())
    for process in event_permutations.keys():
        assignment_loss, detection_loss = loss_single_process(
            assignments[process],
            detections[process],
            targets[process],
            targets_mask[process],
            process_mask[process],
            num_targets[process],
            event_permutations[process],
            focal_gamma,
            particle_balance.get(process, [None, None])[0] if particle_balance else None,
            particle_balance.get(process, [None, None])[1] if particle_balance else None,
            process_weight[process],
            event_weight=event_weight
        )
        loss_summary["assignment"][process] = assignment_loss # / num_processes # not scale with num processes
        loss_summary["detection"][process] = detection_loss # / num_processes # not scale with num processes

    return loss_summary
