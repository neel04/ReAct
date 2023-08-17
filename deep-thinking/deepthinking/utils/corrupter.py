from typing import Tuple
import torch
import random

def corrupt_progress(
    input_tensor: torch.Tensor,
    out_head: torch.nn.Module,
    tgt_vocab_size: int = 3,
    epsilon: float = 2e-4,
    steps: int = 5,
) -> Tuple[torch.Tensor, int]:
    """
    Corrupts the given interim_thought using backpropagation steps.
    Few GD steps wouldn't converge, giving us a slightly more corrupted version
    than what the GD process would have targeted towards. Saves time - 10ms on a T4

    Args:
        input (torch.Tensor): Input tensor to be perturbed. Can be batched.
        out_head (torch.nn.Module): Neural network head for output calculation.
        tgt_vocab_size (int, optional): Number of classes in the target vocabulary. Defaults to 2.
        epsilon (float, optional): Step size for perturbation. Defaults to 2e-4.
        steps (int, optional): Number of backpropagation steps. Defaults to 5.

    Returns:
        torch.Tensor: Perturbed thought after backpropagation steps.
        int: Number of errors generated during perturbation.
    """
    if input_tensor is None:
        return None, [0]

    # Make sure input requires gradient
    n = 2  # number of bits to corrupt
    vanilla_tensor = input_tensor.detach().clone()
    vanilla_tensor.requires_grad = True
    out_head.requires_grad = False

    # Generate a list of unique indices to corrupt for each batch element
    og_output = torch.softmax(out_head(vanilla_tensor), dim=-1).argmax(-1)
    corrupt_indices = [random.sample(range(len(og)), n) for og in og_output]

    # Corrupt the bits at the selected indices for each batch element
    corrupted_output = og_output.clone()

    for i, indices in enumerate(corrupt_indices):
		# TODO: This is a ugly hack
		# Replace the corrupted bits with random bits, withing class range
        corrupted_output[i, indices] = torch.Tensor(
            [random.choices(range(0, tgt_vocab_size))] * n
            ).reshape(-1).long().to(corrupted_output.device)

    target_output = torch.nn.functional.one_hot(corrupted_output.long(), num_classes=tgt_vocab_size).float()

    # Define a loss function
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    for _ in range(steps):
        # Compute the original output
        original_output = torch.softmax(out_head(vanilla_tensor), dim=-1)

        # Compute the loss between the original output and the target output
        loss = loss_fn(original_output, target_output)

        # Backpropagate to compute gradients
        loss.backward()

        # Update vanilla_tensor using gradient descent
        with torch.no_grad():
            vanilla_tensor += epsilon * vanilla_tensor.grad.sign()

        # Zero out the gradients for the next iteration
        vanilla_tensor.grad.zero_()

    perturbed_thought = vanilla_tensor.clone()
    perturbed_output = torch.softmax(out_head(perturbed_thought), dim=-1)

    # list of errors per batch element
    errors_generated = [
        torch.count_nonzero((perturbed_output[i].argmax(-1) == og_output[i]) == 0).item()
        for i in range(og_output.size(0))
    ]

    return perturbed_thought, errors_generated

if __name__ == "__main__":
    # Example usage
    interim_thought = torch.randn(96, 96)
    out_head = torch.nn.Linear(96, 2)
    perturbed_thought, errors_generated = corrupt_progress(interim_thought, out_head, tgt_vocab_size=2)

    print("Perturbed Thought:\n", perturbed_thought)
    print("Errors Generated:", errors_generated)