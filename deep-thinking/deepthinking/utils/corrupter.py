from typing import Tuple, List

import torch
import random
import torch.optim as optim

def corrupt_progress(
    input_tensor: torch.Tensor,
    out_head: torch.nn.Module,
    tgt_vocab_size: int = 3,
    steps: int = 7,
    learning_rate: float = 2e-0,
    weight_decay: float = 1e-5,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Corrupts the given interim_thought using backpropagation steps.
    Few GD steps wouldn't converge, giving us a slightly more corrupted version
    than what the GD process would have targeted towards. Saves time - 1000ms on a T4

    Args:
        input (torch.Tensor): Input tensor to be perturbed. Can be batched.
        out_head (torch.nn.Module): Neural network head for output calculation.
        tgt_vocab_size (int, optional): Number of classes in the target vocabulary. Defaults to 2.
        steps (int, optional): Number of backpropagation steps. Defaults to 5.
        learning_rate (float, optional): Learning rate for optimizer. Defaults to 0.01.
        weight_decay (float, optional): Weight decay for optimizer. Defaults to 1e-5.

    Returns:
        torch.Tensor: Perturbed thought after backpropagation steps.
        List: Number of errors generated during perturbation.
    """
    if input_tensor is None:
        return input_tensor, [0]

    # Make sure input requires gradient
    n = 3  # number of bits to corrupt
    vanilla_tensor = input_tensor.detach().clone()
    vanilla_tensor.requires_grad = True
    out_head.requires_grad = False

    # Generate a list of unique indices to corrupt for each batch element
    og_output = torch.softmax(out_head(vanilla_tensor), dim=-1).argmax(-1)
    corrupt_indices = [random.sample(range(len(og)), n) for og in og_output]

    # Corrupt the bits at the selected indices for each batch element
    corrupted_output = og_output.clone()

    for i, indices in enumerate(corrupt_indices):
        # Replace the corrupted bits with random bits within class range
        corrupted_output[i, indices] = torch.randint(low=0, high=tgt_vocab_size, size=(n,), device=corrupted_output.device)

    target_output = torch.nn.functional.one_hot(corrupted_output.long(), num_classes=tgt_vocab_size).float()

    # Use AdamW optimizer
    optimizer = optim.AdamW([vanilla_tensor], lr=learning_rate, weight_decay=weight_decay)

    for _ in range(steps):
        # Compute the original output
        original_output = torch.softmax(out_head(vanilla_tensor), dim=-1)

        # Compute the loss between the original output and the target output
        loss = torch.nn.functional.cross_entropy(original_output, target_output, reduction='mean')

        # Zero the gradients
        optimizer.zero_grad()

        # Backpropagate to compute gradients
        loss.backward()

        # Update the tensor using optimizer
        optimizer.step()

    perturbed_thought = vanilla_tensor.clone()
    perturbed_output = torch.softmax(out_head(perturbed_thought), dim=-1)

    # list of errors per batch element
    errors_generated = [
        torch.count_nonzero((perturbed_output[i].argmax(-1) == og_output[i]) == 0).item()
        for i in range(og_output.size(0))
    ]

    return perturbed_thought, errors_generated

class Adversarial_Perturbation:
    """Generates progressive loss for training, and applies adversarial perturbation to the thought tensor"""
    steps: int = 10
    learning_rate: float = 4

    def __init__(self, head):
        self.head = head

    def _corrupt_progress(self, interim_thought: torch.Tensor, output_head: torch.nn.Module) -> Tuple[torch.Tensor, List[int]]:
        # Corrupt the thought tensor. override defaults as needed
        interim_thought, num_errors = corrupt_progress(interim_thought, output_head, learning_rate=self.learning_rate, steps=self.steps)
        interim_thought = interim_thought.detach() if interim_thought is not None else interim_thought

        return interim_thought, num_errors

    def _disable_gradients(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def _enable_gradients(self, module):
        for param in module.parameters():
            param.requires_grad = True
    
    def perturb(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        self._disable_gradients(self.head)
        interim_thought, num_errors = self._corrupt_progress(input_tensor, self.head)
        self._enable_gradients(self.head)

        return interim_thought, num_errors

if __name__ == "__main__":
    # Example usage
    interim_thought = torch.randn(96, 96)
    out_head = torch.nn.Linear(96, 2)
    perturbed_thought, errors_generated = corrupt_progress(interim_thought, out_head, tgt_vocab_size=2)

    print("Perturbed Thought:\n", perturbed_thought)
    print("Errors Generated:", errors_generated)