from typing import Tuple
import torch
import random

def corrupt_progress(
	interim_thought: torch.Tensor,
	out_head: torch.nn.Module,
	tgt_vocab_size: int = 2,
	epsilon: float = 2e-4,
	steps: int = 5
) -> Tuple[torch.Tensor, int]:
	
	"""
	Corrupts the given interim_thought using backpropagation steps.
	Few GD steps wouldn't converge, giving us a slightly more corrupted version
	than what the GD process would have targeted towards. Saves time - 10ms on a T4

	Args:
		interim_thought (torch.Tensor): Input tensor to be perturbed.
		epsilon (float, optional): Step size for perturbation. Defaults to 2e-4.
		steps (int, optional): Number of backpropagation steps. Defaults to 5.

	Returns:
		torch.Tensor: Perturbed thought after backpropagation steps.
		int: Number of errors generated during perturbation.
	"""
	# Make sure interim_thought requires gradient
	interim_thought.requires_grad = True

	# Generate a list of unique indices to corrupt
	og_output = torch.softmax(out_head(interim_thought), dim=-1).argmax(-1)
	corrupt_indices = random.sample(range(len(og_output)), 3)

	# Corrupt the bits at the selected indices
	corrupted_output = og_output.clone()
	for index in corrupt_indices:
		corrupted_output[index] = 1 - corrupted_output[index]

	target_output = torch.nn.functional.one_hot(corrupted_output, num_classes=tgt_vocab_size).float()

	# Define a loss function
	loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

	for _ in range(steps):
		# Compute the original output
		original_output = torch.softmax(out_head(interim_thought), dim=-1)

		# Compute the loss between the original output and the target output
		loss = loss_fn(original_output, target_output)

		# Backpropagate to compute gradients
		loss.backward()

		# Update interim_thought using gradient descent
		with torch.no_grad():
			interim_thought += epsilon * interim_thought.grad.sign()

		# Zero out the gradients for the next iteration
		interim_thought.grad.zero_()

	perturbed_thought = interim_thought
	perturbed_output = torch.softmax(out_head(perturbed_thought), dim=-1)

	errors_generated = torch.count_nonzero((perturbed_output.argmax(-1) == og_output) == 0)

	return perturbed_thought, errors_generated

if __name__ == "__main__":
	# Example usage
	interim_thought = torch.randn(96, 96)
	out_head = torch.nn.Linear(96, 2)
	perturbed_thought, errors_generated = corrupt_progress(interim_thought, out_head, tgt_vocab_size=2)

	print("Perturbed Thought:\n", perturbed_thought)
	print("Errors Generated:", errors_generated)