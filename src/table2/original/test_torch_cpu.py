"""Run in the server's existing PyTorch environment; does not need a GPU."""
import torch
import torch.nn.functional as F
from losses import weighted_ce_per_row

x = torch.tensor([[.2, -.1]], requires_grad=True)
y = torch.tensor([0])
base = weighted_ce_per_row(x, y)
weighted = weighted_ce_per_row(x, y, [2., 1.])
assert torch.allclose(weighted, 2 * base)
assert torch.allclose(F.cross_entropy(x, y, weight=torch.tensor([2., 1.])), base)
weighted.backward()
assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0

x = torch.randn(4, 2, requires_grad=True)
y = torch.tensor([0, 1, -100, 0])
batch_loss = weighted_ce_per_row(x, y, [2., .5])
micro_loss = sum(weighted_ce_per_row(x[i:i+1], y[i:i+1], [2., .5]) for i in range(4)) / 4
assert torch.allclose(batch_loss, micro_loss)
grad_a = torch.autograd.grad(batch_loss, x, retain_graph=True)[0]
grad_b = torch.autograd.grad(micro_loss, x)[0]
assert torch.allclose(grad_a, grad_b)

x = torch.randn(3, 8, requires_grad=True)
zero = weighted_ce_per_row(x, torch.full((3,), -100), [1.] * 8)
zero.backward()
assert zero.item() == 0 and torch.isfinite(x.grad).all() and x.grad.abs().sum() == 0
print('PASS: microbatch=1 weights retained, accumulation/batch gradient equivalence, all-masked finite zero gradients.')
