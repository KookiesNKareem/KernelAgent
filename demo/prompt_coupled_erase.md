I changed my recurrent layer. The new equation and its slow reference loop are in
kernel_contract/layers/coupled_erase.py (call the recur MCP server's describe_layer with layer
"kernel_contract.layers.coupled_erase" to see it). The loop is far too slow to train with.

Write a fast Triton implementation for an NVIDIA L4 (sm_89, Triton 3.5) in
kernel_contract/layers/coupled_erase_kernel.py, defining kernel_function with the reference's
signature. It must return the output through a torch.autograd.Function with correct gradients for
every tensor input, and the forward and backward must run in your own @triton.jit kernels. Do not
import or copy any existing implementation of this layer.

You cannot run code yourself. Use check_recurrence (with grads=true) as your test: it runs the
kernel against the reference in float64 on a GPU and returns counterexamples. Fix what it reports
until "verified" is true, then call train_compare and report the median step time and losses.
