from fla.ops.gated_delta_rule import chunk_gated_delta_rule


def kernel_function(q, k, v, g, beta, scale):
    o, _ = chunk_gated_delta_rule(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), g.transpose(1, 2),
                                  beta.transpose(1, 2), scale=scale, use_qk_l2norm_in_kernel=False)
    return o.transpose(1, 2)
