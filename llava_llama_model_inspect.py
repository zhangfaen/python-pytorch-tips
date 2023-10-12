# inspect https://github.com/haotian-liu/LLaVA
# /home/zhangfaen/dev/LLaVA/llava/serve/cli.py
# import pdb; pdb.set_trace()

# v1 = {n: m for n, m in model.named_modules()}
# sum([p.numel() for p in v1['model.layers'].parameters()])

# LlavaLlamaForCausalLM(
#   (model): LlavaLlamaModel(
#     (embed_tokens): Embedding(32000, 4096, padding_idx=0)
#     (layers): ModuleList(
#       (0-31): 32 x LlamaDecoderLayer(
#         (self_attn): LlamaAttention(
#           (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
#           (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
#           (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
#           (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
#           (rotary_emb): LlamaRotaryEmbedding()
#         )
#         (mlp): LlamaMLP(
#           (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
#           (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
#           (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
#           (act_fn): SiLUActivation()
#         )
#         (input_layernorm): LlamaRMSNorm()
#         (post_attention_layernorm): LlamaRMSNorm()
#       )
#     ) # 6,476,267,520 parameters 
#     (norm): LlamaRMSNorm()
#     (vision_tower): CLIPVisionTower(
#       (vision_tower): CLIPVisionModel(
#         (vision_model): CLIPVisionTransformer(
#           (embeddings): CLIPVisionEmbeddings(
#             (patch_embedding): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14), bias=False)
#             (position_embedding): Embedding(577, 1024)
#           )
#           (pre_layrnorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
#           (encoder): CLIPEncoder(
#             (layers): ModuleList(
#               (0-23): 24 x CLIPEncoderLayer(
#                 (self_attn): CLIPAttention(
#                   (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
#                   (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
#                   (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
#                   (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
#                 )
#                 (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
#                 (mlp): CLIPMLP(
#                   (activation_fn): QuickGELUActivation()
#                   (fc1): Linear(in_features=1024, out_features=4096, bias=True)
#                   (fc2): Linear(in_features=4096, out_features=1024, bias=True)
#                 )
#                 (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
#               )
#             )
#           )
#           (post_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
#         )
#       )
#     ) # 303,507,456 parameters
#     (mm_projector): Sequential(
#       (0): Linear(in_features=1024, out_features=4096, bias=True)
#       (1): GELU(approximate='none')
#       (2): Linear(in_features=4096, out_features=4096, bias=True)
#     ) # 20979712
#   )
#   (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
# )