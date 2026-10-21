"""MOSS-Image reference-image (edit) conditioning for ComfyUI.

Adds Kontext-style multi-reference support to MOSS-Image without touching core:

  - ``TextEncodeMOSSImageOstrisEdit`` encodes the prompt + reference images
    through the MOSS-Image Qwen3.5 text encoder (edit-plus style ``Picture N:``
    vision placeholders inside MOSS's own chat template) and attaches the VAE
    reference latents to the conditioning.
  - ``MOSSImageOstrisEditModelPatch`` patches the MOSS-Image model (via
    ModelPatcher object patches, applied/removed per-workflow) so those
    reference latents are appended to the token sequence after the text,
    each on its own index of the RoPE axis the base model leaves at 0, and
    conditioned at t=0 -- the ComfyUI Flux/QwenImage "index_timestep_zero"
    reference method.

Both mirror the ai-toolkit ``moss_image`` edit training implementation
(``model_kwargs.edit``): VL images are downscaled (never upscaled) to fit
384x384 total pixels; reference latents are encoded at the size the image is
fed in (snapped to /16, the FLUX.2 VAE's latent grid) -- scale the reference
upstream (e.g. FluxKontextImageScale) to pick its token budget.

``MOSSImageOstrisEditModelPatch``'s ``kv_cache`` toggle (default ON, matching
ai-toolkit's edit models) is for LoRAs trained with ai-toolkit's ``kv_cache``
model kwarg, where reference tokens attend only to each other. Their per-block
K/V are then timestep-invariant, so they are precomputed in a single ref-only
pass at t=0 and injected as extra attention keys on every denoising step --
the ref tokens never ride along in the per-step sequence (OminiControl2-style
conditioning feature reuse). A LoRA trained without ``kv_cache`` needs the
toggle off.

Layer skipping (the MOSS default: block 22 skipped on the negative branch via
the SkipLayerGuidanceDiTSimple node) is honored in every path, including the
reference K/V precompute.
"""

import math

import torch

import comfy.conds
import comfy.ldm.flux.math
import comfy.model_prefetch
import comfy.utils
import node_helpers
from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.moss_image.model import modulated_rms_norm, rope_split_half, situ

# ---------------------------------------------------------------------------
# Text encode (VL images -> conditioning, refs -> reference_latents)
# ---------------------------------------------------------------------------

VLM_MAX_PIXELS = 384 * 384  # Qwen3.5 sees a coarse reference
REF_SNAP = 16  # FLUX.2 VAE: f8 conv + 2x2 packing -> 16x latent grid
TIME_DIM = 256


def _fit_area(samples, max_pixels):
    """Downscale (never upscale) a (B, C, H, W) image to fit ``max_pixels``,
    keeping aspect."""
    h, w = samples.shape[2], samples.shape[3]
    scale = min(1.0, math.sqrt(max_pixels / (w * h)))
    nw, nh = max(round(w * scale), 1), max(round(h * scale), 1)
    if (nh, nw) == (h, w):
        return samples
    return comfy.utils.common_upscale(samples, nw, nh, "area", "disabled")


def _snap(samples, snap):
    """Round a (B, C, H, W) image's dims to the nearest multiple of ``snap``.
    The reference is otherwise used at the size it is fed in."""
    h, w = samples.shape[2], samples.shape[3]
    nw = max(round(w / snap) * snap, snap)
    nh = max(round(h / snap) * snap, snap)
    if (nh, nw) == (h, w):
        return samples
    return comfy.utils.common_upscale(samples, nw, nh, "area", "disabled")


class TextEncodeMOSSImageOstrisEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                "vae": ("VAE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "ostris/moss_image"
    DESCRIPTION = (
        "Encode a prompt with optional reference images for a MOSS-Image edit "
        "LoRA. Images are fed to the Qwen3.5 text encoder (needs a text encoder "
        "checkpoint that includes the vision weights) and, when a VAE is "
        "connected, attached as reference latents for MOSSImageOstrisEditModelPatch "
        "at the resolution they are fed in (scale them upstream; 1MP recommended)."
    )

    def encode(self, clip, prompt, vae=None, image1=None, image2=None, image3=None):
        images_vl = []
        ref_latents = []
        image_prompt = ""

        for image in (image1, image2, image3):
            if image is None:
                continue
            samples = image.movedim(-1, 1)
            images_vl.append(_fit_area(samples, VLM_MAX_PIXELS).movedim(1, -1))
            if vae is not None:
                # no resize: the reference's resolution is the user's call
                s = _snap(samples, REF_SNAP)
                ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))
            image_prompt += (
                "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(
                    len(images_vl)
                )
            )

        # MOSS's own template (user turn + open think block), applied by the
        # moss_image tokenizer: the trained LoRA saw the Picture markers at the
        # start of the user content.
        tokens = clip.tokenize(image_prompt + prompt, images=images_vl)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(
                conditioning, {"reference_latents": ref_latents}, append=True
            )
        return (conditioning,)


# ---------------------------------------------------------------------------
# Model patch (reference latents -> t=0 tokens in the sequence)
# ---------------------------------------------------------------------------


def _image_ids(index, frame, h, w, device):
    """(h*w, 4) RoPE coordinates (r, f, h, w) of one latent grid: the base
    model's image has r=0 and f=text length; reference i uses r=i+1 with the
    same frame coordinate and its own centered grid."""
    ids = torch.zeros(h * w, 4, device=device, dtype=torch.float32)
    ids[:, 0] = index
    ids[:, 1] = frame
    ids[:, 2] = (torch.arange(h, device=device, dtype=torch.float32) - (h - 1) / 2).repeat_interleave(w)
    ids[:, 3] = (torch.arange(w, device=device, dtype=torch.float32) - (w - 1) / 2).repeat(h)
    return ids


def _text_ids(text_len, device):
    ids = torch.zeros(text_len, 4, device=device, dtype=torch.float32)
    ids[:, 1] = torch.arange(text_len, device=device, dtype=torch.float32)
    return ids


def _pe(dit, ids):
    ids = ids.unsqueeze(0)
    if hasattr(dit, "pe_embedder"):  # ComfyUI <= v0.42.1
        pe = dit.pe_embedder(ids)
    else:
        # same per-axis rope as MOSSImage.rope() (core dropped pe_embedder for rope scaling)
        pe = torch.cat([comfy.ldm.flux.math.rope(ids[..., i], d, 10000) for i, d in enumerate(dit.rope_axes_dim)], dim=-3).unsqueeze(1)
    return pe.transpose(1, 2).contiguous()


def _modulation(dit, vec):
    return dit.shared_block_modulation(vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)


def _time_vec(dit, timestep, dtype):
    return dit.time_embedding(timestep_embedding(timestep, TIME_DIM).to(dtype))


def _pack_refs(dit, ref_latents, bs, text_len, device, dtype):
    """Reference latents -> tokens (B, Lr, C) + RoPE ids (Lr, 4)."""
    tokens, ids = [], []
    for i, ref in enumerate(ref_latents):
        if ref.ndim == 5 and ref.shape[2] == 1:
            ref = ref.squeeze(2)
        ref = comfy.utils.repeat_to_batch_size(ref.to(device, dtype), bs)
        rh, rw = ref.shape[-2], ref.shape[-1]
        tokens.append(ref.flatten(2).transpose(1, 2))
        ids.append(_image_ids(i + 1, text_len, rh, rw, device))
    return torch.cat(tokens, dim=1), torch.cat(ids, dim=0)


def _qkv(block, y):
    if block.fused_qkv:
        return block.proj["qkv"](y).chunk(3, dim=-1)
    return block.proj["q"](y), block.proj["k"](y), block.proj["v"](y)


def _attention(q, k, v, block, pe, transformer_options, kv_capture=None, kv_cache=None):
    """comfy moss_image ``attention`` with optional K/V capture or cached-K/V
    injection. Keys are captured *before* RoPE (normed only) so one cache can
    serve any text length; injected keys must already be RoPE'd at their
    positions for the current pass."""
    B, N, _ = q.shape
    heads = block.num_heads
    q = rope_split_half(block.q_norm(q.view(B, N, heads, -1)), pe)
    k = block.k_norm(k.view(B, N, heads, -1))
    v = v.view(B, N, heads, -1)
    if kv_capture is not None:
        kv_capture.append((k, v))
    k = rope_split_half(k, pe)
    if kv_cache is not None:
        k = torch.cat((k, kv_cache[0].to(k.dtype)), dim=1)
        v = torch.cat((v, kv_cache[1].to(v.dtype)), dim=1)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    return optimized_attention(
        q, k, v, heads, skip_reshape=True, transformer_options=transformer_options
    )


def _block_kv_forward(block, x, mod, pe, transformer_options, kv_capture=None, kv_cache=None):
    """Block forward (single-span modulation) with K/V capture / injection."""
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod
    y = modulated_rms_norm(x, block.norm1, scale_a, shift_a)
    q, k, v = _qkv(block, y)
    attn = _attention(q, k, v, block, pe, transformer_options, kv_capture, kv_cache)
    x = x + block.proj["o"](attn) * gate_a
    y = modulated_rms_norm(x, block.norm2, scale_f, shift_f)
    hidden, gate = block.proj["ffn_up_gate"](y).chunk(2, dim=-1)
    return x + block.proj["ffn_down"](situ(hidden, gate)) * gate_f


def _block_ref_forward(block, x, mod, ref_mod, split, pe, transformer_options):
    """Block forward with per-span modulation: tokens ``[:split]`` (noisy image
    + text) use the real-t modulation, tokens ``[split:]`` (clean reference
    tokens) the t=0 one. References attend everything (the non-kv_cache
    training mode)."""
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod
    rshift_a, rscale_a, rgate_a, rshift_f, rscale_f, rgate_f = ref_mod

    def norm_mod(norm, h, scale, shift, rscale, rshift):
        return torch.cat(
            (
                modulated_rms_norm(h[:, :split].contiguous(), norm, scale, shift),
                modulated_rms_norm(h[:, split:].contiguous(), norm, rscale, rshift),
            ),
            dim=1,
        )

    def gate(h, g, rg):
        return torch.cat((h[:, :split] * g, h[:, split:] * rg), dim=1)

    y = norm_mod(block.norm1, x, scale_a, shift_a, rscale_a, rshift_a)
    q, k, v = _qkv(block, y)
    x = x + gate(block.proj["o"](_attention(q, k, v, block, pe, transformer_options)), gate_a, rgate_a)
    y = norm_mod(block.norm2, x, scale_f, shift_f, rscale_f, rshift_f)
    hidden, g = block.proj["ffn_up_gate"](y).chunk(2, dim=-1)
    return x + gate(block.proj["ffn_down"](situ(hidden, g)), gate_f, rgate_f)


def _run_blocks(dit, h, block_fn, vec, mod, pe, transformer_options):
    """Iterate the DiT blocks like comfy's native forward: honors
    ``patches_replace`` (layer skipping / SLG) and the weight prefetch queue."""
    blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
    transformer_options["total_blocks"] = len(dit.layers)
    transformer_options["block_type"] = "single"
    device = h.device
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(dit.layers), device, transformer_options)
    for i, block in enumerate(dit.layers):
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, h.dtype)
        transformer_options["block_index"] = i
        if ("single_block", i) in blocks_replace:

            def block_wrap(args, block=block, i=i):
                return {"img": block_fn(block, args["img"], i)}

            h = blocks_replace[("single_block", i)](
                {"img": h, "vec": vec, "mod": mod, "pe": pe, "transformer_options": transformer_options},
                {"original_block": block_wrap},
            )["img"]
        else:
            h = block_fn(block, h, i)
    comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)
    return h


def _skipped_blocks(transformer_options):
    replace = transformer_options.get("patches_replace", {}).get("dit", {})
    return tuple(sorted(k[1] for k in replace if isinstance(k, tuple) and k[0] == "single_block"))


def _finish(dit, h, vec, n_img, B, H, W):
    head_shift, head_scale = dit.video_head_modulation(vec).unflatten(-1, (2, -1)).unsqueeze(2).unbind(1)
    out = dit.final_linear_video(
        modulated_rms_norm(h[:, :n_img].contiguous(), dit.final_norm_video, head_scale, head_shift)
    )
    return out.transpose(1, 2).reshape(B, dit.latent_channels, H, W)


def _forward_with_refs(dit, x, timestep, context, ref_latents, transformer_options):
    """MOSS-Image forward with reference latents appended after the text at
    t=0 ("index_timestep_zero"). Mirrors comfy.ldm.moss_image.model.MOSSImage._forward
    with the ref packing / per-span modulation additions."""
    B, _, H, W = x.shape
    dtype, device = x.dtype, x.device
    transformer_options = transformer_options.copy()
    n_img, text_len = H * W, context.shape[1]

    reftok, refids = _pack_refs(dit, ref_latents, B, text_len, device, dtype)
    ids = torch.cat((_image_ids(0, text_len, H, W, device), _text_ids(text_len, device), refids), dim=0)
    pe = _pe(dit, ids)

    vec = _time_vec(dit, timestep, dtype)
    vec0 = _time_vec(dit, torch.zeros_like(timestep), dtype)
    mod, ref_mod = _modulation(dit, vec), _modulation(dit, vec0)

    txt = dit.text_embedder(context, pe[:, n_img : n_img + text_len], transformer_options)
    h = torch.cat((dit.video_embedder(x.flatten(2).transpose(1, 2)), txt, dit.video_embedder(reftok)), dim=1)
    split = n_img + text_len

    def block_fn(block, h, i):
        return _block_ref_forward(block, h, mod, ref_mod, split, pe, transformer_options)

    h = _run_blocks(dit, h, block_fn, vec, mod, pe, transformer_options)
    return _finish(dit, h, vec, n_img, B, H, W)


# ---------------------------------------------------------------------------
# Isolated ref attention: one-pass ref K/V precompute + cached denoising
# ---------------------------------------------------------------------------


def _precompute_ref_kv(dit, x, timestep, ref_latents, transformer_options, reuse=None, reuse_upto=0, store_device=None):
    """Run only the clean reference tokens through the blocks at t=0 and record
    each block's K (post-norm, pre-RoPE) and V.

    A kv_cache LoRA never lets ref tokens see the text / noisy tokens, and all
    ref tokens share one frame coordinate, so this pass is independent of both
    the text length (RoPE is relative; the shared offset cancels) and the batch
    size. It therefore runs once at the refs' own batch size (normally 1) and
    is shared by cond, uncond and every batch item.

    ``reuse``/``reuse_upto``: blocks ``< reuse_upto`` are identical to an
    existing cache built with a different skip-layer set (the two passes only
    diverge at the first block where their skip sets differ), so those entries
    are shared by reference instead of stored twice. Skipped blocks record
    nothing; the live pass skips them too."""
    dtype, device = x.dtype, x.device
    transformer_options = transformer_options.copy()
    rb = max(r.shape[0] for r in ref_latents)
    reftok, refids = _pack_refs(dit, ref_latents, rb, 0, device, dtype)
    pe = _pe(dit, refids)
    vec0 = _time_vec(dit, torch.zeros(rb, device=device, dtype=timestep.dtype), dtype)
    mod0 = _modulation(dit, vec0)
    h = dit.video_embedder(reftok)

    ref_kv = {}

    def block_fn(block, h, i):
        if reuse is not None and i < reuse_upto:
            if i in reuse:
                ref_kv[i] = reuse[i]
            return _block_kv_forward(block, h, mod0, pe, transformer_options)
        cap = []
        h = _block_kv_forward(block, h, mod0, pe, transformer_options, kv_capture=cap)
        k, v = cap[0]
        if store_device is not None and store_device.type == "cpu":
            k = k.to("cpu").pin_memory() if torch.cuda.is_available() else k.to("cpu")
            v = v.to("cpu").pin_memory() if torch.cuda.is_available() else v.to("cpu")
        ref_kv[i] = (k, v)
        return h

    _run_blocks(dit, h, block_fn, vec0, mod0, pe, transformer_options)
    return ref_kv


def _forward_with_cached_refs(dit, x, timestep, context, ref_kv, ref_pe, transformer_options):
    """Denoising pass with the reference tokens replaced by their cached K/V:
    the live sequence is just noisy image + text tokens, and every block's
    attention appends the cached ref K/V (RoPE'd here at this pass's text
    length, broadcast to the batch) as extra keys."""
    B, _, H, W = x.shape
    dtype, device = x.dtype, x.device
    transformer_options = transformer_options.copy()
    n_img, text_len = H * W, context.shape[1]

    ids = torch.cat((_image_ids(0, text_len, H, W, device), _text_ids(text_len, device)), dim=0)
    pe = _pe(dit, ids)
    vec = _time_vec(dit, timestep, dtype)
    mod = _modulation(dit, vec)

    txt = dit.text_embedder(context, pe[:, n_img:], transformer_options)
    h = torch.cat((dit.video_embedder(x.flatten(2).transpose(1, 2)), txt), dim=1)

    def live_kv(i):
        entry = ref_kv.get(i)
        if entry is None:
            return None
        k, v = entry
        k = k.to(device, dtype, non_blocking=True)
        v = v.to(device, dtype, non_blocking=True)
        k = rope_split_half(k, ref_pe)
        if k.shape[0] != B:
            if k.shape[0] == 1:
                k, v = k.expand(B, -1, -1, -1), v.expand(B, -1, -1, -1)
            else:
                k = comfy.utils.repeat_to_batch_size(k, B)
                v = comfy.utils.repeat_to_batch_size(v, B)
        return k, v

    def block_fn(block, h, i):
        return _block_kv_forward(block, h, mod, pe, transformer_options, kv_cache=live_kv(i))

    h = _run_blocks(dit, h, block_fn, vec, mod, pe, transformer_options)
    return _finish(dit, h, vec, n_img, B, H, W)


def _ref_fingerprint(ref_latents):
    """Cheap content key for the ref K/V cache: per-ref shape plus two
    reduction sums (the ref tensors are rebuilt per step by comfy's cond
    batching, so object identity can't be used). Batch size and text length
    are deliberately excluded: the cache is independent of both."""
    key = []
    for r in ref_latents:
        rf = r.float()
        key.append((tuple(r.shape), float(rf.sum()), float(rf.square().sum())))
    return tuple(key)


def _divergence(skip_a, skip_b, n_blocks):
    """First block index where two skip-layer sets differ (n_blocks if equal)."""
    diff = set(skip_a) ^ set(skip_b)
    return min(diff) if diff else n_blocks


class MOSSImageOstrisEditModelPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "kv_cache": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Cache the reference tokens' attention K/V: they "
                            "are precomputed in one t=0 pass and reused every "
                            "denoising step, so the refs never ride along in "
                            "the per-step sequence. Matches LoRAs trained with "
                            "ai-toolkit's kv_cache model kwarg (the default "
                            "for its MOSS-Image edit models). Turn off for a "
                            "LoRA trained with kv_cache disabled."
                        ),
                    },
                ),
            },
            "optional": {
                "cache_device": (
                    ["gpu", "cpu"],
                    {
                        "default": "gpu",
                        "tooltip": (
                            "Where the reference K/V cache lives. 'cpu' keeps it "
                            "in pinned system RAM and streams one block at a "
                            "time to the GPU: near-zero extra VRAM, slightly "
                            "slower steps."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "ostris/moss_image"
    DESCRIPTION = (
        "Enable reference latents on a MOSS-Image model (index_timestep_zero "
        "method, as trained by ai-toolkit). Chain conditioning from "
        "TextEncodeMOSSImageOstrisEdit. kv_cache (default on) is the cached "
        "one-pass reference mode ai-toolkit's edit models train with; turn it "
        "off for a LoRA trained with kv_cache disabled."
    )

    def patch(self, model, kv_cache=True, cache_device="gpu"):
        m = model.clone()
        base_model = m.model
        dit = m.get_model_object("diffusion_model")

        orig_extra_conds = base_model.extra_conds
        orig_extra_conds_shapes = base_model.extra_conds_shapes
        orig_forward = dit.forward
        latent_channels = dit.latent_channels
        store_device = torch.device("cpu") if cache_device == "cpu" else None

        def extra_conds(**kwargs):
            out = orig_extra_conds(**kwargs)
            ref_latents = kwargs.get("reference_latents", None)
            if ref_latents is not None:
                out["ref_latents"] = comfy.conds.CONDList(
                    [base_model.process_latent_in(lat) for lat in ref_latents]
                )
            return out

        def extra_conds_shapes(**kwargs):
            out = orig_extra_conds_shapes(**kwargs)
            ref_latents = kwargs.get("reference_latents", None)
            if ref_latents is not None:
                out["ref_latents"] = list(
                    [1, latent_channels, sum(map(lambda a: math.prod(a.size()), ref_latents)) // latent_channels]
                )
            return out

        # caches: {ref_fingerprint: {skipped_blocks: ref_kv}}
        # pes:    {(ref_fingerprint, text_len): ref RoPE}
        state = {"last_sigma": None, "caches": {}, "pes": {}}

        def clear():
            state["caches"].clear()
            state["pes"].clear()
            state["last_sigma"] = None

        def outer_sample_wrapper(executor, *args, **kwargs):
            # Fresh cache per sampling run, and released as soon as the run
            # ends instead of sitting in VRAM until the next one.
            clear()
            try:
                return executor(*args, **kwargs)
            finally:
                clear()

        def forward(x, timestep, context, transformer_options={}, ref_latents=None, **kwargs):
            if ref_latents is None or len(ref_latents) == 0:
                return orig_forward(x, timestep, context, transformer_options=transformer_options, **kwargs)
            if not kv_cache:
                return _forward_with_refs(dit, x, timestep, context, ref_latents, transformer_options)

            # Fallback new-run detection (if the outer-sample wrapper is
            # unavailable): sigmas only decrease within a run.
            sig = float(timestep.max())
            sample_sigmas = transformer_options.get("sample_sigmas", None)
            new_run = state["last_sigma"] is None or sig > state["last_sigma"]
            if sample_sigmas is not None and sig == float(sample_sigmas[0]) and sig != state["last_sigma"]:
                new_run = True
            if new_run:
                state["caches"].clear()
                state["pes"].clear()
            state["last_sigma"] = sig

            text_len = context.shape[1]
            fp = _ref_fingerprint(ref_latents)
            skipped = _skipped_blocks(transformer_options)
            by_skip = state["caches"].setdefault(fp, {})
            ref_kv = by_skip.get(skipped)
            if ref_kv is None:
                reuse, upto = None, 0
                n_blocks = len(dit.layers)
                for other_skip, other_kv in by_skip.items():
                    d = _divergence(skipped, other_skip, n_blocks)
                    if d > upto:
                        reuse, upto = other_kv, d
                ref_kv = _precompute_ref_kv(
                    dit, x, timestep, ref_latents, transformer_options,
                    reuse=reuse, reuse_upto=upto, store_device=store_device,
                )
                by_skip[skipped] = ref_kv

            pe_key = (fp, text_len)
            ref_pe = state["pes"].get(pe_key)
            if ref_pe is None:
                _, refids = _pack_refs(dit, ref_latents, 1, text_len, x.device, x.dtype)
                ref_pe = _pe(dit, refids)
                state["pes"][pe_key] = ref_pe
            return _forward_with_cached_refs(dit, x, timestep, context, ref_kv, ref_pe, transformer_options)

        m.add_object_patch("extra_conds", extra_conds)
        m.add_object_patch("extra_conds_shapes", extra_conds_shapes)
        m.add_object_patch("memory_usage_factor_conds", ("ref_latents",))
        m.add_object_patch("diffusion_model.forward", forward)
        if kv_cache:
            try:
                import comfy.patcher_extension as pe_ext
                m.add_wrapper_with_key(
                    pe_ext.WrappersMP.OUTER_SAMPLE, "moss_ostris_edit_ref_kv", outer_sample_wrapper
                )
            except (ImportError, AttributeError):
                pass  # older ComfyUI: rely on the sigma-based fallback
        return (m,)


NODE_CLASS_MAPPINGS = {
    "TextEncodeMOSSImageOstrisEdit": TextEncodeMOSSImageOstrisEdit,
    "MOSSImageOstrisEditModelPatch": MOSSImageOstrisEditModelPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextEncodeMOSSImageOstrisEdit": "Text Encode MOSS-Image Ostris Edit",
    "MOSSImageOstrisEditModelPatch": "MOSS-Image Ostris Edit Model Patch",
}