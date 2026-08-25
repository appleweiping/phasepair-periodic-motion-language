# PhasePair local CPython 3.12 CPU wheelhouse — public integrity receipt

Date: 2026-08-25 (Asia/Shanghai)  
Status: `LOCAL_PY312_WINDOWS_CPU_WHEELHOUSE_PASS / TEXT_ONLY_PREFLIGHT_PASS_CANDIDATE_ONLY`  
Authority: `0`  
Production: `false`  
Training authorized: `false`

## Result

An exact offline runtime closure was resolved for **CPython 3.12 on Windows
`win_amd64`, CPU only**. It contains 25 wheels totaling **153,244,336 bytes**.
Every archive passed ZIP CRC and path-safety checks; every tag is compatible with
the target; every active dependency is present; and every filename, byte count,
and SHA-256 matches exactly one non-yanked `bdist_wheel` row in the official PyPI
release JSON for that version.

The five selected roots are:

```text
huggingface-hub==0.36.0
numpy==2.4.6
tokenizers==0.22.2
torch==2.10.0
transformers==4.57.3
```

`transformers==4.57.3` was selected because the inspected wheel contains the
contracted `CLIPTextModelWithProjection` and `CLIPTokenizerFast` classes. The
locally observed Transformers 5.5.3 installation does not preserve the exact
frozen fast-tokenizer class name, so using it would require a contract amendment.
No existing Python environment was changed.

## Exact wheel census

| Distribution | Version | Wheel tag(s) | Bytes | SHA-256 | Official release metadata |
|---|---:|---|---:|---|---|
| certifi | 2026.7.22 | `py3-none-any` | 136,983 | `62f22742b58a1a33014a2b6b706588a8d7e2a88ae7bd1a6ebe8c992928483775` | [PyPI JSON](https://pypi.org/pypi/certifi/2026.7.22/json) |
| charset-normalizer | 3.5.1 | `cp312-cp312-win_amd64` | 200,551 | `3617ac3cfd8b9888f145ad89dd6e692285834b0201c6074a5eeaad3fd4d668c2` | [PyPI JSON](https://pypi.org/pypi/charset-normalizer/3.5.1/json) |
| colorama | 0.4.6 | `py2-none-any`, `py3-none-any` | 25,335 | `4f1d9991f5acc0ca119f9d443620b77f9d6b33703e51011c16baf57afb285fc6` | [PyPI JSON](https://pypi.org/pypi/colorama/0.4.6/json) |
| filelock | 3.32.4 | `py3-none-any` | 99,864 | `22e58ca3b1ae3b98993b762d7338367ae64fe50252bf78d59da3bfebcdf1cedd` | [PyPI JSON](https://pypi.org/pypi/filelock/3.32.4/json) |
| fsspec | 2026.7.0 | `py3-none-any` | 206,583 | `b57ddbafedfaef7018c1ecab32aa200a9d7ca26b77965f64e48b70061249d279` | [PyPI JSON](https://pypi.org/pypi/fsspec/2026.7.0/json) |
| huggingface-hub | 0.36.0 | `py3-none-any` | 566,094 | `7bcc9ad17d5b3f07b57c78e79d527102d08313caa278a641993acddcb894548d` | [PyPI JSON](https://pypi.org/pypi/huggingface-hub/0.36.0/json) |
| idna | 3.19 | `py3-none-any` | 68,550 | `815e7be7a7806d54abb586dc943addc79e8b2ee16915059658cbeff4b1b43bf4` | [PyPI JSON](https://pypi.org/pypi/idna/3.19/json) |
| jinja2 | 3.1.6 | `py3-none-any` | 134,899 | `85ece4451f492d0c13c5dd7c13a64681a86afae63a5f347908daf103ce6d2f67` | [PyPI JSON](https://pypi.org/pypi/jinja2/3.1.6/json) |
| markupsafe | 3.0.3 | `cp312-cp312-win_amd64` | 15,105 | `26a5784ded40c9e318cfc2bdb30fe164bdb8665ded9cd64d500a34fb42067b1c` | [PyPI JSON](https://pypi.org/pypi/markupsafe/3.0.3/json) |
| mpmath | 1.3.0 | `py3-none-any` | 536,198 | `a0b2b9fe80bbcd81a6647ff13108738cfb482d481d826cc0e02f5b35e5c88d2c` | [PyPI JSON](https://pypi.org/pypi/mpmath/1.3.0/json) |
| networkx | 3.6.1 | `py3-none-any` | 2,068,504 | `d47fbf302e7d9cbbb9e2555a0d267983d2aa476bac30e90dfbe5669bd57f3762` | [PyPI JSON](https://pypi.org/pypi/networkx/3.6.1/json) |
| numpy | 2.4.6 | `cp312-cp312-win_amd64` | 12,321,687 | `d8e8286dd7cea7895157318d1b91cdacac64c479f3cbc8dce548331728484751` | [PyPI JSON](https://pypi.org/pypi/numpy/2.4.6/json) |
| packaging | 26.3 | `py3-none-any` | 129,956 | `d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c` | [PyPI JSON](https://pypi.org/pypi/packaging/26.3/json) |
| pyyaml | 6.0.3 | `cp312-cp312-win_amd64` | 154,003 | `5fcd34e47f6e0b794d17de1b4ff496c00986e1c83f7ab2fb8fcfe9616ff7477b` | [PyPI JSON](https://pypi.org/pypi/pyyaml/6.0.3/json) |
| regex | 2026.7.19 | `cp312-cp312-win_amd64` | 277,777 | `e30d40268a28d54ce0437031750497004c22602b8e3ab891f759b795a003b312` | [PyPI JSON](https://pypi.org/pypi/regex/2026.7.19/json) |
| requests | 2.34.2 | `py3-none-any` | 73,075 | `2a0d60c172f83ac6ab31e4554906c0f3b3588d37b5cb939b1c061f4907e278e0` | [PyPI JSON](https://pypi.org/pypi/requests/2.34.2/json) |
| safetensors | 0.8.0 | `cp310-abi3-win_amd64` | 355,540 | `096ec1a98435df7beb08853bb5aa9081a84f23d0adc67ed1a0a10550f608373f` | [PyPI JSON](https://pypi.org/pypi/safetensors/0.8.0/json) |
| setuptools | 84.0.0 | `py3-none-any` | 818,216 | `51a52592b3b99e102b609654876bd65f19f999935166d1352678931132b0c670` | [PyPI JSON](https://pypi.org/pypi/setuptools/84.0.0/json) |
| sympy | 1.14.0 | `py3-none-any` | 6,299,353 | `e091cc3e99d2141a0ba2847328f5479b05d94a6635cb96148ccb3f34671bd8f5` | [PyPI JSON](https://pypi.org/pypi/sympy/1.14.0/json) |
| tokenizers | 0.22.2 | `cp39-abi3-win_amd64` | 2,747,786 | `c9ea31edff2968b44a88f97d784c2f16dc0729b8b143ed004699ebca91f05c48` | [PyPI JSON](https://pypi.org/pypi/tokenizers/0.22.2/json) |
| torch | 2.10.0 | `cp312-cp312-win_amd64` | 113,757,972 | `2c66c61f44c5f903046cc696d088e21062644cbe541c7f1c4eaae88b2ad23547` | [PyPI JSON](https://pypi.org/pypi/torch/2.10.0/json) |
| tqdm | 4.70.0 | `py3-none-any` | 80,184 | `7f585706bfddbdebf89daac705b2dfcc16890130727d3197ca62c732b4310953` | [PyPI JSON](https://pypi.org/pypi/tqdm/4.70.0/json) |
| transformers | 4.57.3 | `py3-none-any` | 11,993,463 | `c77d353a4851b1880191603d36acb313411d3577f6e2897814f333841f7003f4` | [PyPI JSON](https://pypi.org/pypi/transformers/4.57.3/json) |
| typing-extensions | 4.16.0 | `py3-none-any` | 45,571 | `481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8` | [PyPI JSON](https://pypi.org/pypi/typing-extensions/4.16.0/json) |
| urllib3 | 2.7.0 | `py3-none-any` | 131,087 | `9fb4c81ebbb1ce9531cce37674bbc6f1360472bc18ca9a553ede278ef7276897` | [PyPI JSON](https://pypi.org/pypi/urllib3/2.7.0/json) |

The closure contains 32 active dependency edges for the exact target marker
environment. There are zero missing active dependencies and zero unreferenced
non-root distributions. The two frozen input identities are:

| Input | Bytes | SHA-256 |
|---|---:|---|
| Five-root requirements | 91 | `82dd3cdda236445890cb9dbef5acb2ba9a9893b2d5aef75f37ab3e19352d953f` |
| Complete `--require-hashes` lock | 2,404 | `e72315fc4f998273646553afe8f22574d81a8fca2472ebb1022fc571c87bd852` |

The isolated verification install used only the private wheel directory with
`--no-index --require-hashes`; installation completed and `pip check` reported
no broken requirements.

## Text-only offline preflight

The exact private snapshot of `openai/clip-vit-base-patch32` revision
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` was loaded with:

```text
CLIPTextModelWithProjection
CLIPTokenizerFast
local_files_only=True
trust_remote_code=False
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

The first candidate-script attempt loaded the checkpoint but stopped before the
forward because the internal `CLIPTextTransformer.forward` signature does not
accept a caller-supplied `return_dict` argument. The argument was removed to
match the observed 4.57.3 signature. Two subsequent isolated runs both passed
and reproduced the same golden hashes.

Observed facts:

- all 8 frozen snapshot files were opened; full scans before and after each load
  still matched 608,863,580 bytes and the frozen per-file SHA-256 values;
- 0 network calls were attempted under both offline flags and explicit request /
  connection blockers;
- exact loader/tokenizer types matched; tokenizer width was 77;
- 0 vision modules, 0 vision parameters, and no `vision_model` attribute were
  instantiated;
- the full checkpoint produced 202 expected unused state keys, all limited to
  `vision_model.*`, `visual_projection.*`, or `logit_scale`; there were 0 missing,
  0 mismatched, and 0 error keys;
- the retained text registry has 197 state keys and 63,165,952 trainable text
  parameters; the pretrained text projection has 262,144 parameters and was
  immediately frozen;
- the train-mode forward made 12 scaled-dot-product attention calls, each with
  dropout probability exactly 0; active dropout calls/modules were 0 and the
  RNG state SHA-256 was unchanged;
- output shapes were pooled EOS `[3,512]`, pretrained projection `[3,512]`, and
  float64 L2-normalized `[3,512]`; all values were finite and each final vector
  had unit L2 norm within absolute tolerance `1e-12`.

Candidate golden identities (three fixed ASCII captions, each encoded with a
trailing LF, in order):

| Evidence | SHA-256 |
|---|---|
| Caption bytes | `dd2202c7fd1c1e7e49ecd1f77060e526f6c12d6eb9e991d83460273c541784bb` |
| Input IDs (`int64`, `[3,77]`) | `2c1bc6af4fe563c4b2dee0bc78bcd8bf2868ad3bd8f74d4126329f43a280cc8a` |
| Attention mask (`int64`, `[3,77]`) | `764c9a7807146b3dd2c20deae73cffe92db80069861a45d3ec3f0863c6b8f34e` |
| Final-LN pooled EOS (`float32`, `[3,512]`) | `07acbb2f6d1c803f12dfd937913d4f2165ca0ed08e248baef2a725a6e3c99a1a` |
| Pretrained projection (`float32`, `[3,512]`) | `a4ec38be61e2db328652bd1119d4e8aba3f218a594562d0eb0060f07cf6e0f61` |
| L2-normalized projection (`float64`, `[3,512]`) | `26df05351d13d0e66048bb6f63c3fc84b62aced5d4c765cc6b18d4606201ee97` |

Private candidate artifacts are bound without publishing their locators:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Preflight script | 17,016 | `f2bd44127602d383fc3345a0380ba91dae9734a7057c212c284b83abe8ec2db2` |
| Candidate preflight receipt | 5,588 | `673fed24f9dc6da6820315b1d2c1b39085d66225ac26f0c64973e5905211737d` |
| Exact private wheelhouse receipt | 17,322 | `1618cab3f231409e4b47270da8999a81b2887a517699b592e5ca14c6537ad088` |

This is a **candidate local qualification**, not the fresh-reviewed one-shot
production runtime lease required by the live bridge contract. No optimizer was
constructed and no training was authorized.

## Rights and scope boundary

Package-integrity metadata is not a substitute for a license review, and a
package license cannot grant rights in separately hosted model weight bytes.
The CLIP rights verdict therefore remains exactly the conservative verdict in
[the public CLIP acquisition receipt](./PHASEPAIR_CLIP_RIGHTS_RESOLUTION_PUBLIC_20260825_004016.md):

- private, non-deployed research: conditional `GO`;
- deployment: `NO_GO`;
- CLIP weight redistribution/mirroring: `NO_GO`;
- no model or wheel archive is present in the public repository.

This receipt does not authorize production, deployment, redistribution,
commercial use, or training. It is an engineering and operational record, not
legal advice.

## Remaining blocker

This Windows CPU wheelhouse must not be projected onto the server. A separate
CUDA runtime can be frozen only after the actual remote GPU model, driver,
operating system, Python ABI, and compatible CUDA requirement are observed.
Until then the server CUDA wheelhouse status is
`DEFERRED_FAIL_CLOSED_NO_GUESSING`.
