import os
import time
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import PreTrainedModel

from fireredasr.hf.configuration_fireredasr import FireRedASRConfig

# Prefer library imports; fall back to local packaged code when running with trust_remote_code
try:
    from fireredasr.models.fireredasr_aed import FireRedAsrAed
    from fireredasr.models.fireredasr_llm import FireRedAsrLlm
    from fireredasr.data.asr_feat import ASRFeatExtractor
    from fireredasr.tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer
    from fireredasr.tokenizer.llm_tokenizer import LlmTokenizerWrapper
except Exception:
    from models.fireredasr_aed import FireRedAsrAed
    from models.fireredasr_llm import FireRedAsrLlm
    from data.asr_feat import ASRFeatExtractor
    from tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer
    from tokenizer.llm_tokenizer import LlmTokenizerWrapper


def _load_fireredasr_aed_model(model_path: str):
    package = torch.load(model_path, map_location=lambda storage, loc: storage)
    model = FireRedAsrAed.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=True)
    return model


def _load_firered_llm_model_and_tokenizer(model_path: str, encoder_path: str, llm_dir: str):
    package = torch.load(model_path, map_location=lambda storage, loc: storage)
    package["args"].encoder_path = encoder_path
    package["args"].llm_dir = llm_dir
    model = FireRedAsrLlm.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=False)
    tokenizer = LlmTokenizerWrapper.build_llm_tokenizer(llm_dir)
    return model, tokenizer


@dataclass
class SpeechToTextOutput:
    sequences: List[str]
    rtf: Optional[float] = None


class FireRedASRForSpeechToText(PreTrainedModel):
    config_class = FireRedASRConfig

    def __init__(self, config: FireRedASRConfig):
        super().__init__(config)
        self.asr_type = config.asr_type

        # Placeholders; real modules get loaded in from_pretrained
        self.model = None  # torch.nn.Module (AED or LLM wrapper)
        self.featurizer: Optional[ASRFeatExtractor] = None
        self.tokenizer = None
        self._model_dir: Optional[str] = None  # source dir for packaging

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, *model_args, **kwargs
    ):
        """Custom loader.

        Expects a directory containing FireRedASR checkpoints.
        For AED: requires `model.pth.tar`, `cmvn.ark`, `dict.txt`, `train_bpe1000.model`.
        For LLM: requires `model.pth.tar`, `asr_encoder.pth.tar`, `cmvn.ark`, and an `llm` dir.
        Optionally accepts `asr_type` in `kwargs` to override config.
        """
        # Load config if present; otherwise use defaults.
        config = kwargs.pop("config", None)
        if config is None:
            try:
                # Use HF-style config.json if available
                config = FireRedASRConfig.from_pretrained(pretrained_model_name_or_path)
            except Exception:
                config = FireRedASRConfig()

        # Allow override of asr_type
        asr_type = kwargs.pop("asr_type", None) or config.asr_type

        # Instantiate wrapper without expecting HF weights in the folder
        model = cls(config)

        # Build featurizer
        cmvn_path = config.cmvn_path or os.path.join(pretrained_model_name_or_path, "cmvn.ark")
        model.featurizer = ASRFeatExtractor(cmvn_path)

        # Device selection allows caller override via kwargs (e.g., CLI flag)
        use_gpu = kwargs.pop("use_gpu", None)
        if use_gpu is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")

        # Build internal model + tokenizer
        if asr_type == "aed":
            model_path = os.path.join(pretrained_model_name_or_path, "model.pth.tar")
            dict_path = os.path.join(pretrained_model_name_or_path, "dict.txt")
            spm_model = os.path.join(pretrained_model_name_or_path, "train_bpe1000.model")
            model.model = _load_fireredasr_aed_model(model_path).to(device).eval()
            model.tokenizer = ChineseCharEnglishSpmTokenizer(
                dict_path, spm_model
            )
        elif asr_type == "llm":
            model_path = os.path.join(pretrained_model_name_or_path, "model.pth.tar")
            encoder_path = os.path.join(pretrained_model_name_or_path, "asr_encoder.pth.tar")
            llm_dir = config.llm_dir or os.path.join(pretrained_model_name_or_path, "Qwen2-7B-Instruct")
            model.model, model.tokenizer = _load_firered_llm_model_and_tokenizer(
                model_path, encoder_path, llm_dir
            )
            model.model = model.model.to(device).eval()
        else:
            raise ValueError(f"Unsupported asr_type: {asr_type}")

        model.asr_type = asr_type
        model._model_dir = pretrained_model_name_or_path
        return model

    @torch.no_grad()
    def generate(
        self,
        wav_paths: List[str],
        beam_size: int = 1,
        max_new_tokens: int = 0,
        nbest: int = 1,
        softmax_smoothing: float = 1.0,
        repetition_penalty: float = 1.0,
        length_penalty: float = 0.0,
        eos_penalty: float = 1.0,
        temperature: float = 1.0,
        return_timestamps: bool = False,  # unused placeholder
    ) -> SpeechToTextOutput:
        """Transcribe a batch of wav files.

        Parameters mirror internal model parameters and will be routed based on asr_type.
        """
        assert self.model is not None, "Call from_pretrained before generate()"
        feats, lengths, durs = self.featurizer(wav_paths)
        total_dur = float(sum(durs)) if len(durs) else 0.0
        device = next(self.model.parameters()).device
        feats, lengths = feats.to(device), lengths.to(device)

        start_time = time.time()
        if self.asr_type == "aed":
            hyps = self.model.transcribe(
                feats,
                lengths,
                beam_size=beam_size,
                nbest=nbest,
                decode_max_len=max_new_tokens,
                softmax_smoothing=softmax_smoothing,
                length_penalty=length_penalty,
                eos_penalty=eos_penalty,
            )
            texts =[]
            for hyp in hyps:
                hyp = hyp[0]
                hyp_ids = [int(i) for i in hyp["yseq"].cpu()]
                text = self.tokenizer.detokenize(hyp_ids)
                texts.append(text)
        else:  # llm
            input_ids, attention_mask, _, _ = LlmTokenizerWrapper.preprocess_texts(
                origin_texts=[""] * feats.size(0),
                tokenizer=self.tokenizer,
                max_len=128,
                decode=True,
            )
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            generated_ids = self.model.transcribe(
                feats,
                lengths,
                input_ids,
                attention_mask,
                beam_size=beam_size,
                decode_max_len=max_new_tokens,
                decode_min_len=0,
                repetition_penalty=repetition_penalty,
                llm_length_penalty=length_penalty,
                temperature=temperature,
            )
            texts = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

        elapsed = time.time() - start_time
        rtf = 0.0 if total_dur == 0.0 else elapsed / total_dur
        return SpeechToTextOutput(sequences=texts, rtf=rtf)

    def save_pretrained(self, save_directory: str, **kwargs):
        """Export a folder that AutoModel can load in a fresh env.

        This writes a `config.json` and copies required artifact files from either
        - the original `_model_dir` used to load the model, or
        - explicit paths passed via kwargs.

        For AED, copies: model.pth.tar, cmvn.ark, dict.txt, train_bpe1000.model
        For LLM, copies: model.pth.tar, asr_encoder.pth.tar, cmvn.ark
          - LLM weights are referenced by `config.llm_dir` (set to a repo id or local dir).
        """
        import shutil
        os.makedirs(save_directory, exist_ok=True)

        # Add auto_map and architectures for trust_remote_code loading
        self.config.auto_map = {
            "AutoConfig": "configuration_fireredasr.FireRedASRConfig",
            "AutoModel": "modeling_fireredasr.FireRedASRForSpeechToText",
        }
        self.config.architectures = ["FireRedASRForSpeechToText"]
        # Save config.json
        self.config.save_pretrained(save_directory)

        # Resolve source directory
        src_dir = kwargs.get("src_dir", self._model_dir)
        if src_dir is None:
            raise ValueError("Please provide `src_dir` or load the model via from_pretrained first.")

        def _copy(name: str):
            src = kwargs.get(name) or os.path.join(src_dir, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(save_directory, os.path.basename(name)))
            else:
                raise FileNotFoundError(f"Missing artifact: {src}")

        # Common
        _copy("model.pth.tar")
        _copy("cmvn.ark")

        if self.asr_type == "aed":
            _copy("dict.txt")
            _copy("train_bpe1000.model")
        elif self.asr_type == "llm":
            _copy("asr_encoder.pth.tar")
            # LLM weights directory: either copy local dir or set repo id
            llm_repo_id = kwargs.get("llm_repo_id")
            copy_llm_dir = kwargs.get("copy_llm_dir", False)
            # Resolve source LLM dir
            src_llm_dir = (
                kwargs.get("llm_dir")
                or self.config.llm_dir
                or os.path.join(src_dir, "Qwen2-7B-Instruct")
            )
            if copy_llm_dir:
                if src_llm_dir and os.path.exists(src_llm_dir):
                    dst_llm_dir = os.path.join(
                        save_directory, os.path.basename(src_llm_dir)
                    )
                    shutil.copytree(src_llm_dir, dst_llm_dir, dirs_exist_ok=True)
                    self.config.llm_dir = dst_llm_dir
                else:
                    raise FileNotFoundError(
                        f"LLM dir not found for copying: {src_llm_dir}"
                    )
            elif llm_repo_id:
                # Set to remote repo id so a fresh env can fetch
                self.config.llm_dir = llm_repo_id
            # Persist updated config
            self.config.save_pretrained(save_directory)

        # Optionally include minimal code for trust_remote_code usage
        include_code = kwargs.get("include_code", False)
        if include_code:
            import inspect
            import shutil

            def _copy_code(src_rel: str, dst_rel: str):
                # Map importable module path for inspect
                module_path = src_rel.replace("/", ".").rstrip(".py")
                src_path = None
                try:
                    mod = __import__(module_path, fromlist=[None])
                    src_path = inspect.getsourcefile(mod)
                except Exception:
                    # Fallback to repo path
                    candidate = os.path.join(src_dir, src_rel)
                    if os.path.exists(candidate):
                        src_path = candidate
                if not src_path or not os.path.exists(src_path):
                    raise FileNotFoundError(f"Code file not found: {src_rel}")
                dst_path = os.path.join(save_directory, dst_rel)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)

            # Core HF code
            _copy_code("fireredasr/hf/configuration_fireredasr.py", "configuration_fireredasr.py")
            _copy_code("fireredasr/hf/modeling_fireredasr.py", "modeling_fireredasr.py")
            # Runtime dependencies expected by modeling
            _copy_code("fireredasr/data/asr_feat.py", "data/asr_feat.py")
            _copy_code("fireredasr/tokenizer/aed_tokenizer.py", "tokenizer/aed_tokenizer.py")
            _copy_code("fireredasr/tokenizer/llm_tokenizer.py", "tokenizer/llm_tokenizer.py")
            _copy_code("fireredasr/models/fireredasr_aed.py", "models/fireredasr_aed.py")
            _copy_code("fireredasr/models/fireredasr_llm.py", "models/fireredasr_llm.py")
            _copy_code("fireredasr/models/module/adapter.py", "models/module/adapter.py")
            _copy_code("fireredasr/models/module/conformer_encoder.py", "models/module/conformer_encoder.py")
            _copy_code("fireredasr/models/module/transformer_decoder.py", "models/module/transformer_decoder.py")
            _copy_code("fireredasr/utils/param.py", "utils/param.py")
            # Add README_CODE note
            with open(os.path.join(save_directory, "README_CODE.md"), "w", encoding="utf-8") as f:
                f.write(
                    "This package includes minimal runtime code for Transformers loading with trust_remote_code=True.\n"
                )
        else:
            raise ValueError(f"Unsupported asr_type: {self.asr_type}")
