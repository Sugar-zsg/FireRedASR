from transformers import PretrainedConfig


class FireRedASRConfig(PretrainedConfig):
    """Minimal config to wrap FireRedASR with Hugging Face.

    Supports two modes:
    - "aed": Conformer encoder + Transformer decoder (beam search inference)
    - "llm": Conformer encoder + LLM (generate)

    Only inference settings are included here. Training is out of scope.
    """

    model_type = "fireredasr"

    def __init__(
        self,
        asr_type: str = "aed",  # one of {"aed", "llm"}
        # Common
        idim: int = 80,
        d_model: int = 512,
        n_head: int = 8,
        residual_dropout: float = 0.1,
        dropout_rate: float = 0.1,
        kernel_size: int = 33,
        pe_maxlen: int = 5000,
        sos_id: int = 1,
        eos_id: int = 2,
        pad_id: int = 0,
        odim: int = 1000,  # vocabulary size for AED
        # Encoder/Decoder depth
        n_layers_enc: int = 12,
        n_layers_dec: int = 6,
        # LLM settings
        freeze_encoder: bool = True,
        freeze_llm: bool = True,
        encoder_downsample_rate: int = 4,
        use_flash_attn: bool = False,
        use_fp16: bool = False,
        use_lora: bool = False,
        # Paths (used by custom from_pretrained)
        encoder_path: str | None = None,
        llm_dir: str | None = None,
        cmvn_path: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.asr_type = asr_type

        # Core audio/transformer
        self.idim = idim
        self.d_model = d_model
        self.n_head = n_head
        self.residual_dropout = residual_dropout
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.pe_maxlen = pe_maxlen

        # Token settings for AED
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.odim = odim

        # Depth
        self.n_layers_enc = n_layers_enc
        self.n_layers_dec = n_layers_dec

        # LLM + adapter
        self.freeze_encoder = freeze_encoder
        self.freeze_llm = freeze_llm
        self.encoder_downsample_rate = encoder_downsample_rate
        self.use_flash_attn = use_flash_attn
        self.use_fp16 = use_fp16
        self.use_lora = use_lora

        # Optional paths (used by our custom loader)
        self.encoder_path = encoder_path
        self.llm_dir = llm_dir
        self.cmvn_path = cmvn_path

