

from transformers.configuration_utils import PretrainedConfig

class MiniQwenConfig(PretrainedConfig):
    """
    MiniQwen configuration class.
    """

    model_type = "mini_qwen"

    def __init__(
        self,
        vocab_size=30522,  # 基于BERT词表大小
        hidden_size=768,   # 隐藏层维度
        intermediate_size=2048,  # FFN中间层维度
        num_hidden_layers=4,  # Transformer层数
        num_attention_heads=12,  # 注意力头数量
        num_key_value_heads=4,  # GQA中的KV头数量
        hidden_act="swiglu",  # 激活函数
        max_position_embeddings=1024,  # 最大位置编码
        initializer_range=0.02,  # 初始化范围
        rms_norm_eps=1e-6,  # RMSNorm epsilon
        use_cache=True,  # 是否使用KV缓存
        pad_token_id=0,  # PAD token ID
        bos_token_id=1,  # BOS token ID
        eos_token_id=2,  # EOS token ID
        tie_word_embeddings=False,  # 是否绑定输入输出嵌入
        rope_theta=10000.0,  # RoPE旋转参数
        rope_scaling=None,  # RoPE缩放配置
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )