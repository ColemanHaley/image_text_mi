from transformers import PaliGemmaForConditionalGeneration, PaliGemmaConfig

class PaliGemmaCached(PaliGemmaForConditionalGeneration):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__(config)
        self.past_key_values = None

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        token_type_ids = None,
        cache_position = None,
        input_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        super().forward(input_ids, attention_mask, position_ids, token_type_ids, cache_position, self.past_key_values, input_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict)

    def cache_prefix(self, prefix):
        self.past_key_values = prefix



