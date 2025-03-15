from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
import torch
import torch.nn as nn

class MyPaliGemmaForConditionalGeneration(PaliGemmaForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        
        # Example: Add a new layer or modify an existing one
        self.new_layer = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size)

    def forward(self, *args, **kwargs):
        # Call the original forward method
        outputs = super().forward(*args, **kwargs)
        
        # Example: Apply a transformation to the logits
        modified_logits = self.new_layer(outputs.logits)
        
        # Return modified output
        return modified_logits
