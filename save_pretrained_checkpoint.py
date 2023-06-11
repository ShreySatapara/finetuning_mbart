from transformers import MBartForConditionalGeneration, MBartConfig
pretrained_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
config = MBartConfig.from_pretrained("facebook/mbart-large-50-many-to-one-mmt",encoder_layers=6, decoder_layers=6)
model = MBartForConditionalGeneration(config)

model.model.shared.load_state_dict(pretrained_model.model.shared.state_dict())

model.model.encoder.embed_tokens.load_state_dict(pretrained_model.model.encoder.embed_tokens.state_dict())
model.model.encoder.embed_positions.load_state_dict(pretrained_model.model.encoder.embed_positions.state_dict())
model.model.encoder.layernorm_embedding.load_state_dict(pretrained_model.model.encoder.layernorm_embedding.state_dict())
model.model.encoder.layer_norm.load_state_dict(pretrained_model.model.encoder.layer_norm.state_dict())

model.model.decoder.embed_tokens.load_state_dict(pretrained_model.model.decoder.embed_tokens.state_dict())
model.model.decoder.embed_positions.load_state_dict(pretrained_model.model.decoder.embed_positions.state_dict())
model.model.decoder.layernorm_embedding.load_state_dict(pretrained_model.model.decoder.layernorm_embedding.state_dict())
model.model.decoder.layer_norm.load_state_dict(pretrained_model.model.decoder.layer_norm.state_dict())

for i in range(6):
    model.model.encoder.layers[i].load_state_dict(pretrained_model.model.encoder.layers[i].state_dict())
    model.model.decoder.layers[i].load_state_dict(pretrained_model.model.decoder.layers[i].state_dict())
model.lm_head.load_state_dict(pretrained_model.lm_head.state_dict())

model.save_pretrained("../mbart50_pretrained_loaded_6_en_6_de")
#model.save_state_dict("mbart59_pretrained_loaded_6_en_6_de_state_dict.pt")