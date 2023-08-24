import clip
from clip.validation import ImageNetValidator, CosineSimValidator


model, preprocess = clip.load("ViT-B/32")
# validator = ImageNetValidator(model, preprocess, "cpu", None)
# validator.validate()


cosineValidator = CosineSimValidator(model, "cpu", None)
cosineValidator.validate(0)