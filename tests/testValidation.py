from training import clip
from training.clip.validation import ImageNetValidator, CosineSimValidator, MNISTValidator, SST2Validator
from training.training import Trainer, parse_args

model, preprocess = clip.load("ViT-B/32")

args = parse_args()
trainer = Trainer(model, preprocess, 0, args)

validator = MNISTValidator(trainer, preprocess, "cpu", None)
validator.validate(0, True)

