import os
from accelerate import Accelerator
from clip.validation import ImageNetValidator, CosineSimValidator
from clip import clip
from clip.model import CLIP

class TestModel():
    def __init__(self):

        print("Prepare the model")
        model = CLIP(embed_dim=512, image_resolution=224, vision_layers=12, vision_width=768, vision_patch_size=32,
                 transformer_layers=12, transformer_width=512, transformer_heads=8, vocab_size=49408, context_length=77)

        self.preprocess = clip._transform(model.visual.input_resolution)

        # Open the accelerate model
        self.accelerator = Accelerator()
        self.model = self.accelerator.prepare(model)

        # Load the model
        print("Load the model")
        self.load_model()

        # Validate the model
        print("Prepare the validators")
        self.imageNetValidator = ImageNetValidator(self, self.preprocess, self.accelerator.device, None)
        self.cosineValidator = CosineSimValidator(self, self.accelerator.device, None)

        print("Validate the model")
        self.imageNetValidator.validate(0, True)
        self.cosineValidator.validate(0, True)

        

    def load_model(self):
        self.accelerator.load_state(os.path.join("outputs", "checkpoints_step700"))
        self.accelerator.wait_for_everyone()



if __name__ == '__main__':

    testModel = TestModel()
