import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import clip
from imagenetv2_pytorch import ImageNetV2Dataset

from scipy.stats import spearmanr, pearsonr
import scipy.stats as stats

import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from clip.dataset import STS, SST, MNIST

class ImageNetValidator():
    def __init__(self, trainer, preprocess, device, writer) -> None:

        self.trainer = trainer
        self.device = device
        self.writer = writer

        self.imagenet_classes = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper", "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra", "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake", "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider", "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl", "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog", "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf", "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly", "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse", "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish", "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)", "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra", "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton", "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw", "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker", "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard", "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed", "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed", "freight car", "French horn", "frying pan", "fur coat", "garbage truck", "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine", "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag", "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor", "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa", "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail", "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag", "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho", "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug", "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard", "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store", "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater", "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge", "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling", "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink", "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle", "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing", "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website", "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate", "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef", "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player", "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom", "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]


        self.imagenet_templates = [
            'a bad photo of a {}.',
            'a photo of many {}.',
            'a sculpture of a {}.',
            'a photo of the hard to see {}.',
            'a low resolution photo of the {}.',
            'a rendering of a {}.',
            'graffiti of a {}.',
            'a bad photo of the {}.',
            'a cropped photo of the {}.',
            'a tattoo of a {}.',
            'the embroidered {}.',
            'a photo of a hard to see {}.',
            'a bright photo of a {}.',
            'a photo of a clean {}.',
            'a photo of a dirty {}.',
            'a dark photo of the {}.',
            'a drawing of a {}.',
            'a photo of my {}.',
            'the plastic {}.',
            'a photo of the cool {}.',
            'a close-up photo of a {}.',
            'a black and white photo of the {}.',
            'a painting of the {}.',
            'a painting of a {}.',
            'a pixelated photo of the {}.',
            'a sculpture of the {}.',
            'a bright photo of the {}.',
            'a cropped photo of a {}.',
            'a plastic {}.',
            'a photo of the dirty {}.',
            'a jpeg corrupted photo of a {}.',
            'a blurry photo of the {}.',
            'a photo of the {}.',
            'a good photo of the {}.',
            'a rendering of the {}.',
            'a {} in a video game.',
            'a photo of one {}.',
            'a doodle of a {}.',
            'a close-up photo of the {}.',
            'a photo of a {}.',
            'the origami {}.',
            'the {} in a video game.',
            'a sketch of a {}.',
            'a doodle of the {}.',
            'a origami {}.',
            'a low resolution photo of a {}.',
            'the toy {}.',
            'a rendition of the {}.',
            'a photo of the clean {}.',
            'a photo of a large {}.',
            'a rendition of a {}.',
            'a photo of a nice {}.',
            'a photo of a weird {}.',
            'a blurry photo of a {}.',
            'a cartoon {}.',
            'art of a {}.',
            'a sketch of the {}.',
            'a embroidered {}.',
            'a pixelated photo of a {}.',
            'itap of the {}.',
            'a jpeg corrupted photo of the {}.',
            'a good photo of a {}.',
            'a plushie {}.',
            'a photo of the nice {}.',
            'a photo of the small {}.',
            'a photo of the weird {}.',
            'the cartoon {}.',
            'art of the {}.',
            'a drawing of the {}.',
            'a photo of the large {}.',
            'a black and white photo of a {}.',
            'the plushie {}.',
            'a dark photo of a {}.',
            'itap of a {}.',
            'graffiti of the {}.',
            'a toy {}.',
            'itap of my {}.',
            'a photo of a cool {}.',
            'a photo of a small {}.',
            'a tattoo of the {}.',
        ]

        print(f"{len(self.imagenet_classes)} classes, {len(self.imagenet_templates)} templates")

        os.makedirs("datasetImageNet", exist_ok=True)
        images = ImageNetV2Dataset(transform=preprocess, location="datasetImageNet")
        self.loader = DataLoader(images, batch_size=32, num_workers=0)


    def zeroshot_classifier(self, classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames, miniters=20, mininterval=50, desc="Computing ImageNet classes weights"):
                texts = [template.format(classname) for template in templates] #format with class
                texts = clip.tokenize(texts).to(self.device) #tokenize
                if isinstance(self.trainer.model, nn.parallel.DistributedDataParallel):
                    class_embeddings = self.trainer.model.module.encode_text(texts)
                else:
                    class_embeddings = self.trainer.model.encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(self.device)
        return zeroshot_weights
    
    def accuracy(self, output, target, topk=(1,)):
        pred = output.topk(max(topk), 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
    

    def validate(self, step, verbose=False):
        if isinstance(self.trainer.model, nn.parallel.DistributedDataParallel):
            self.trainer.model.module.eval()
        else:
            self.trainer.model.eval()

        self.zeroshot_weights = self.zeroshot_classifier(self.imagenet_classes, self.imagenet_templates)

        with torch.no_grad():
            top1, top5, n = 0., 0., 0.
            for i, (images, target) in enumerate(tqdm(self.loader, miniters=20, mininterval=50, desc="Image net validation")):
                images = images.to(self.device)
                target = target.to(self.device)
                
                # predict
                if isinstance(self.trainer.model, nn.parallel.DistributedDataParallel):
                    image_features = self.trainer.model.module.encode_image(images)
                else:
                    image_features = self.trainer.model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = 100. * image_features @ self.zeroshot_weights

                # measure accuracy
                acc1, acc5 = self.accuracy(logits, target, topk=(1, 5))
                top1 += acc1
                top5 += acc5
                n += images.size(0)

        top1 = (top1 / n) * 100
        top5 = (top5 / n) * 100

        if verbose:
            print(f"Top-1 accuracy: {top1:.2f}%")
            print(f"Top-5 accuracy: {top5:.2f}%")

        if self.writer is not None:
            self.writer.add_scalar("Top-1 accuracy", top1, step)
            self.writer.add_scalar("Top-5 accuracy", top5, step)




class CosineSimValidator():
    def __init__(self, trainer, device, writer) -> None:

        self.trainer = trainer
        self.device = device
        self.writer = writer

        self.datasets = [STS(selectedSet=set) for set in ["sick", "mteb/sts16-sts", "mteb/sts15-sts", "mteb/sts14-sts", "mteb/sts13-sts", "mteb/sts12-sts"]] #

    def validate(self, step, verbose=False):
        for dataset in self.datasets:
            self._validateDataset(dataset, step, verbose=verbose)



    def _validateDataset(self, dataset, step, verbose=False):
        self.trainer.model.eval()

        linfSimilarities = []
        l2Similarities = []
        cosineSimilarities = []
        truth = []

        for batch in DataLoader(dataset, batch_size=32):
            text1, text2, label = batch
            text1 = clip.tokenize(text1, truncate=True).to(self.device)
            text2 = clip.tokenize(text2, truncate=True).to(self.device)

            with torch.no_grad():
                if isinstance(self.trainer.model, nn.parallel.DistributedDataParallel):
                    text_features_1 = self.trainer.model.module.encode_text(text1)
                    text_features_2 = self.trainer.model.module.encode_text(text2)
                else:
                    text_features_1 = self.trainer.model.encode_text(text1)
                    text_features_2 = self.trainer.model.encode_text(text2)
          
                linfSim = torch.max(torch.abs(text_features_1 - text_features_2), dim=1)[0]
                cosineSim = F.cosine_similarity(text_features_1, text_features_2)
                l2Sim = torch.norm(text_features_1 - text_features_2, 2, dim=1)

            linfSimilarities.append(linfSim)
            l2Similarities.append(l2Sim)
            cosineSimilarities.append(cosineSim)
            truth.append(label)
        
        linfSimilarities = torch.concat(linfSimilarities).cpu().numpy()
        l2Similarities = torch.concat(l2Similarities).cpu().numpy()
        cosineSimilarities = torch.concat(cosineSimilarities).cpu().numpy()
        truth = torch.concat(truth).cpu().numpy()


        srcclinf = spearmanr(linfSimilarities, truth)
        srccCosine = spearmanr(cosineSimilarities, truth)
        pcCosine = pearsonr(cosineSimilarities, truth)

        if self.writer:
            if step is not None:
                self.writer.add_scalar(f"{dataset.datasetName}/SRCC Linf", srcclinf.correlation, global_step=step)
                self.writer.add_scalar(f"{dataset.datasetName}/SRCC Cosine", srccCosine.correlation, global_step=step)
                self.writer.add_scalar(f"{dataset.datasetName}/PC Cosine", pcCosine.statistic, global_step=step)
            else:
                self.writer.add_text(f"{dataset.datasetName}/SRCC Linf", str(srcclinf.correlation))
                self.writer.add_text(f"{dataset.datasetName}/SRCC Cosine", str(srccCosine.correlation))
                self.writer.add_text(f"{dataset.datasetName}/SRCC Cosine", str(pcCosine.statistic))

        if verbose:
            print(f"Spearmen Ranking correlation coefficient Linf {srcclinf.correlation:.3f}")
            print(f"Spearmen Ranking correlation coefficient Cosine {srccCosine.correlation:.3f}")
            print(f"Pearson correlation coefficient Cosine {pcCosine.statistic:.3f}")

        # Plot the distance with a noise in y, the distance in x and a different color if they are a duplicate
        cdict = {0: "red", 1: "green", 2: "blue", 3: "orange", 4: "purple"}
        legend = ["0 - 1", "1 - 2", "2 - 3", "3 - 4", "4 - 5"]

        fig, ax = plt.subplots()
        for g in range(len(cdict)):
            ix = np.where((g <= truth) & (truth < g + 1))

            if ix == []:
                continue
            n, x, _ = ax.hist(
                l2Similarities[ix], bins=np.linspace(0, max(l2Similarities), 100), histtype="step", density=True, alpha=0.5, color=cdict[g]
            )
            if len(np.unique(l2Similarities[ix])) > 1:
                density = stats.gaussian_kde(l2Similarities[ix])
                ax.plot(x, density(x), c=cdict[g], label=legend[g])

        ax.legend()
        ax.get_yaxis().set_visible(False)
        ax.set_xlabel("L2 Distance between pair")
        ax.set_title("Distances for duplicate and non-duplicate pairs")

        if self.writer and step is not None: self.writer.add_figure(f"{dataset.datasetName}/neighborContinuousHistogramL2", fig, step)
        if verbose: plt.savefig("neighborContinuousHistogramL2.png")

        fig, ax = plt.subplots()
        for g in range(len(cdict)):
            ix = np.where((g <= truth) & (truth < g + 1))

            if ix == []:
                continue
            n, x, _ = ax.hist(
                cosineSimilarities[ix],
                bins=np.linspace(-1, 1, 100),
                histtype="step",
                density=True,
                alpha=0.5,
                color=cdict[g],
            )
            if len(np.unique(cosineSimilarities[ix])) > 1:
                density = stats.gaussian_kde(cosineSimilarities[ix])
                ax.plot(x, density(x), c=cdict[g], label=legend[g])

        ax.legend()
        ax.get_yaxis().set_visible(False)
        ax.set_xlabel("Cosine Distance between pair")
        ax.set_title("Distances for duplicate and non-duplicate pairs")

        if self.writer and step is not None: self.writer.add_figure(f"{dataset.datasetName}/neighborContinuousHistogramCosine", fig, step)
        if verbose: plt.savefig("neighborContinuousHistogramCosine.png")


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class SST2Validator():
    def __init__(self, trainer, device, writer) -> None:

        self.trainer = trainer
        self.device = device
        self.writer = writer

        self.datasetTrain = SST("train")
        self.datasetTest = SST("validation")
    
    def validate(self, step, verbose=False):
        # Compute the embeddings for all samples in the dataset
        embeddings = []
        labels = []
        self.trainer.model.eval()
        for sample, label in DataLoader(self.datasetTrain, batch_size=32):
            sample = clip.tokenize(sample, truncate=True).to(self.device)
            with torch.no_grad():
                if isinstance(self.trainer.model, nn.parallel.DistributedDataParallel):
                    text_features = self.trainer.model.module.encode_text(sample)
                else:
                    text_features = self.trainer.model.encode_text(sample)
            embeddings.append(text_features)
            labels.append(label)

        # Train a linear classifier on top of the embeddings
        embeddings = torch.cat(embeddings)
        labels = torch.cat(labels).type(torch.LongTensor).to(self.device)
        classifier = LinearClassifier(embeddings.size(1), 2).to(self.device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(10):
            for i in range(0, len(embeddings), 32):
                optimizer.zero_grad()
                output = classifier(embeddings[i:i+32])
                loss = criterion(output, labels[i:i+32])
                loss.backward()
                optimizer.step()
        
        # Compute the accuracy on the test set
        embeddings = []
        labels = []
        for sample, label in DataLoader(self.datasetTest, batch_size=32):
            sample = clip.tokenize(sample, truncate=True).to(self.device)
            with torch.no_grad():
                if isinstance(self.trainer.model, nn.parallel.DistributedDataParallel):
                    text_features = self.trainer.model.module.encode_text(sample)
                else:
                    text_features = self.trainer.model.encode_text(sample)
            embeddings.append(text_features)
            labels.append(label)
        embeddings = torch.cat(embeddings)
        labels = torch.cat(labels).type(torch.LongTensor).to(self.device)
        output = classifier(embeddings)
        accuracy = (output.argmax(dim=1) == labels).float().mean() * 100

        if verbose: print(f"Accuracy on SST-2: {accuracy:.2f}%")
        if self.writer is not None:
            self.writer.add_scalar("Accuracy on SST-2", accuracy, step)


class MNISTValidator():
    def __init__(self, trainer, preprocess, device, writer) -> None:
        self.mnist_classes = ["zero", "one", "two", "three", "four", "five", "six", "sevem", "eight", "nine"]

        self.mnist_templates = [
            'a bad photo of a {}.',
            'a photo of many {}.',
            'a sculpture of a {}.',
            'a photo of the hard to see {}.',
            'a low resolution photo of the {}.',
            'a rendering of a {}.',
            'graffiti of a {}.',
            'a bad photo of the {}.',
            'a cropped photo of the {}.',
            'a tattoo of a {}.',
            'the embroidered {}.',
            'a photo of a hard to see {}.',
            'a bright photo of a {}.',
            'a photo of a clean {}.',
            'a photo of a dirty {}.',
            'a dark photo of the {}.',
            'a drawing of a {}.',
            'a photo of my {}.',
            'the plastic {}.',
            'a photo of the cool {}.',
            'a close-up photo of a {}.',
            'a black and white photo of the {}.',
            'a painting of the {}.',
            'a painting of a {}.',
            'a pixelated photo of the {}.',
            'a sculpture of the {}.',
            'a bright photo of the {}.',
            'a cropped photo of a {}.',
            'a plastic {}.',
            'a photo of the dirty {}.',
            'a jpeg corrupted photo of a {}.',
            'a blurry photo of the {}.',
            'a photo of the {}.',
            'a good photo of the {}.',
            'a rendering of the {}.',
            'a {} in a video game.',
            'a photo of one {}.',
            'a doodle of a {}.',
            'a close-up photo of the {}.',
            'a photo of a {}.',
            'the origami {}.',
            'the {} in a video game.',
            'a sketch of a {}.',
            'a doodle of the {}.',
            'a origami {}.',
            'a low resolution photo of a {}.',
            'the toy {}.',
            'a rendition of the {}.',
            'a photo of the clean {}.',
            'a photo of a large {}.',
            'a rendition of a {}.',
            'a photo of a nice {}.',
            'a photo of a weird {}.',
            'a blurry photo of a {}.',
            'a cartoon {}.',
            'art of a {}.',
            'a sketch of the {}.',
            'a embroidered {}.',
            'a pixelated photo of a {}.',
            'itap of the {}.',
            'a jpeg corrupted photo of the {}.',
            'a good photo of a {}.',
            'a plushie {}.',
            'a photo of the nice {}.',
            'a photo of the small {}.',
            'a photo of the weird {}.',
            'the cartoon {}.',
            'art of the {}.',
            'a drawing of the {}.',
            'a photo of the large {}.',
            'a black and white photo of a {}.',
            'the plushie {}.',
            'a dark photo of a {}.',
            'itap of a {}.',
            'graffiti of the {}.',
            'a toy {}.',
            'itap of my {}.',
            'a photo of a cool {}.',
            'a photo of a small {}.',
            'a tattoo of the {}.',
        ]

        self.trainer = trainer
        self.device = device
        self.writer = writer

        dataset = MNIST("test", preprocess=preprocess)
        self.loader = DataLoader(dataset, batch_size=32, num_workers=0)

    def zeroshot_classifier(self, classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames, miniters=20, mininterval=50, desc="Computing MNIST classes weights"):
                texts = [template.format(classname) for template in templates] #format with class
                texts = clip.tokenize(texts).to(self.device) #tokenize
                if isinstance(self.trainer.model, nn.parallel.DistributedDataParallel):
                    class_embeddings = self.trainer.model.module.encode_text(texts)
                else:
                    class_embeddings = self.trainer.model.encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(self.device)
        return zeroshot_weights
    
    def accuracy(self, output, target, topk=(1,)):
        pred = output.topk(max(topk), 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
    

    def validate(self, step, verbose=False):
        if isinstance(self.trainer, nn.parallel.DistributedDataParallel):
            self.trainer.model.module.eval()
        else:
            self.trainer.model.eval()

        self.zeroshot_weights = self.zeroshot_classifier(self.mnist_classes, self.mnist_templates)

        with torch.no_grad():
            top1, top5, n = 0., 0., 0.
            for i, (images, target) in enumerate(tqdm(self.loader, miniters=20, mininterval=50, desc="Image net validation")):
                images = images.to(self.device)
                target = target.to(self.device)
                
                # predict
                if isinstance(self.trainer.model, nn.parallel.DistributedDataParallel):
                    image_features = self.trainer.model.module.encode_image(images)
                else:
                    image_features = self.trainer.model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = 100. * image_features @ self.zeroshot_weights

                # measure accuracy
                acc1, acc5 = self.accuracy(logits, target, topk=(1, 5))
                top1 += acc1
                top5 += acc5
                n += images.size(0)

        top1 = (top1 / n) * 100
        top5 = (top5 / n) * 100

        if verbose:
            print(f"Top-1 accuracy: {top1:.2f}%")
            print(f"Top-5 accuracy: {top5:.2f}%")

        if self.writer is not None:
            self.writer.add_scalar("mnist/Top-1 accuracy", top1, step)
            self.writer.add_scalar("mnist/Top-5 accuracy", top5, step)