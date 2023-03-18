from huggingface_hub import hf_api, ModelFilter
import transformers
from transformers import AutoTokenizer, AutoImageProcessor, AutoFeatureExtractor, AutoProcessor
import json
import time

model_names = set()
# Each auto modeling files contains multiple mappings. Let's get them in a dynamic way.
for module_name in ["modeling_auto", "modeling_tf_auto", "modeling_flax_auto"]:
    module = getattr(transformers.models.auto, module_name, None)
    if module is None:
        continue
    # all mappings in a single auto modeling file
    mapping_names = [x for x in dir(module) if x.endswith("_MAPPING_NAMES") and x.startswith("MODEL_")]
    for name in mapping_names:
        mapping = getattr(module, name)
        if mapping is not None:
            for v in mapping.values():
                if isinstance(v, (list, tuple)):
                    model_names.update(v)
                elif isinstance(v, str):
                    model_names.add(v)

model_names = sorted(model_names)
print(model_names)

models = hf_api.list_models(filter=ModelFilter(author="hf-internal-testing",))
_models = set()
for x in models:
    model = x.modelId
    org, model = model.split("/")
    if not model.startswith("tiny-random-"):
        continue
    model = model.replace("tiny-random-", "")
    if not model[0].isupper():
        continue
    if model not in model_names:
        continue
    _models.add(model)

models = sorted(_models)
sorted(models)

summary = {}
for model in models[:3]:
    content = {"tokenizer_classes": set(), "processor_classes": set(), "model_classes": set()}
    repo_id = f"hf-internal-testing/tiny-random-{model}"
    try:
        time.sleep(1)
        tokenizer_fast = AutoTokenizer.from_pretrained(repo_id)
        content["tokenizer_classes"].add(tokenizer_fast.__class__.__name__)
    except:
        pass
    try:
        time.sleep(1)
        tokenizer_slow = AutoTokenizer.from_pretrained(repo_id, use_fast=False)
        content["tokenizer_classes"].add(tokenizer_slow.__class__.__name__)
    except:
        pass
    try:
        time.sleep(1)
        img_p = AutoImageProcessor.from_pretrained(repo_id)
        content["processor_classes"].add(img_p.__class__.__name__)
    except:
        pass
    try:
        time.sleep(1)
        feat_p = AutoFeatureExtractor.from_pretrained(repo_id)
        content["processor_classes"].add(feat_p.__class__.__name__)
    except:
        pass
    # try:
    #     p = AutoProcessor.from_pretrained(repo_id)
    # except:
    #     pass
    try:
        time.sleep(1)
        model_class = getattr(transformers, model)
        m = model_class.from_pretrained(repo_id)
        content["model_classes"].add(m.__class__.__name__)
    except:
        pass
    try:
        time.sleep(1)
        model_class = getattr(transformers, f"TF{model}")
        m = model_class.from_pretrained(repo_id)
        content["model_classes"].add(m.__class__.__name__)
    except:
        pass
    content["tokenizer_classes"] = sorted(content["tokenizer_classes"])
    content["processor_classes"] = sorted(content["processor_classes"])
    content["model_classes"] = sorted(content["model_classes"])

    summary[model] = content
    with open("model_summary.json", "w") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=4)

exit(0)

import json

from create_dummy_models import create_tiny_models


if __name__ == "__main__":
    with open("tests/utils/tiny_model_summary.json") as fp:
        tiny_model_info = json.load(fp)

    tiny_models_on_hub = set()
    for name in tiny_model_info:
        tiny_models_on_hub.update(tiny_model_info[name]["model_classes"])
    existing_model_classes = sorted(tiny_models_on_hub)

    output_path = "tiny_models"
    all = True
    model_types = None
    models_to_skip = existing_model_classes
    no_check = True
    upload = False
    organization = "hf-internal-testing"

    create_tiny_models(
        output_path,
        all,
        model_types,
        models_to_skip,
        no_check,
        upload,
        organization,
    )
