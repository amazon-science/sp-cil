# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import yaml
import string


def get_yaml_list(strs, indents=0):
    return "\n".join(" " * indents + "- " + s for s in strs)


templates = [
    "imnet_delta_layer4_resnet10_500_tmpl.yml",
    "imnet_delta_layer4_resnet18_20shot_500_tmpl.yml",  # Note that this is 20-shot to compare with prior work
    "imnet_delta_fc_resnet10_500_tmpl.yml",
    "imnet_delta_fc_resnet18_20shot_500_tmpl.yml",  # Note that this is 20-shot to compare with prior work
]
model_templates = [
    (
        "runs/imnet_base_resnet10_500/resnet10_500_norm4/ep-90_model.pkl",
        "runs/imnet_novel_resnet10_layer4_{n_novel_cls}/FT_resnet10_500_norm4_layer4_all_s{n_novel_cls}/ep-30_model.pkl",
    ),
    (
        "runs/imnet_base_resnet18_500/resnet18_500_norm4/ep-90_model.pkl",
        "runs/imnet_novel_resnet18_layer4_{n_novel_cls}/FT_resnet18_500_norm4_layer4_all_s{n_novel_cls}/ep-30_model.pkl",
    ),
    (
        "runs/imnet_base_resnet10_500/resnet10_500_norm4/ep-90_model.pkl",
        "runs/imnet_novel_resnet10_fc_{n_novel_cls}/FT_resnet10_500_norm4_fc_all_s{n_novel_cls}/ep-30_model.pkl",
    ),
    (
        "runs/imnet_base_resnet18_500/resnet18_500_norm4/ep-90_model.pkl",
        "runs/imnet_novel_resnet18_fc_{n_novel_cls}/FT_resnet18_500_norm4_fc_all_s{n_novel_cls}/ep-30_model.pkl",
    ),
]

for template, (base_model, model_template) in zip(templates, model_templates):
    with open(template, "rt") as fp:
        cfg = fp.read()

    cfg_tmpl = string.Template(cfg)
    for n in range(1, 11):
        with open(template.replace("tmpl", "split%d" % n), "wt") as fp:
            models_3_indents = """
            - {base_model}
""".format(
                base_model=base_model
            ) + get_yaml_list(
                [model_template.format(n_novel_cls=(x * 50)) for x in range(1, n + 1)],
                12,
            )
            fp.write(
                cfg_tmpl.substitute(
                    n_novel=n,
                    n_novel_plus_1=n + 1,
                    n_novel_times_50=n * 50,
                    n_base_cls_3_indents="\n" + get_yaml_list([str(x) for x in [500] + [50] * (n - 1)], 12),
                    models_3_indents=models_3_indents,
                )
            )
