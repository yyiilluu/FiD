# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import transformers
import numpy as np
from pathlib import Path
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler

import src.slurm
import src.util
from src.options import Options
import src.data
import src.evaluation
import src.model


def evaluate(model, dataloader, tokenizer, opt):
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    if opt.write_crossattention_scores:
        model.overwrite_forward_crossattention()
        model.reset_score_storage()
    total = 0
    if opt.write_results:
        write_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        fw = open(write_path / ('%d.txt' % opt.global_rank), 'a')

    answers = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, _, _, context_ids, context_mask) = batch

            if opt.write_crossattention_scores:
                model.reset_score_storage()

            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=100,
            )

            if opt.write_crossattention_scores:
                crossattention_scores = model.get_crossattention_scores(context_mask.cuda())

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                answers.append(ans)
                # example = dataset.data[idx[k]]
                # if 'answers' in example:
                #     score = src.evaluation.ems(ans, example['answers'])
                #     exactmatch.append(score)
                #
                # if opt.write_results:
                #     fw.write(str(example['id']) + "\t" + ans + '\n')
                # if opt.write_crossattention_scores:
                #     for j in range(context_ids.size(1)):
                #         example['ctxs'][j]['score'] = crossattention_scores[k, j].item()

                total += 1
    #         if (i + 1) % opt.eval_print_freq == 0:
    #             log = f'Process rank:{opt.global_rank}, {i + 1} / {len(dataloader)}'
    #             if len(exactmatch) == 0:
    #                 log += '| no answer to compute scores'
    #             else:
    #                 log += f' | average = {np.mean(exactmatch):.3f}'
    #             logger.warning(log)
    #
    # logger.warning(
    #     f'Process rank:{opt.global_rank}, total {total} | average = {np.mean(exactmatch):.3f}')
    # if opt.is_distributed:
    #     torch.distributed.barrier()
    # score, total = src.util.weighted_average(np.mean(exactmatch), total, opt)

    return answers, total


def inference(examples, opt):
    tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base', return_dict=False)
    collator_function = src.data.Collator(opt.text_maxlength, tokenizer)

    eval_dataset = src.data.Dataset(
        examples,
        opt.n_context,
    )

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=opt.per_gpu_batch_size,
        num_workers=20,
        collate_fn=collator_function
    )

    model_class = src.model.FiDT5
    model = model_class.from_pretrained(opt.model_path)
    model = model.to(opt.device)

    answers, total = evaluate(model, eval_dataloader, tokenizer, opt)
    print(answers)


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_eval_options()
    opt = options.parse()
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()
    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)

    if opt.eval_data:
        eval_examples = src.data.load_data(
            opt.eval_data,
            global_rank=opt.global_rank,
            # use the global rank and world size attibutes to split the eval set on multiple gpus
            world_size=opt.world_size
        )
        example = eval_examples[0]
        print(example['answers'])
        example["answers"] = [""]
    else:
        example = {
            'question': "this is the question",
            'answers': "",
            'ctxs': [],
        }

        for i, passage in enumerate([]):
            ctxs = []
            ctxs.append({
                'id': str(i),
                'title': passage,
                'text': passage,
            })
            example['ctxs'] = ctxs

    inference([example], opt)
