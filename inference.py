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
    scores = []
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
            raw_scores = torch.concat(outputs.scores)
            final_score = torch.mean(torch.max(raw_scores, dim=1).values)
            scores.append(final_score.cpu().detach().item())
            if opt.write_crossattention_scores:
                crossattention_scores = model.get_crossattention_scores(context_mask.cuda())

            for k, o in enumerate(outputs.sequences):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                answers.append(ans)
                total += 1

    return answers, total, scores


def inference(examples, opt, model=None):
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

    if not model:
        model_class = src.model.FiDT5
        model = model_class.from_pretrained(opt.model_path)
        model = model.to(opt.device)

    answers, _, scores = evaluate(model, eval_dataloader, tokenizer, opt)
    print(f"Generated answer: {answers}")
    print(f"Score: {scores}")
    return answers, scores


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_eval_options()
    opt = options.parse()
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()
    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)

    if opt.eval_data != 'none':
        eval_examples = src.data.load_data(
            opt.eval_data,
            global_rank=opt.global_rank,
            # use the global rank and world size attibutes to split the eval set on multiple gpus
            world_size=opt.world_size
        )
        example = eval_examples[0]
        print(f"Original Question {example['question']}")
        print(f"Original Answer {example['answers']}")
        example["answers"] = [""]
    else:
        example = {
            'id': "123",
            'question': "Salina </s> How can i track times for my contractors. reporting is "
                        "delayed",
            'answers': [""],
            "ctxs": [
                {
                    "title": "Add time tracking for contractors",
                    "text": "Hi Fleming, Thanks for writing in to Gusto, Nate here! I'd be glad to clarify. No worries, once you have set up your contractors with our Time Tracking feature. It will not affect your free trial months whatsoever. If you wish to proceed with this, you may follow the steps below: Go to the Time tracking section of your account. If you’re brand new to Time tracking, click the Get started button. You’ll be walked through the remainder of the setup steps here. Click the Settings tab. Scroll down to the People headline. Click +Add or remove people.. Search for a name. Only employees and contractors with an email linked to their Gusto account appear here. To add an email to someone's account, go to the People tab. Select the person. Click Save. Contractors added will be able to track their hours worked right from their Gusto account. Once hours have been reported, you can sync your contractor hours to the Contractor Payments section of your account. I hope this clears things up! If you have additional questions or concerns, or I did not touch specifically on what you're looking to have addressed, please feel free to reach back out to me. I'm always here to help you! Keep safe and be well!",
                    "id": "standardized_ticket:47:salesforce:5001M00001YJ6pCQAT",
                    "score": 1,
                },
                {
                    "title": "Track hours to a project",
                    "text": " Once your company admin has created projects , you can track your time to them. Follow the applicable instructions below based on whether you're tracking time to both payroll and projects, or just projects alone. Project time tracking is available to our Complete and Concierge customers—at this time, contractors can not track hours to projects (only employees). If you don’t have access to this feature, administrators can upgrade the plan .  ",
                    "id": "ce1efa17-5778-4789-b8fc-3f27bb12ab75",
                    "score": 1,
                },
                {
                    "title": "Add contractors to Gusto Time Tracking",
                    "text": "Domestic contractors must have an hourly rate (rather than a fixed wage) assigned to use contractor time tracking. Multiple rates for domestic contractors are not supported. Time tracking for International contractors is not supported at this time. This feature is available for Complete and Concierge plans - you can upgrade your plan at any time. Click the Time tools section and select Time tracking. If you’re brand new to Time tracking, click the Learn more button and select Get started. You’ll be walked through the remainder of the setup steps here. Click the Settings tile. Scroll down to the People headline. Click +Add or remove people. Search for a name. Only employees and contractors with an email linked to their Gusto account appear here. To add an email to someone's account, go to their profile by clicking the People section and selecting Team members. Select the person. Click Save. Contractors added will be able to track their hours worked right from their Gusto account. Once hours have been reported, you can sync your contractor hours by clicking the Payroll section and selecting Pay contractors .  ",
                    "id": "c45c1ff6-5234-4d44-b366-f4f68c5dd3c3",
                    "score": 1,
                },
                {
                    "title": "Set up Gusto Time Tracking",
                    "text": " With Gusto Time Tracking , you can track, review and approve your team’s (employee's) hours in Gusto then run payroll as usual — it’s all automatic. This feature is included in Complete or Concierge plans - you can upgrade at anytime. You can also add contractors to Gusto Time Tracking . At this time, Gusto Time Tracking is not compatible with pay schedules set up by \"employee type\".  ",
                    "id": "7287fed2-1071-4eb8-ab14-36c8a0dd92b4",
                    "score": 1,
                },
                {
                    "title": "Sync contractor hours to contractor payments",
                    "text": " Click the Time tools section and select Time tracking. Click the My team's hours tile. Click the For contractors tab. Choose a date range. Review total hours or click View for more details about the hours reported. If a contractor is not listed, you may need to add them to Time Tracking first. Once reviewed and approved, click sync hours to payments. This will save the hours reported by your contractors, and sync it to your Pay contractors tab. Once in the Pay contractors section of your account, you can begin processing payment(s). If you’ve previously synced hours for a time frame but don’t want to issue a payment for the whole time frame, return to the Time tracking section, update the dates to only the ones you intend to pay for, and click sync hours to payments again.  ",
                    "id": "f530a1af-0267-4a01-9bea-662c56b68668",
                    "score": 1,
                },
                {
                    "title": "Time Tracking | Internal Q&A",
                    "text": "Up to what point can employees and contractors edit their time? Employees and contractors can edit their time up until: Their timesheet is approved by an admin or manager Payroll was processed for that pay period An employee or contractor was overpaid, how do I go about adjusting their hours/pay? Advocates should follow the normal procedure of reversing the individual's pay and the admin can re-process as an off-cycle or a re-processing of a contractor's pay with the correct number of hours. How can employees or contractors add hours that were missed on a processed payment? Admins will need to process an off-cycle payroll including the missed hours or catch-up on the following payroll. Customer is being blocked when setting up the pay schedule. This means the current pay schedule is non-compliant. The pay schedule must be updated so that the pay period is in arrears, meaning the pay period end-date must occur before the processing deadline. If there are only contractors, this will not be an issue. There are two paths for updating the pay schedule to arrears: Single pay schedule: update the pay schedule to arrears (this will change the pay schedule for both hourly and salaried employees) Multiple pay schedules: create an hourly pay schedule with arrears for hourly employees only  ",
                    "id": "729bd5b9-c1c6-47e1-b7c0-24c257d21e18",
                    "score": 1,
                },
                {
                    "title": "What You Need to Know Before Hiring Contingent Workers | Gusto",
                    "text": "Contractors Contractors work for their clients for a set amount of time—for example, 15 hours per month—or for the duration of a project. The key difference between an independent contractor and an employee is that for contractors, a client has no legal right to dictate how, when, or where the work will be completed . An employer can require that their employees work at an office in-person two days per week. But a company cannot require that a contractor complete their work at a certain time of day, for example. Many contractors are location-independent for this reason. This is the main difference between contractors and employees. But there are other differences as well, which we will get into later. Notably, all contractors are contingent workers, but only some contingent workers are contractors.  ",
                    "id": "7d7dabc4-17a0-40e8-b687-7ebeb1e0cc5c",
                    "score": 1,
                },
                {
                    "title": "Review, edit, approve, and sync Gusto Time Tracking hours",
                    "text": "Using Gusto Time Tracking , admins can review, edit, approve, and sync employees’ reported hours to payroll. Managers with assigned direct reports in Gusto can also help approve hours . Review, edit, approve and sync time worked Click the Time tools section and select Time tracking. Confirm the pay period selected is the one for review, or edit from the drop-down. Employee's total hours will appear to review. Click view for additional detail about the time reported. Toggle to approved when hours are confirmed. This can be done in the top right corner of the employee pay period view, or from the main page. View edits made to an employee's hours by scrolling to the bottom of the pay period's hours and clicking the \"Version history\" drop-down. Click Sync Hours to Payroll when all hours have been confirmed and approved. You can continue to sync hours until payroll has been run. When you go to Run Payroll, employee’s synced hours will appear. If you need to edit hours from the Run Payroll screen, you can do so but they will not be updated in the time tracking log for you or the employee. Important : At this time, approving hours is purely a visual-aid to identify hours that have been approved. Unapproved hours will still sync to payroll and it is up to an admin to overwrite/correct hours that are synced. Did you know? You can also add contractors to Gusto Time Tracking and sync their hours to contractor payments .  ",
                    "id": "516ed308-2b7f-4745-87f4-68bc00ef7e19",
                    "score": 1,
                },
                {
                    "title": "What Is a Timesheet? | Gusto Small Business Resources",
                    "text": " A timesheet is a tool used to track how many hours an employee or contractor works over a: Time period;Specific project; orBoth. Gusto makes time tracking easy and efficient; check out our time-tracking tool .  ",
                    "id": "457386e7-1067-40c4-b290-df33f044068f",
                    "score": 1,
                },
                {
                    "title": "Set up Gusto Time Tracking",
                    "text": "Gusto Time Tracking is now enabled. Learn how to review, edit and approve hours before syncing them to payroll . If you have employees with multiple pay rates, review this article to understand how their hours will translate to the run payroll flow. Important : At this time, approving hours is purely a visual-aid for admins to identify hours that are approved by a manager. Unapproved hours will still sync to payroll and it is up to an admin to overwrite/correct hours that are synced.  ",
                    "id": "71a378e6-9dd3-461e-b7d2-bca887d50dbe",
                    "score": 1,
                },
                {
                    "title": "Track time by project to see workforce costs",
                    "text": "Here's how it works When you have new projects for clients, add them to Gusto by creating a project. Your team can track time by project and task, and add detailed notes. If they're tracking time for payroll as well, they'll select which projects they're working on when clocking in and out. Both non-exempt and exempt employees can include notes detailing the time spent on the task. See project insights in real-time. View total hours and filter across employee, project, and task for any date range. Costs are calculated from processed payrolls. Generate a report to view more details.  ",
                    "id": "a6142f0a-6144-417a-9b46-2fe5dc3d60c8",
                    "score": 1,
                }
            ],
        }

    inference([example], opt)
