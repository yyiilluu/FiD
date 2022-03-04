import uuid
from argparse import Namespace
from typing import List, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import src.model
import src.slurm
from inference import inference
from search import search_es, format_ticket_and_kas_into_ctxs
from server.response_trimer import trim_response

app = FastAPI(docs_url="/another_docs")
origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

opt = Namespace(answer_maxlength=-1,
                checkpoint_dir='../checkpoint',
                eval_data='none',
                eval_freq=500,
                eval_print_freq=1000,
                local_rank=-1,
                main_port=-1,
                maxload=-1,
                model_path='../checkpoint/base_exp_2epochs_v3/checkpoint/last',
                model_size='base', n_context=10,
                name='eval_test',
                no_title=False,
                per_gpu_batch_size=1,
                save_freq=5000,
                seed=0,
                text_maxlength=200,
                train_data='none',
                use_checkpoint=False,
                write_crossattention_scores=False,
                write_results=False)

src.slurm.init_distributed_mode(opt)
src.slurm.init_signal_handler()
opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)

model_class = src.model.FiDT5
model = model_class.from_pretrained(opt.model_path)
model = model.to(opt.device)


class RequestBody(BaseModel):
    ticket_subject: str
    ticket_body: str
    username: str
    is_test: bool


class ResponseBody(BaseModel):
    ticket_body: str
    context: List[Dict[str, str]]
    confidence_score: float


def construct_example(question, ctxs):
    return {
        "id": str(uuid.uuid4()),
        "question": question,
        "answers": [""],
        "ctxs": ctxs
    }


@app.get("/")
async def root():
    return {"message": "Hello World"}


dummy_ctxs = [
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
        "id": "71a378e6-9dd3-461e-b7d2-bca887d50dbe", "score": 1,

    },
    {
        "title": "Track time by project to see workforce costs",
        "text": "Here's how it works When you have new projects for clients, add them to Gusto by creating a project. Your team can track time by project and task, and add detailed notes. If they're tracking time for payroll as well, they'll select which projects they're working on when clocking in and out. Both non-exempt and exempt employees can include notes detailing the time spent on the task. See project insights in real-time. View total hours and filter across employee, project, and task for any date range. Costs are calculated from processed payrolls. Generate a report to view more details.  ",
        "id": "a6142f0a-6144-417a-9b46-2fe5dc3d60c8",
        "score": 1,
    }
]


@app.post("/predict/")
async def predict(request: RequestBody):
    if request.is_test:
        response = ResponseBody(ticket_body=request.ticket_body, context=dummy_ctxs,
                                confidence_score=0.1)
    else:
        ticket_results, kas_results = search_es(ticket_subject=request.ticket_subject,
                                                ticket_body=request.ticket_body)
        ctxs = format_ticket_and_kas_into_ctxs(ticket_results, kas_results, 3)
        question = f"{request.username.strip()} </s> {request.ticket_subject.strip()} " \
                   f"{request.ticket_body.strip()}"

        input_example = construct_example(question, ctxs)
        inferred_resp, scores = inference([input_example], opt, model=model)

        print(inferred_resp[0])
        trimmed_response = trim_response(inferred_resp[0])
        response = ResponseBody(ticket_body=trimmed_response, context=ctxs,
                                confidence_score=scores[0])

    return response
