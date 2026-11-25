# Lab 02 — AI in Everyday Challenges: The Busy Student

**Module:** PUSL3205 — Large Language Models and Generative AI  
**Session:** Lab 02 — AI in Everyday Challenges  
**Group:** 5  
**Assigned scenario:** The Busy Student  
**Portfolio entry:** Individual reflection on a group product-design activity

## What this lab was about

This lab was different from building or training a model. The task was to think about an everyday problem, use an LLM as a partner in developing an idea, and explain the proposed product to others. Group 5 was assigned **The Busy Student**. The lab activity asked us to consider a product concept, name, key features, intended users and a visual mock-up, then explain how an LLM contributed to the process.

I found the scenario interesting because a busy student does not necessarily lack a calendar. The harder problem is deciding **what can realistically be finished with the time available**. A timetable can tell someone that an assignment is due on Friday; it cannot, by itself, decide how to divide the work around lectures, travel, group meetings and rest. That difference between *recording tasks* and *planning work* became the main focus of this reflection.

This README records my understanding of the task and develops an illustrative concept to explain it. **The concept, prompts and wireframe below are portfolio examples; they are not presented as saved transcripts, an original group-approved design or a working application.**

## Understanding the problem before proposing a solution

I considered a typical week in which a student has lectures on Monday and Tuesday, a programming assignment due Friday, and a group presentation to prepare for the following week. If the student only keeps a list of deadlines, the assignment might look manageable until the remaining free hours are counted. A last-minute group meeting or an underestimated programming task can make the plan fall apart.

A useful student planner would need to answer a few practical questions:

- What has a fixed time, and what can be moved?
- Which deadline needs attention first, considering both urgency and effort?
- How much free time is genuinely available after classes and personal commitments?
- If the plan changes, what has to be postponed?

I would not assume that every student wants more notifications. Frequent reminders could make a stressful week worse. The main goal should be to make priorities understandable, not to fill every empty hour.

## A product concept to explore: StudyFlow

For this portfolio reflection, I have used **StudyFlow** as a working name for a proposed LLM-assisted student planner. It is not a claim about the name selected during the original group session.

The intended user is a university student balancing classes, coursework, group responsibilities and time outside university. The proposed product has one simple purpose: bring commitments into one place and help the student decide what to work on next.

Rather than adding many features, I would start with four:

| Feature | What the student gets | Why it matters |
|---|---|---|
| Week view | Lectures, fixed events and deadlines together | Makes available study time visible |
| Task breakdown | A large assignment split into smaller steps | Gives a realistic starting point |
| Suggested next task | A recommendation with a short reason | Helps when several tasks compete |
| Change-of-plan support | A revised suggestion when a meeting moves | Makes the plan flexible rather than rigid |

For example, the interface might suggest: *“Work on the programming assignment for one hour today. It has the nearest deadline and still needs implementation and testing.”* I would want that recommendation to be editable. A student might know that the assignment is almost finished, or might be too tired to do productive coding at that point.

An important design decision is **not to use an LLM for everything**. Dates, deadline comparisons and timetable clashes should be handled by predictable program logic. The LLM would be more useful for breaking down tasks, explaining a suggested priority and offering alternatives in plain language. This is safer than asking a language model to remember every event or calculate every available time slot by itself.

## Where the LLM could help

I would use prompting in stages rather than accepting the first list of app features. The examples below show how I would structure that exploration; **they are illustrative prompts, not records of responses obtained during the classroom activity**.

**First prompt — understand the problem**

> A university student has lectures on two days, a programming assignment due Friday, a group presentation next week and limited study time in the evenings. Identify the planning difficulties. Do not assume every free hour is suitable for study.

This gives the LLM a specific situation rather than asking vaguely for “student app ideas.” It also forces the response to consider constraints.

**Second prompt — keep the product realistic**

> Suggest a small first version of a student planner. Choose only four features. Explain what the student should control manually and what an AI assistant could reasonably suggest.

This matters because generating more features is easy; deciding which ones actually solve the problem is harder.

**Third prompt — challenge the design**

> What could go wrong if a student follows this planner's suggested schedule? Consider underestimated task durations, unexpected group meetings, missing breaks and inaccurate deadlines. Recommend safeguards.

I find this last type of prompt particularly valuable. An LLM can sound confident while overlooking details that matter to an actual student. Asking it to criticise an idea is more useful than repeatedly asking it to make the idea sound better.

In a proper prompt experiment, I would keep the exact responses and note which suggestions were retained or rejected. Those response records are not part of the evidence currently attached to this Lab 02 folder.

## Sketching the idea

This is a **low-fidelity text mock-up made for the portfolio**, not a screenshot of a built system:

```text
+--------------------------------------------------+
| StudyFlow                         This week       |
+--------------------------------------------------+
| TODAY                                            |
| 10:00  Data Programming lecture                  |
| 14:00  Free study block                          |
|                                                  |
| UPCOMING                                         |
| Friday       Programming assignment              |
| Next week    Group presentation                  |
|                                                  |
| SUGGESTED NEXT TASK                              |
| Start assignment testing — 60 minutes            |
| Why? Friday deadline; testing still unfinished.  |
|                                                  |
| [Use suggestion]  [Change plan]  [Why this?]     |
+--------------------------------------------------+
```

I kept the suggested task close to the deadline list so the explanation is visible in context. The **Change plan** option is important: the student should never feel that an AI-generated timetable is a fixed instruction. A later prototype could show different screen sizes and allow the user to adjust study duration, but this lab portfolio does not claim such an interface was developed or tested.

## What I would check before calling the idea useful

A convincing demonstration should go beyond showing an attractive screen. I would test the concept with a sample schedule and check whether it behaves sensibly in awkward situations.

| Test scenario | What I would look for |
|---|---|
| Two assignments due on the same day | A clear priority explanation, not an arbitrary choice |
| A meeting overlaps a study block | A warning and a revised suggestion without deleting the meeting |
| A task takes longer than expected | An easy way to move unfinished work |
| A student rejects the suggested plan | No automatic change to their calendar |

These are **proposed checks**, not measured usability results. Feedback from students would be necessary before saying the product improves productivity or reduces stress.

Privacy would also matter. A student's calendar may reveal personal routines, locations or academic difficulties. Any future implementation should minimise the information sent to an LLM provider and obtain consent before connecting to external accounts.

## What I took away from the activity

The main thing I took from this exercise is that **understanding the user's problem comes before choosing an AI feature**. It would be easy to describe a planner with a chatbot, reminders, dashboards and automatic scheduling. But if the student still cannot tell which task needs attention today, those features have not solved much.

I also see prompting differently after thinking through this scenario. A good prompt should do more than request an answer. It should provide the situation, identify constraints and leave room to question the result. A suggestion that sounds reasonable in general may be unworkable when lectures, travel or group commitments are added.

Finally, this lab helped me distinguish between a **concept** and an **implemented product**. The StudyFlow example communicates a possible solution, but it does not prove that students would use it or that its schedules would be dependable. That would require an actual prototype, recorded tests and feedback. For an LLM-based product, being able to explain those limitations is as important as describing its benefits.

## Evidence and scope of this submission

This is an **individual learning-and-reflection README** for the Group 5 scenario. It covers the problem, a proposed concept, target users, key features, a conceptual mock-up, possible LLM prompts and my critical reflection.

The original group presentation, classroom prompt-response history and group-approved mock-up are **not included in this folder**. I have not presented the new illustrative examples as historical evidence. No working application or user-study result is claimed. If authentic group artifacts are available, they can be linked here and discussed separately.

## Conclusion

The Busy Student scenario made me think beyond the idea of an AI that simply reminds students about deadlines. The more useful challenge is helping someone make a realistic decision when several responsibilities compete for limited time. An LLM could help explain priorities and adjust a study plan, but reliable calendar logic and final decisions should remain under the student's control.

The value of this lab was learning how to question an initial idea: who needs it, which features are worth keeping, where the model might be wrong, and what evidence would be required before calling the solution successful.

## References and AI assistance

- PUSL3205, *Large Language Models and Generative AI*: Lab 02 — AI in Everyday Challenges (module laboratory sheet).
- PUSL3205, *Enhanced Assessment Brief 2026–27* (module assessment guidance).

